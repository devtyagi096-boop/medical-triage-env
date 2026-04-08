import os
import json
import time
import sys
from typing import Any, Dict, List

from openai import OpenAI

from environment import MedicalTriageEnv
from models import Action
from grader import grade_task


TASKS = ["easy", "medium", "hard"]
MAX_STEPS = 100
SEED = 42


def emit(tag: str, payload: Dict[str, Any]) -> None:
    """Emit structured log line to stdout with tag clearly visible."""
    # Print plain-text tag line first (for simple grep-based validators)
    parts = " ".join(f"{k}={v}" for k, v in payload.items() if not isinstance(v, (dict, list)))
    print(f"{tag} {parts}", flush=True)
    # Also print full JSON for structured validators
    record = {"tag": tag, **payload}
    print(json.dumps(record, ensure_ascii=False, default=str), flush=True)


def make_client():
    api_base_url = os.environ.get("API_BASE_URL", "")
    model_name = os.environ.get("MODEL_NAME", "")
    hf_token = os.environ.get("HF_TOKEN", "")

    if not api_base_url:
        raise RuntimeError("Missing required environment variable: API_BASE_URL")
    if not model_name:
        raise RuntimeError("Missing required environment variable: MODEL_NAME")
    if not hf_token:
        raise RuntimeError("Missing required environment variable: HF_TOKEN")

    client = OpenAI(base_url=api_base_url, api_key=hf_token)
    return client, model_name


def system_prompt() -> str:
    return (
        "You are an AI triage nurse in an emergency department.\n"
        "Assign triage priority:\n"
        "1 = Critical\n"
        "2 = Emergency\n"
        "3 = Urgent\n"
        "4 = Less Urgent\n"
        "5 = Non-urgent\n\n"
        "Return ONLY valid JSON:\n"
        "{\n"
        '  "action_type": "triage",\n'
        '  "patient_id": "P0001",\n'
        '  "priority_level": 2,\n'
        '  "reasoning": "brief explanation"\n'
        "}"
    )


def format_observation(obs) -> str:
    lines: List[str] = []
    lines.append(f"Time: {obs.current_time:.1f} min")
    lines.append(f"Beds: {obs.available_beds}/{obs.total_beds}")
    lines.append(f"Staff: {obs.staff_available}")
    lines.append("")

    if obs.waiting_patients:
        lines.append("WAITING PATIENTS:")
        for p in obs.waiting_patients:
            vs = p.vital_signs
            lines.append(f"- id={p.id}, age={p.age}, complaint={p.chief_complaint}")
            lines.append(
                f"  vitals=HR={vs.heart_rate}, "
                f"BP={vs.blood_pressure_systolic}/{vs.blood_pressure_diastolic}, "
                f"RR={vs.respiratory_rate}, Temp={vs.temperature}, "
                f"SpO2={vs.oxygen_saturation}, Pain={vs.pain_level}"
            )
            if p.medical_history:
                lines.append(f"  history={', '.join(p.medical_history)}")
    else:
        lines.append("No waiting patients.")

    if obs.triaged_patients:
        lines.append("")
        lines.append("TRIAGED PATIENTS:")
        for p in obs.triaged_patients:
            lines.append(
                f"- id={p.id}, priority={p.assigned_priority}, complaint={p.chief_complaint}"
            )

    return "\n".join(lines)


def parse_action(raw_text: str, obs) -> Action:
    try:
        start = raw_text.find("{")
        end = raw_text.rfind("}") + 1
        if start >= 0 and end > start:
            payload = json.loads(raw_text[start:end])
            return Action(**payload)
    except Exception:
        pass

    if obs.waiting_patients:
        return Action(
            action_type="triage",
            patient_id=obs.waiting_patients[0].id,
            priority_level=3,
            reasoning="fallback_action_due_to_parse_failure"
        )
    return Action(action_type="wait", reasoning="no_waiting_patients")


def run_single_task(task: str, client: OpenAI, model_name: str) -> Dict[str, Any]:
    env = MedicalTriageEnv(task=task, max_steps=MAX_STEPS, seed=SEED)
    obs, state = env.reset()
    done = False
    total_reward = 0.0
    step_idx = 0

    messages = [{"role": "system", "content": system_prompt()}]

    emit("[START]", {
        "task": task,
        "seed": SEED,
        "max_steps": MAX_STEPS,
        "timestamp": round(time.time(), 3)
    })

    while not done and step_idx < MAX_STEPS:
        user_prompt = format_observation(obs)
        messages.append({"role": "user", "content": user_prompt})

        model_error = None
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=300
            )
            raw_content = response.choices[0].message.content or ""
            action = parse_action(raw_content, obs)
            messages.append({"role": "assistant", "content": raw_content})
        except Exception as exc:
            model_error = str(exc)
            action = Action(action_type="wait", reasoning="fallback_due_to_model_error")

        obs, reward, done, info, state = env.step(action)
        total_reward += reward.total
        step_idx += 1

        emit("[STEP]", {
            "task": task,
            "step": step_idx,
            "action_type": action.action_type,
            "patient_id": action.patient_id or "null",
            "priority_level": action.priority_level or "null",
            "reward": round(reward.total, 4),
            "done": done,
            "model_error": model_error or "null"
        })

        if len(messages) > 10:
            messages = [messages[0]] + messages[-8:]

    score = grade_task(env, task)
    metrics = env.get_episode_metrics()

    emit("[END]", {
        "task": task,
        "score": round(score, 4),
        "total_reward": round(total_reward, 4),
        "steps_completed": step_idx,
        "patients_treated": metrics.get("patients_treated", 0),
        "triage_accuracy": round(metrics.get("triage_accuracy", 0.0), 4)
    })

    return {
        "task": task,
        "score": score,
        "metrics": metrics,
        "total_reward": total_reward,
        "steps_completed": step_idx
    }


def main():
    started_at = time.time()

    try:
        client, model_name = make_client()
    except RuntimeError as e:
        print(f"[ERROR] {e}", flush=True)
        sys.exit(1)

    all_results = []
    for task in TASKS:
        try:
            result = run_single_task(task, client, model_name)
            all_results.append(result)
        except Exception as e:
            print(f"[ERROR] task={task} error={e}", flush=True)
            # Still emit [END] so validator sees the tag
            emit("[END]", {
                "task": task,
                "score": 0.0,
                "total_reward": 0.0,
                "steps_completed": 0,
                "error": str(e)
            })

    print(
        f"[SUMMARY] tasks={len(all_results)} "
        f"runtime={round(time.time() - started_at, 2)}s",
        flush=True
    )


if __name__ == "__main__":
    main()
