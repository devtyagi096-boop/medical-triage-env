import os
import time
from openai import OpenAI

from environment import MedicalTriageEnv
from models import Action
from grader import grade_task

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise RuntimeError("Missing HF_TOKEN")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

TASKS = ["easy", "medium", "hard"]
MAX_STEPS = 100
SEED = 42
BENCHMARK = "medical-triage"


def log_start(task):
    print(f"[START] task={task} env={BENCHMARK} model={MODEL_NAME}", flush=True)


def log_step(step, action_str, reward, done, error=None):
    err = error if error else "null"
    print(
        f"[STEP] step={step} action={action_str} reward={reward:.2f} done={str(done).lower()} error={err}",
        flush=True
    )


def log_end(success, steps, score, rewards_list):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards_list)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True
    )


def get_system_prompt():
    return (
        "You are an AI triage nurse in an emergency department.\n"
        "Assign triage priority:\n"
        "1 = Critical, 2 = Emergency, 3 = Urgent, 4 = Less Urgent, 5 = Non-urgent\n\n"
        "Return ONLY valid JSON:\n"
        '{"action_type": "triage", "patient_id": "P0001", "priority_level": 2, '
        '"reasoning": "brief explanation"}\n'
    )


def format_observation(obs):
    lines = []
    lines.append(f"Time: {obs.current_time:.1f} min")
    lines.append(f"Beds: {obs.available_beds}/{obs.total_beds}")
    lines.append(f"Staff: {obs.staff_available}")

    if obs.waiting_patients:
        lines.append("WAITING PATIENTS:")
        for p in obs.waiting_patients:
            vs = p.vital_signs
            lines.append(f"- id={p.id}, age={p.age}, complaint={p.chief_complaint}")
            lines.append(
                f"  vitals=HR={vs.heart_rate}, BP={vs.blood_pressure_systolic}/{vs.blood_pressure_diastolic}, "
                f"RR={vs.respiratory_rate}, Temp={vs.temperature}, SpO2={vs.oxygen_saturation}, Pain={vs.pain_level}"
            )
            if p.medical_history:
                lines.append(f"  history={', '.join(p.medical_history)}")
    else:
        lines.append("No waiting patients.")

    return "\n".join(lines)


def parse_action(raw_text, obs):
    import json
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
            reasoning="fallback"
        )
    return Action(action_type="wait", reasoning="no_patients")


def action_to_str(action):
    if action.action_type == "triage":
        return f"triage({action.patient_id},{action.priority_level})"
    elif action.action_type == "reassess":
        return f"reassess({action.patient_id},{action.priority_level})"
    elif action.action_type == "call_specialist":
        return f"call_specialist({action.patient_id},{action.specialist_type})"
    else:
        return "wait()"


def run_task(task):
    env = MedicalTriageEnv(task=task, max_steps=MAX_STEPS, seed=SEED)
    obs, state = env.reset()
    done = False
    step_idx = 0
    rewards_list = []

    messages = [{"role": "system", "content": get_system_prompt()}]

    log_start(task)

    while not done and step_idx < MAX_STEPS:
        user_prompt = format_observation(obs)
        messages.append({"role": "user", "content": user_prompt})

        error_msg = None
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.0,
                max_tokens=300
            )
            raw = response.choices[0].message.content or ""
            action = parse_action(raw, obs)
            messages.append({"role": "assistant", "content": raw})
        except Exception as exc:
            error_msg = str(exc)
            action = Action(action_type="wait", reasoning="model_error")

        obs, reward, done, info, state = env.step(action)
        step_idx += 1
        rewards_list.append(reward.total)

        log_step(step_idx, action_to_str(action), reward.total, done, error_msg)

        if len(messages) > 10:
            messages = [messages[0]] + messages[-8:]

    score = grade_task(env, task)
    success = score > 0.5

    log_end(success, step_idx, score, rewards_list)
    return score


if __name__ == "__main__":
    for task in TASKS:
        run_task(task)