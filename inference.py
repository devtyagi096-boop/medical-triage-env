import os
import json
import time
from openai import OpenAI

from environment import MedicalTriageEnv
from models import Action
from grader import grade_task


def log_start(task: str):
    print(json.dumps({
        "tag": "[START]",
        "task": task,
        "timestamp": time.time()
    }), flush=True)


def log_step(task: str, step: int, action: dict, reward: float, done: bool):
    print(json.dumps({
        "tag": "[STEP]",
        "task": task,
        "step": step,
        "action": action,
        "reward": reward,
        "done": done
    }), flush=True)


def log_end(task: str, score: float, metrics: dict):
    print(json.dumps({
        "tag": "[END]",
        "task": task,
        "score": score,
        "metrics": metrics,
        "timestamp": time.time()
    }), flush=True)


def get_client():
    api_base = os.environ["API_BASE_URL"]
    model_name = os.environ["MODEL_NAME"]
    hf_token = os.environ["HF_TOKEN"]
    client = OpenAI(base_url=api_base, api_key=hf_token)
    return client, model_name


def get_system_prompt():
    return """You are an AI triage nurse in an emergency department.

Assign triage priority:
1 = Critical
2 = Emergency
3 = Urgent
4 = Less Urgent
5 = Non-urgent

Return JSON:
{
  "action_type": "triage",
  "patient_id": "P0001",
  "priority_level": 1,
  "reasoning": "brief explanation"
}
"""


def format_observation(obs):
    text = f"Time: {obs.current_time:.1f} min\n"
    text += f"Beds: {obs.available_beds}/{obs.total_beds}\n"
    text += f"Staff: {obs.staff_available}\n\n"

    if obs.waiting_patients:
        text += "WAITING PATIENTS:\n"
        for p in obs.waiting_patients:
            text += f"- {p.id}, age {p.age}, complaint: {p.chief_complaint}\n"
            vs = p.vital_signs
            text += f"  vitals: HR={vs.heart_rate}, BP={vs.blood_pressure_systolic}/{vs.blood_pressure_diastolic}, RR={vs.respiratory_rate}, Temp={vs.temperature}, SpO2={vs.oxygen_saturation}, Pain={vs.pain_level}\n"
            if p.medical_history:
                text += f"  history: {', '.join(p.medical_history)}\n"
    else:
        text += "No waiting patients.\n"

    return text


def parse_action(text, obs):
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            payload = json.loads(text[start:end])
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
    return Action(action_type="wait")


def run_task(task: str):
    client, model_name = get_client()
    env = MedicalTriageEnv(task=task, max_steps=100, seed=42)

    obs, state = env.reset()
    done = False
    step_idx = 0
    total_reward = 0.0

    messages = [{"role": "system", "content": get_system_prompt()}]

    log_start(task)

    while not done and step_idx < 100:
        user_msg = format_observation(obs)
        messages.append({"role": "user", "content": user_msg})

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=300
            )
            content = response.choices[0].message.content or ""
            messages.append({"role": "assistant", "content": content})
            action = parse_action(content, obs)
        except Exception:
            action = Action(action_type="wait")

        obs, reward, done, info, state = env.step(action)
        total_reward += reward.total
        step_idx += 1

        log_step(
            task=task,
            step=step_idx,
            action=action.model_dump(),
            reward=reward.total,
            done=done
        )

        if len(messages) > 10:
            messages = [messages[0]] + messages[-8:]

    score = grade_task(env, task)
    metrics = env.get_episode_metrics()

    log_end(task, score, metrics)

    return {
        "task": task,
        "score": score,
        "metrics": metrics,
        "total_reward": total_reward
    }


def main():
    results = []
    for task in ["easy", "medium", "hard"]:
        results.append(run_task(task))

    print(json.dumps({
        "tag": "[END]",
        "summary": results
    }), flush=True)


if __name__ == "__main__":
    main()