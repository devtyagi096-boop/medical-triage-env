"""
Rule-based agent baseline — no LLM required.
Useful for benchmarking and testing the environment.
"""

from src.env.medical_triage_env import MedicalTriageEnv


def rule_based_action(obs: dict) -> int:
    spo2     = float(obs['oxygen_saturation'][0])
    hr       = float(obs['heart_rate'][0])
    sbp      = float(obs['systolic_bp'][0])
    severity = int(obs['severity'])

    if spo2 < 85 or hr > 150 or hr < 45 or sbp < 85 or severity >= 4:
        return 0
    if severity == 3:
        return 1
    if severity == 2:
        return 2
    return 3


def run_rule_based(episodes: int = 5, max_steps: int = 30):
    env = MedicalTriageEnv(max_patients_per_episode=max_steps, seed=42)
    results = []

    for ep in range(1, episodes + 1):
        obs = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action = rule_based_action(obs)
            obs, reward, done, info = env.step(action)
            total_reward += reward

        results.append({
            'episode':      ep,
            'total_reward': total_reward,
            'accuracy':     info['accuracy'],
        })
        print(f"Episode {ep}: reward={total_reward:.2f}  accuracy={info['accuracy']:.1%}")

    avg_reward   = sum(r['total_reward'] for r in results) / len(results)
    avg_accuracy = sum(r['accuracy']     for r in results) / len(results)
    print(f"\nAverage reward  : {avg_reward:.2f}")
    print(f"Average accuracy: {avg_accuracy:.1%}")


if __name__ == "__main__":
    run_rule_based()
