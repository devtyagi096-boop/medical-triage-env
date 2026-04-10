"""
Evaluation suite — runs multiple episodes and reports metrics.
"""

from src.env.medical_triage_env import MedicalTriageEnv
from examples.rule_based_agent import rule_based_action


def evaluate(episodes: int = 10, max_steps: int = 30):
    env = MedicalTriageEnv(max_patients_per_episode=max_steps, seed=0)
    all_rewards, all_accuracy, all_deteriorations = [], [], []

    for ep in range(episodes):
        obs = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            obs, reward, done, info = env.step(rule_based_action(obs))
            ep_reward += reward

        all_rewards.append(ep_reward)
        all_accuracy.append(info['accuracy'])
        all_deteriorations.append(info['deteriorations'])

    print("=== Evaluation Results ===")
    print(f"Episodes         : {episodes}")
    print(f"Avg Total Reward : {sum(all_rewards)/len(all_rewards):.2f}")
    print(f"Avg Accuracy     : {sum(all_accuracy)/len(all_accuracy):.1%}")
    print(f"Avg Deteriorations: {sum(all_deteriorations)/len(all_deteriorations):.1f}")


if __name__ == "__main__":
    evaluate()
