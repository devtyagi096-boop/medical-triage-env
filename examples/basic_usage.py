"""
Basic usage example for the Medical Triage Environment
"""

from src.env.medical_triage_env import MedicalTriageEnv

env = MedicalTriageEnv(max_patients_per_episode=10, seed=42)
obs = env.reset()

print("=== Medical Triage Environment - Basic Usage ===\n")
print(f"First patient:")
print(f"  Severity : {obs['severity']}/5")
print(f"  HR       : {obs['heart_rate'][0]:.0f} bpm")
print(f"  SpO2     : {obs['oxygen_saturation'][0]:.0f}%")
print(f"  SBP      : {obs['systolic_bp'][0]:.0f} mmHg")

done = False
total_reward = 0.0
step = 0

while not done:
    # Simple rule-based agent
    spo2     = float(obs['oxygen_saturation'][0])
    hr       = float(obs['heart_rate'][0])
    sbp      = float(obs['systolic_bp'][0])
    severity = int(obs['severity'])

    if spo2 < 85 or hr > 150 or hr < 45 or sbp < 85 or severity >= 4:
        action = 0
    elif severity == 3:
        action = 1
    elif severity == 2:
        action = 2
    else:
        action = 3

    obs, reward, done, info = env.step(action)
    total_reward += reward
    step += 1
    print(f"Step {step:2d}: action={action} reward={reward:7.2f} accuracy={info['accuracy']:.1%}")

print(f"\nTotal reward : {total_reward:.2f}")
print(f"Final accuracy: {info['accuracy']:.1%}")
