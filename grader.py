# grader.py
from typing import Dict
from environment import MedicalTriageEnv

def grade_task(env: MedicalTriageEnv, task: str) -> float:
    """
    Grade agent performance on a task.
    Returns score between 0.0 and 1.0
    """
    metrics = env.get_episode_metrics()
    
    if task == "easy":
        return _grade_easy(metrics)
    elif task == "medium":
        return _grade_medium(metrics)
    elif task == "hard":
        return _grade_hard(metrics)
    else:
        return 0.0

def _grade_easy(metrics: Dict) -> float:
    """
    Easy task: Handle basic triage with low patient volume
    Success criteria:
    - Treat at least 10 patients
    - Average wait time < 20 minutes
    - Triage accuracy > 70%
    """
    score = 0.0
    
    # Patients treated (30 points)
    if metrics["patients_treated"] >= 10:
        score += 0.3
    else:
        score += (metrics["patients_treated"] / 10) * 0.3
    
    # Wait time (30 points)
    if metrics["avg_wait_time"] < 20:
        score += 0.3
    elif metrics["avg_wait_time"] < 40:
        score += 0.3 * (1 - (metrics["avg_wait_time"] - 20) / 20)
    
    # Triage accuracy (40 points)
    score += metrics["triage_accuracy"] * 0.4
    
    return min(1.0, max(0.0, score))

def _grade_medium(metrics: Dict) -> float:
    """
    Medium task: Handle moderate patient volume with resource constraints
    Success criteria:
    - Treat at least 18 patients
    - Critical patient wait time < 15 minutes
    - Triage accuracy > 75%
    - Average wait time < 25 minutes
    """
    score = 0.0
    
    # Patients treated (25 points)
    if metrics["patients_treated"] >= 18:
        score += 0.25
    else:
        score += (metrics["patients_treated"] / 18) * 0.25
    
    # Critical patient wait (30 points)
    if metrics["critical_patient_wait"] < 15:
        score += 0.3
    elif metrics["critical_patient_wait"] < 30:
        score += 0.3 * (1 - (metrics["critical_patient_wait"] - 15) / 15)
    
    # Triage accuracy (30 points)
    if metrics["triage_accuracy"] >= 0.75:
        score += 0.3
    else:
        score += metrics["triage_accuracy"] * 0.3 / 0.75
    
    # Average wait time (15 points)
    if metrics["avg_wait_time"] < 25:
        score += 0.15
    elif metrics["avg_wait_time"] < 50:
        score += 0.15 * (1 - (metrics["avg_wait_time"] - 25) / 25)
    
    return min(1.0, max(0.0, score))

def _grade_hard(metrics: Dict) -> float:
    """
    Hard task: High patient volume, limited resources, complex cases
    Success criteria:
    - Treat at least 25 patients
    - Critical patient wait time < 10 minutes
    - Triage accuracy > 80%
    - Average wait time < 30 minutes
    """
    score = 0.0
    
    # Patients treated (20 points)
    if metrics["patients_treated"] >= 25:
        score += 0.2
    else:
        score += (metrics["patients_treated"] / 25) * 0.2
    
    # Critical patient wait (35 points)
    if metrics["critical_patient_wait"] < 10:
        score += 0.35
    elif metrics["critical_patient_wait"] < 20:
        score += 0.35 * (1 - (metrics["critical_patient_wait"] - 10) / 10)
    
    # Triage accuracy (30 points)
    if metrics["triage_accuracy"] >= 0.80:
        score += 0.3
    else:
        score += metrics["triage_accuracy"] * 0.3 / 0.80
    
    # Average wait time (15 points)
    if metrics["avg_wait_time"] < 30:
        score += 0.15
    elif metrics["avg_wait_time"] < 60:
        score += 0.15 * (1 - (metrics["avg_wait_time"] - 30) / 30)
    
    return min(1.0, max(0.0, score))


if __name__ == "__main__":
    # Test the grader
    print("Testing grader...")
    
    for task in ["easy", "medium", "hard"]:
        env = MedicalTriageEnv(task=task, seed=42)
        obs, state = env.reset()
        
        # Run a few random steps
        from models import Action
        import random
        
        for _ in range(20):
            if obs.waiting_patients:
                patient = random.choice(obs.waiting_patients)
                action = Action(
                    action_type="triage",
                    patient_id=patient.id,
                    priority_level=random.randint(1, 5)
                )
                obs, reward, done, info, state = env.step(action)
                if done:
                    break
        
        score = grade_task(env, task)
        metrics = env.get_episode_metrics()
        print(f"\n{task.upper()} Task:")
        print(f"  Patients treated: {metrics['patients_treated']}")
        print(f"  Avg wait: {metrics['avg_wait_time']:.1f} min")
        print(f"  Accuracy: {metrics['triage_accuracy']:.1%}")
        print(f"  Score: {score:.3f}")