# baseline.py
import os
from openai import OpenAI
from environment import MedicalTriageEnv
from models import Action
from grader import grade_task
import json

def run_baseline(task: str = "medium", max_steps: int = 100, api_key: str = None) -> float:
    """
    Run baseline agent using Groq API (OpenAI-compatible)
    """
    # Use Groq with OpenAI client
    client = OpenAI(
        api_key=api_key or os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1"
    )
    
    env = MedicalTriageEnv(task=task, max_steps=max_steps, seed=42)
    
    obs, state = env.reset()
    done = False
    total_reward = 0.0
    
    system_prompt = get_system_prompt()
    conversation_history = [
        {"role": "system", "content": system_prompt}
    ]
    
    step = 0
    while not done and step < max_steps:
        # Create observation message
        obs_text = format_observation(obs)
        conversation_history.append({"role": "user", "content": obs_text})
        
        # Get action from LLM
        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",  # NEW - current model
                messages=conversation_history,
                temperature=0.7,
                max_tokens=500
            )
            
            action_text = response.choices[0].message.content
            conversation_history.append({"role": "assistant", "content": action_text})
            
            # Parse action
            action = parse_action(action_text, obs)
            
            # Execute action
            obs, reward, done, info, state = env.step(action)
            total_reward += reward.total
            
            step += 1
            
            # Keep conversation manageable
            if len(conversation_history) > 10:
                conversation_history = [conversation_history[0]] + conversation_history[-8:]
                
        except Exception as e:
            print(f"Error at step {step}: {e}")
            # Default to wait action
            action = Action(action_type="wait")
            obs, reward, done, info, state = env.step(action)
            step += 1
    
    # Grade performance
    score = grade_task(env, task)
    metrics = env.get_episode_metrics()
    
    print(f"\n=== Task: {task.upper()} ===")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Score: {score:.3f}")
    print(f"Patients Treated: {metrics['patients_treated']}")
    print(f"Avg Wait Time: {metrics['avg_wait_time']:.1f} min")
    print(f"Critical Wait Time: {metrics['critical_patient_wait']:.1f} min")
    print(f"Triage Accuracy: {metrics['triage_accuracy']:.2%}")
    
    return score

def get_system_prompt() -> str:
    return """You are an AI triage nurse in an emergency department. Your job is to:

1. Assess incoming patients based on their vital signs, symptoms, and medical history
2. Assign appropriate triage priority levels (1=Critical, 2=Emergency, 3=Urgent, 4=Less Urgent, 5=Non-urgent)
3. Manage limited resources (beds, staff) to minimize patient harm

PRIORITY GUIDELINES:
- Priority 1 (Critical): Life-threatening, needs immediate care (e.g., cardiac arrest, severe trauma, stroke)
- Priority 2 (Emergency): Serious conditions, needs care within 15 min (e.g., severe pain, high fever, serious injuries)
- Priority 3 (Urgent): Moderate conditions, can wait 30-60 min (e.g., minor fractures, moderate pain)
- Priority 4 (Less Urgent): Minor conditions, can wait 1-2 hours (e.g., minor lacerations, mild infections)
- Priority 5 (Non-urgent): Very minor, can wait 2+ hours (e.g., cold symptoms, prescription refills)

RED FLAGS (Priority 1 or 2):
- Chest pain + abnormal vitals
- O2 saturation < 90%
- Severe difficulty breathing
- Altered mental status
- Very high/low blood pressure
- Heart rate > 120 or < 50

Respond with a JSON object:
{
  "action_type": "triage",
  "patient_id": "P0001",
  "priority_level": 1,
  "reasoning": "brief explanation"
}"""

def format_observation(obs) -> str:
    """Format observation for LLM"""
    text = f"=== Emergency Department Status (Time: {obs.current_time:.1f} min) ===\n"
    text += f"Available Beds: {obs.available_beds}/{obs.total_beds}\n"
    text += f"Staff Available: {obs.staff_available}\n\n"
    
    if obs.waiting_patients:
        text += "WAITING PATIENTS (not yet triaged):\n"
        for p in obs.waiting_patients:
            wait_time = obs.current_time - p.arrival_time
            text += f"\n{p.id} (Age {p.age}, waiting {wait_time:.1f} min):\n"
            text += f"  Complaint: {p.chief_complaint}\n"
            text += f"  Vitals: "
            if p.vital_signs.heart_rate:
                text += f"HR={p.vital_signs.heart_rate}, "
            if p.vital_signs.blood_pressure_systolic:
                text += f"BP={p.vital_signs.blood_pressure_systolic}/{p.vital_signs.blood_pressure_diastolic}, "
            if p.vital_signs.respiratory_rate:
                text += f"RR={p.vital_signs.respiratory_rate}, "
            if p.vital_signs.temperature:
                text += f"Temp={p.vital_signs.temperature:.1f}°C, "
            if p.vital_signs.oxygen_saturation:
                text += f"SpO2={p.vital_signs.oxygen_saturation}%, "
            if p.vital_signs.pain_level:
                text += f"Pain={p.vital_signs.pain_level}/10"
            text += "\n"
            if p.medical_history:
                text += f"  History: {', '.join(p.medical_history)}\n"
    
    if obs.triaged_patients:
        text += f"\nTRIAGED PATIENTS ({len(obs.triaged_patients)} waiting for beds):\n"
        for p in obs.triaged_patients:
            text += f"  {p.id}: Priority {p.assigned_priority}, {p.chief_complaint}\n"
    
    if not obs.waiting_patients:
        text += "\nNo patients waiting for triage.\n"
    
    text += "\nWhat action do you take?"
    return text

def parse_action(action_text: str, obs) -> Action:
    """Parse LLM response into Action"""
    try:
        # Try to find JSON in response
        start = action_text.find('{')
        end = action_text.rfind('}') + 1
        if start >= 0 and end > start:
            action_json = json.loads(action_text[start:end])
            return Action(**action_json)
    except:
        pass
    
    # Fallback: triage first waiting patient with medium priority
    if obs.waiting_patients:
        return Action(
            action_type="triage",
            patient_id=obs.waiting_patients[0].id,
            priority_level=3,
            reasoning="Fallback action"
        )
    else:
        return Action(action_type="wait")

if __name__ == "__main__":
    print("Medical Triage Baseline Evaluation")
    print("=" * 50)
    
    # Check for API key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("\n⚠️  WARNING: GROQ_API_KEY not found in environment variables")
        print("Set it with: export GROQ_API_KEY=your_key_here")
        print("\nRunning with dummy scores for demonstration...\n")
        
        # Show what would happen
        for task in ["easy", "medium", "hard"]:
            print(f"\n=== Task: {task.upper()} ===")
            print("(Skipped - no API key)")
    else:
        # Run all three tasks
        scores = {}
        for task in ["easy", "medium", "hard"]:
            scores[task] = run_baseline(task=task, max_steps=100, api_key=api_key)
        
        print("\n" + "=" * 50)
        print("FINAL SCORES")
        print("=" * 50)
        for task, score in scores.items():
            print(f"{task.upper():8s}: {score:.3f}")