# play_game.py
from environment import MedicalTriageEnv
from models import Action
import random

# Start game WITHOUT seed for variety
env = MedicalTriageEnv(task="easy", seed=None)  # ← Changed to None
obs, state = env.reset()

print("="*60)
print("🏥 EMERGENCY DEPARTMENT TRIAGE GAME")
print("="*60)
print(f"\nYou have {obs.available_beds} beds available")
print(f"Staff: {obs.staff_available} nurses\n")

# Show waiting patients
print("WAITING PATIENTS:")
for i, patient in enumerate(obs.waiting_patients, 1):
    print(f"\n👤 #{i} - {patient.id} (Age {patient.age})")
    print(f"   ⚠️  Complaint: {patient.chief_complaint}")
    print(f"   📊 Vitals:")
    if patient.vital_signs.heart_rate:
        print(f"      ❤️  Heart Rate: {patient.vital_signs.heart_rate} bpm")
    if patient.vital_signs.blood_pressure_systolic:
        print(f"      🩸 Blood Pressure: {patient.vital_signs.blood_pressure_systolic}/{patient.vital_signs.blood_pressure_diastolic} mmHg")
    if patient.vital_signs.oxygen_saturation:
        print(f"      🫁 O2 Saturation: {patient.vital_signs.oxygen_saturation}%")
    if patient.vital_signs.temperature:
        print(f"      🌡️  Temperature: {patient.vital_signs.temperature:.1f}°C")
    if patient.vital_signs.respiratory_rate:
        print(f"      💨 Respiratory Rate: {patient.vital_signs.respiratory_rate}/min")
    if patient.vital_signs.pain_level:
        print(f"      😣 Pain Level: {patient.vital_signs.pain_level}/10")
    if patient.medical_history:
        print(f"   📋 History: {', '.join(patient.medical_history)}")

# Interactive decision
print("\n" + "="*60)
print("🎮 YOUR DECISION:")
print("="*60)

if obs.waiting_patients:
    # Let's make a smarter decision based on severity
    patient = obs.waiting_patients[0]
    
    # Determine priority based on vitals
    o2 = patient.vital_signs.oxygen_saturation or 100
    hr = patient.vital_signs.heart_rate or 80
    pain = patient.vital_signs.pain_level or 0
    
    # Smart triage logic
    if o2 < 90 or hr > 120 or pain >= 8:
        priority = 1  # Critical
        emoji = "🚨"
    elif hr > 100 or pain >= 6:
        priority = 2  # Emergency
        emoji = "⚠️"
    elif pain >= 4:
        priority = 3  # Urgent
        emoji = "🟡"
    else:
        priority = 4  # Less urgent
        emoji = "🟢"
    
    print(f"\n{emoji} Triaging {patient.id}: {patient.chief_complaint}")
    print(f"   Priority: {priority} ({'Critical' if priority==1 else 'Emergency' if priority==2 else 'Urgent' if priority==3 else 'Less Urgent'})")
    print(f"   Reasoning: O2={o2}%, HR={hr} bpm, Pain={pain}/10")
    
    action = Action(
        action_type="triage",
        patient_id=patient.id,
        priority_level=priority,
        reasoning=f"Based on vitals: O2={o2}%, HR={hr}, Pain={pain}"
    )
    
    obs, reward, done, info, state = env.step(action)
    
    print(f"\n✅ ACTION RESULT:")
    print(f"   💰 Total Reward: {reward.total:.2f}")
    print(f"   ✔️  Triage Accuracy: {reward.triage_accuracy:.2f}")
    print(f"   🏥 Patient Outcome: {reward.patient_outcome_score:.2f}")
    print(f"   ⏱️  Waiting Penalty: {reward.waiting_time_penalty:.2f}")
    print(f"\n   📈 Current Status:")
    print(f"      Triaged: {len(obs.triaged_patients)}")
    print(f"      Waiting: {len(obs.waiting_patients)}")
    print(f"      Available Beds: {obs.available_beds}/{obs.total_beds}")

print("\n" + "="*60)
print("🎮 This is ONE step. Run again to see different patients!")
print("="*60)