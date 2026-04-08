# environment.py
import random
import uuid
from typing import Dict, Tuple, Optional, List
from models import Observation, Action, Reward, Patient, VitalSigns, State

class MedicalTriageEnv:
    """
    Medical Emergency Department Triage Environment
    
    Agent must prioritize patients, manage resources, and minimize harm.
    """
    
    def __init__(self, task: str = "medium", max_steps: int = 100, seed: Optional[int] = None):
        self.task = task
        self.max_steps = max_steps
        self.seed = seed
        if seed is not None:
            random.seed(seed)
        
        # Task configurations
        self.task_configs = {
            "easy": {
                "arrival_rate": 0.3,
                "total_beds": 8,
                "staff_count": 4,
                "max_patients": 15,
            },
            "medium": {
                "arrival_rate": 0.5,
                "total_beds": 6,
                "staff_count": 3,
                "max_patients": 25,
            },
            "hard": {
                "arrival_rate": 0.7,
                "total_beds": 5,
                "staff_count": 2,
                "max_patients": 35,
            }
        }
        
        self.config = self.task_configs.get(task, self.task_configs["medium"])
        
        self.current_time = 0.0
        self.step_count = 0
        self.waiting_patients: List[Patient] = []
        self.triaged_patients: List[Patient] = []
        self.completed_patients: List[Patient] = []
        self.patient_counter = 0
        self.available_beds = self.config["total_beds"]
        self._ground_truth: Dict[str, int] = {}
        self._episode_id = str(uuid.uuid4())
        
    def reset(self) -> Tuple[Observation, State]:
        """Reset environment to initial state"""
        if self.seed is not None:
            random.seed(self.seed)
            
        self.current_time = 0.0
        self.step_count = 0
        self.waiting_patients = []
        self.triaged_patients = []
        self.completed_patients = []
        self.patient_counter = 0
        self.available_beds = self.config["total_beds"]
        self._ground_truth = {}
        self._episode_id = str(uuid.uuid4())
        
        # Generate initial patients
        initial_count = random.randint(2, 5)
        for _ in range(initial_count):
            self.waiting_patients.append(self._generate_patient())
        
        obs = self._get_observation()
        state = self._get_state()
        return obs, state
    
    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict, State]:
        """Execute action and return next state"""
        self.step_count += 1
        reward_components = {
            "patient_outcome_score": 0.0,
            "efficiency_score": 0.0,
            "waiting_time_penalty": 0.0,
            "triage_accuracy": 0.0
        }
        
        # Process action
        if action.action_type == "triage" and action.patient_id:
            self._triage_patient(action, reward_components)
        elif action.action_type == "reassess" and action.patient_id:
            self._reassess_patient(action, reward_components)
        elif action.action_type == "call_specialist" and action.patient_id:
            self._call_specialist(action, reward_components)
        
        # Advance time
        time_increment = random.uniform(1.0, 3.0)
        self.current_time += time_increment
        
        # Process triaged patients
        self._process_triaged_patients(reward_components)
        
        # Generate new arrivals
        if len(self.waiting_patients) + len(self.triaged_patients) < self.config["max_patients"]:
            if random.random() < self.config["arrival_rate"] * time_increment / 10:
                new_patient = self._generate_patient()
                self.waiting_patients.append(new_patient)
        
        # Calculate waiting penalties
        self._calculate_waiting_penalties(reward_components)
        
        # Check episode termination
        done = (self.step_count >= self.max_steps or 
                len(self.completed_patients) >= self.config["max_patients"])
        
        # Build reward
        total_reward = sum(reward_components.values())
        reward = Reward(
            total=total_reward,
            patient_outcome_score=reward_components["patient_outcome_score"],
            efficiency_score=reward_components["efficiency_score"],
            waiting_time_penalty=reward_components["waiting_time_penalty"],
            triage_accuracy=reward_components["triage_accuracy"],
            details={"step": self.step_count}
        )
        
        info = {
            "total_patients_seen": len(self.completed_patients),
            "waiting_count": len(self.waiting_patients),
            "triaged_count": len(self.triaged_patients)
        }
        
        obs = self._get_observation()
        state = self._get_state()
        return obs, reward, done, info, state
    
    def _get_observation(self) -> Observation:
        """Return current observation"""
        return Observation(
            current_time=self.current_time,
            waiting_patients=self.waiting_patients.copy(),
            triaged_patients=self.triaged_patients.copy(),
            available_beds=self.available_beds,
            total_beds=self.config["total_beds"],
            staff_available=self.config["staff_count"],
            recent_arrivals=[p.id for p in self.waiting_patients[-3:]]
        )
    
    def _get_state(self) -> State:
        """Return current state"""
        done = (self.step_count >= self.max_steps or 
                len(self.completed_patients) >= self.config["max_patients"])
        return State(
            episode_id=self._episode_id,
            step_count=self.step_count,
            done=done
        )
    
    def _generate_patient(self) -> Patient:
        """Generate a new patient"""
        self.patient_counter += 1
        patient_id = f"P{self.patient_counter:04d}"
        age = random.randint(1, 95)
        acuity = random.choices([1, 2, 3, 4, 5], weights=[0.1, 0.2, 0.3, 0.3, 0.1])[0]
        
        vital_signs, complaint, history = self._generate_clinical_presentation(age, acuity)
        
        patient = Patient(
            id=patient_id,
            arrival_time=self.current_time,
            age=age,
            chief_complaint=complaint,
            vital_signs=vital_signs,
            medical_history=history,
            current_medications=self._generate_medications(history),
            allergies=random.choice([[], ["Penicillin"], ["NSAIDs"]])
        )
        
        self._ground_truth[patient_id] = acuity
        return patient
    
    def _generate_clinical_presentation(self, age: int, acuity: int) -> Tuple[VitalSigns, str, List[str]]:
        """Generate realistic vital signs and complaints"""
        presentations = {
            1: [
                ("Chest pain radiating to left arm, diaphoresis",
                 {"heart_rate": 140, "bp_sys": 90, "bp_dias": 60, "resp_rate": 28, "o2_sat": 88, "pain": 9},
                 ["Hypertension", "Diabetes"]),
                ("Unresponsive, found collapsed at home",
                 {"heart_rate": 40, "bp_sys": 70, "bp_dias": 40, "resp_rate": 6, "o2_sat": 82},
                 ["Coronary artery disease"]),
                ("Severe difficulty breathing, unable to speak in full sentences",
                 {"heart_rate": 130, "bp_sys": 100, "bp_dias": 65, "resp_rate": 34, "o2_sat": 84, "pain": 7},
                 ["Asthma", "COPD"]),
                ("Sudden onset severe headache, worst of life, neck stiffness",
                 {"heart_rate": 110, "bp_sys": 190, "bp_dias": 110, "resp_rate": 20, "o2_sat": 96, "pain": 10},
                 []),
                ("Massive GI bleed, hematemesis, hypotensive",
                 {"heart_rate": 145, "bp_sys": 80, "bp_dias": 50, "resp_rate": 24, "o2_sat": 91, "pain": 8},
                 ["Liver cirrhosis", "Alcohol use disorder"]),
            ],
            2: [
                ("Severe abdominal pain, vomiting, rigid abdomen",
                 {"heart_rate": 115, "bp_sys": 118, "bp_dias": 76, "temp": 38.6, "pain": 8},
                 ["Gallstones"]),
                ("High fever, confusion, rigors",
                 {"heart_rate": 120, "bp_sys": 95, "bp_dias": 60, "temp": 39.8, "resp_rate": 26, "o2_sat": 93, "pain": 5},
                 ["Diabetes", "Immunocompromised"]),
                ("Acute allergic reaction, throat tightness, hives",
                 {"heart_rate": 118, "bp_sys": 105, "bp_dias": 68, "resp_rate": 22, "o2_sat": 94, "pain": 6},
                 ["Penicillin allergy"]),
                ("Stroke symptoms: facial droop, arm weakness, slurred speech",
                 {"heart_rate": 88, "bp_sys": 175, "bp_dias": 100, "resp_rate": 18, "o2_sat": 96},
                 ["Hypertension", "Atrial fibrillation"]),
                ("Diabetic ketoacidosis, fruity breath, altered mentation",
                 {"heart_rate": 112, "bp_sys": 105, "bp_dias": 65, "temp": 37.2, "resp_rate": 28, "o2_sat": 97, "pain": 4},
                 ["Type 1 Diabetes"]),
            ],
            3: [
                ("Fever and productive cough for 3 days, moderate distress",
                 {"heart_rate": 98, "temp": 38.8, "resp_rate": 20, "o2_sat": 95, "pain": 4},
                 []),
                ("Moderate flank pain, dysuria, frequency",
                 {"heart_rate": 92, "temp": 38.2, "bp_sys": 128, "bp_dias": 80, "pain": 6},
                 ["Recurrent UTIs"]),
                ("Closed head injury after fall, GCS 14, no LOC",
                 {"heart_rate": 85, "bp_sys": 135, "bp_dias": 82, "resp_rate": 16, "o2_sat": 98, "pain": 5},
                 []),
                ("Moderate asthma exacerbation, wheezing, speaking in phrases",
                 {"heart_rate": 105, "resp_rate": 24, "o2_sat": 93, "pain": 3},
                 ["Asthma"]),
                ("Acute back pain after lifting, unable to walk comfortably",
                 {"heart_rate": 88, "bp_sys": 130, "bp_dias": 84, "pain": 7},
                 ["Hypertension"]),
            ],
            4: [
                ("Minor laceration on forearm requiring sutures",
                 {"heart_rate": 80, "bp_sys": 122, "bp_dias": 78, "pain": 4},
                 []),
                ("Sprained ankle after sports injury, weight-bearing",
                 {"heart_rate": 76, "bp_sys": 118, "bp_dias": 74, "pain": 5},
                 []),
                ("Ear pain and mild fever for 2 days",
                 {"heart_rate": 82, "temp": 37.9, "pain": 4},
                 []),
                ("Mild allergic reaction, localized hives, no airway involvement",
                 {"heart_rate": 78, "bp_sys": 120, "bp_dias": 76, "pain": 2},
                 []),
                ("Dental pain, unable to see dentist",
                 {"heart_rate": 80, "pain": 6},
                 []),
            ],
            5: [
                ("Cold symptoms for 1 week, mild congestion",
                 {"heart_rate": 72, "temp": 37.3, "o2_sat": 99},
                 []),
                ("Prescription refill request, stable chronic condition",
                 {"heart_rate": 70, "bp_sys": 125, "bp_dias": 80},
                 ["Hypertension"]),
                ("Mild rash on arm for 3 days, no systemic symptoms",
                 {"heart_rate": 74, "pain": 1},
                 []),
                ("Routine follow-up, no acute complaints",
                 {"heart_rate": 68, "bp_sys": 118, "bp_dias": 76},
                 ["Diabetes"]),
            ]
        }

        complaint, vitals_dict, history = random.choice(presentations[acuity])

        vital_signs = VitalSigns(
            heart_rate=vitals_dict.get("heart_rate"),
            blood_pressure_systolic=vitals_dict.get("bp_sys"),
            blood_pressure_diastolic=vitals_dict.get("bp_dias"),
            respiratory_rate=vitals_dict.get("resp_rate"),
            temperature=vitals_dict.get("temp"),
            oxygen_saturation=vitals_dict.get("o2_sat"),
            pain_level=vitals_dict.get("pain")
        )

        return vital_signs, complaint, history
    
    def _generate_medications(self, history: List[str]) -> List[str]:
        """Generate medications based on history"""
        meds = []
        if "Hypertension" in history:
            meds.append("Lisinopril")
        if "Diabetes" in history:
            meds.append("Metformin")
        return meds
    
    def _triage_patient(self, action: Action, reward_components: dict):
        """Assign triage priority"""
        patient = None
        for p in self.waiting_patients:
            if p.id == action.patient_id:
                patient = p
                break
        
        if patient and action.priority_level:
            patient.assigned_priority = action.priority_level
            self.waiting_patients.remove(patient)
            self.triaged_patients.append(patient)
            
            ground_truth = self._ground_truth.get(patient.id, 3)
            accuracy = 1.0 - abs(ground_truth - action.priority_level) / 4.0
            reward_components["triage_accuracy"] = accuracy * 2.0
            
            if ground_truth <= 2 and action.priority_level <= 2:
                reward_components["patient_outcome_score"] = 5.0
            elif ground_truth <= 2 and action.priority_level > 2:
                reward_components["patient_outcome_score"] = -10.0
    
    def _reassess_patient(self, action: Action, reward_components: dict):
        """Reassess patient priority"""
        for patient in self.triaged_patients:
            if patient.id == action.patient_id and action.priority_level:
                patient.assigned_priority = action.priority_level
                reward_components["triage_accuracy"] = 1.0
    
    def _call_specialist(self, action: Action, reward_components: dict):
        """Call specialist"""
        for patient in self.triaged_patients:
            if patient.id == action.patient_id:
                ground_truth = self._ground_truth.get(patient.id, 3)
                if ground_truth <= 2:
                    reward_components["patient_outcome_score"] = 2.0
    
    def _process_triaged_patients(self, reward_components: dict):
        """Move patients to treatment"""
        self.triaged_patients.sort(key=lambda p: (p.assigned_priority or 999, p.arrival_time))
        
        while self.triaged_patients and self.available_beds > 0:
            patient = self.triaged_patients.pop(0)
            patient.seen_by_doctor = True
            patient.time_to_treatment = self.current_time - patient.arrival_time
            self.completed_patients.append(patient)
            self.available_beds -= 1
            
            ground_truth = self._ground_truth.get(patient.id, 3)
            if ground_truth == 1 and patient.time_to_treatment < 10:
                reward_components["patient_outcome_score"] += 3.0
        
        if random.random() < 0.3:
            self.available_beds = min(self.available_beds + 1, self.config["total_beds"])
    
    def _calculate_waiting_penalties(self, reward_components: dict):
        """Penalize excessive waiting"""
        penalty = 0.0
        for patient in self.waiting_patients:
            wait_time = self.current_time - patient.arrival_time
            ground_truth = self._ground_truth.get(patient.id, 3)
            
            if ground_truth == 1 and wait_time > 5:
                penalty -= (wait_time - 5) * 0.5
        
        reward_components["waiting_time_penalty"] = penalty
    
    def get_episode_metrics(self) -> dict:
        """Calculate final episode metrics"""
        if not self.completed_patients:
            return {
                "avg_wait_time": 0,
                "critical_patient_wait": 0,
                "triage_accuracy": 0,
                "patients_treated": 0
            }
        
        wait_times = [p.time_to_treatment for p in self.completed_patients if p.time_to_treatment]
        avg_wait = sum(wait_times) / len(wait_times) if wait_times else 0
        
        critical_waits = []
        correct_triage = 0
        total_triaged = 0
        
        for patient in self.completed_patients:
            ground_truth = self._ground_truth.get(patient.id, 3)
            if ground_truth <= 2 and patient.time_to_treatment:
                critical_waits.append(patient.time_to_treatment)
            
            if patient.assigned_priority:
                total_triaged += 1
                if abs(ground_truth - patient.assigned_priority) <= 1:
                    correct_triage += 1
        
        return {
            "avg_wait_time": avg_wait,
            "critical_patient_wait": sum(critical_waits) / len(critical_waits) if critical_waits else 0,
            "triage_accuracy": correct_triage / total_triaged if total_triaged > 0 else 0,
            "patients_treated": len(self.completed_patients)
        }