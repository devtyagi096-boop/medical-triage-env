# models.py
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any

class VitalSigns(BaseModel):
    """Patient vital signs"""
    heart_rate: Optional[int] = Field(None, description="Beats per minute")
    blood_pressure_systolic: Optional[int] = Field(None, description="mmHg")
    blood_pressure_diastolic: Optional[int] = Field(None, description="mmHg")
    respiratory_rate: Optional[int] = Field(None, description="Breaths per minute")
    temperature: Optional[float] = Field(None, description="Celsius")
    oxygen_saturation: Optional[int] = Field(None, description="SpO2 percentage")
    pain_level: Optional[int] = Field(None, ge=0, le=10, description="Self-reported pain 0-10")

class Patient(BaseModel):
    """Individual patient data"""
    id: str
    arrival_time: float
    age: int
    chief_complaint: str
    vital_signs: VitalSigns
    medical_history: List[str] = Field(default_factory=list)
    current_medications: List[str] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list)
    assigned_priority: Optional[int] = Field(None, ge=1, le=5)
    seen_by_doctor: bool = False
    time_to_treatment: Optional[float] = None

class Observation(BaseModel):
    """Current state of the emergency department"""
    current_time: float = Field(description="Simulation time in minutes")
    waiting_patients: List[Patient] = Field(default_factory=list, description="Patients not yet triaged")
    triaged_patients: List[Patient] = Field(default_factory=list, description="Patients assigned priority")
    available_beds: int = Field(description="Number of available treatment beds")
    total_beds: int = Field(description="Total treatment capacity")
    staff_available: int = Field(description="Medical staff available")
    recent_arrivals: List[str] = Field(default_factory=list, description="Patient IDs who just arrived")

class Action(BaseModel):
    """Agent action in the environment"""
    action_type: Literal["triage", "reassess", "wait", "call_specialist"]
    patient_id: Optional[str] = None
    priority_level: Optional[int] = Field(None, ge=1, le=5, description="1=Critical, 5=Non-urgent")
    specialist_type: Optional[str] = None
    reasoning: Optional[str] = Field(None, description="Agent's reasoning for action")

class Reward(BaseModel):
    """Reward signal with detailed breakdown"""
    total: float = Field(description="Total reward for this step")
    patient_outcome_score: float = Field(0.0, description="Based on patient outcomes")
    efficiency_score: float = Field(0.0, description="Resource utilization")
    waiting_time_penalty: float = Field(0.0, description="Penalty for excessive waits")
    triage_accuracy: float = Field(0.0, description="Correctness of priority assignment")
    details: Dict[str, Any] = Field(default_factory=dict)

class State(BaseModel):
    """Episode state"""
    episode_id: str
    step_count: int = 0
    done: bool = False