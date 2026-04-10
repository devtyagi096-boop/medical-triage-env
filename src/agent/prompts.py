"""
Prompt templates for medical triage LLM agent
"""

from typing import Dict, Any, List


class TriagePromptBuilder:
    """Build prompts for triage decisions"""

    def get_system_prompt(self) -> str:
        return (
            "You are an expert emergency department triage nurse with 20+ years of experience.\n\n"
            "Your role is to rapidly assess patients and assign appropriate triage priority based on:\n"
            "- Vital signs (heart rate, blood pressure, oxygen saturation, respiratory rate, temperature)\n"
            "- Patient severity level\n"
            "- Clinical presentation\n"
            "- Risk of deterioration\n\n"
            "You must make quick, evidence-based decisions that prioritize patient safety "
            "while managing ED resources effectively.\n\n"
            "CRITICAL RULE: When in doubt, assign HIGHER priority to ensure patient safety."
        )

    def build_prompt(
        self,
        observation: Dict[str, Any],
        history: List[Dict],
        step: int,
    ) -> str:
        """Build complete prompt for current decision"""
        patient_id  = observation.get('patient_id', 0)
        age         = float(observation.get('age', [0])[0])
        hr          = float(observation.get('heart_rate', [0])[0])
        sbp         = float(observation.get('systolic_bp', [0])[0])
        dbp         = float(observation.get('diastolic_bp', [0])[0])
        rr          = float(observation.get('respiratory_rate', [0])[0])
        spo2        = float(observation.get('oxygen_saturation', [0])[0])
        temp        = float(observation.get('temperature', [0])[0])
        severity    = int(observation.get('severity', 0))
        wait_time   = float(observation.get('wait_time', [0])[0])
        queue_len   = int(observation.get('queue_length', 0))

        prompt = (
            f"## PATIENT ASSESSMENT - Step {step}\n\n"
            f"**Patient ID:** {patient_id}\n"
            f"**Age:** {int(age)} years\n"
            f"**Current Wait Time:** {int(wait_time)} minutes\n"
            f"**ED Queue Length:** {queue_len} patients\n\n"
            f"### VITAL SIGNS\n\n"
            f"| Parameter | Value | Normal Range | Status |\n"
            f"|-----------|-------|--------------|--------|\n"
            f"| Heart Rate | {int(hr)} bpm | 60-100 | {self._assess_hr(hr)} |\n"
            f"| Blood Pressure | {int(sbp)}/{int(dbp)} mmHg | 120/80 | {self._assess_bp(sbp, dbp)} |\n"
            f"| Respiratory Rate | {int(rr)}/min | 12-20 | {self._assess_rr(rr)} |\n"
            f"| Oxygen Saturation | {int(spo2)}% | >95% | {self._assess_spo2(spo2)} |\n"
            f"| Temperature | {temp:.1f}C | 36.5-37.5 | {self._assess_temp(temp)} |\n\n"
            f"**Clinical Severity Score:** {severity}/5\n\n"
            "### TRIAGE PRIORITY LEVELS\n\n"
            "**0 - IMMEDIATE** - Life-threatening, act now (SpO2<85%, HR>150 or <45, SBP<90)\n"
            "**1 - URGENT** - Serious, treat within 15 min (abnormal vitals, severity>=3)\n"
            "**2 - SEMI-URGENT** - Stable, short wait OK (severity 2, stable vitals)\n"
            "**3 - NON-URGENT** - Minor, can wait (severity 1, normal vitals)\n\n"
            "### YOUR TASK\n\n"
            "Respond in this EXACT format:\n\n"
            "Priority: [0, 1, 2, or 3]\n"
            "Reasoning: [Brief 1-sentence clinical justification]\n"
            "Action: triage([your priority number])\n"
        )

        if history and len(history) >= 2:
            prompt += "\n### RECENT DECISIONS\n\n"
            for h in history[-2:]:
                prompt += f"- Step {h['step']}: {h['action']}\n"

        return prompt

    def _assess_hr(self, hr: float) -> str:
        if hr < 50:   return "BRADYCARDIA"
        if hr > 120:  return "TACHYCARDIA"
        if hr > 100:  return "Elevated"
        return "Normal"

    def _assess_bp(self, sbp: float, dbp: float) -> str:
        if sbp < 90:   return "HYPOTENSION"
        if sbp > 180:  return "HYPERTENSION"
        if sbp > 140:  return "Elevated"
        return "Normal"

    def _assess_rr(self, rr: float) -> str:
        if rr < 10:  return "BRADYPNEA"
        if rr > 24:  return "TACHYPNEA"
        if rr > 20:  return "Elevated"
        return "Normal"

    def _assess_spo2(self, spo2: float) -> str:
        if spo2 < 85:  return "CRITICAL"
        if spo2 < 90:  return "HYPOXIA"
        if spo2 < 95:  return "Low"
        return "Normal"

    def _assess_temp(self, temp: float) -> str:
        if temp < 35.5:  return "HYPOTHERMIA"
        if temp > 38.5:  return "FEVER"
        if temp > 37.5:  return "Elevated"
        return "Normal"
