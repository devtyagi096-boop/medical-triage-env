"""
LLM-based Medical Triage Agent
"""

import re
import time
from typing import Dict, Any, List, Optional

from openai import OpenAI


class LLMTriageAgent:
    """
    Intelligent triage agent using LLM reasoning with rule-based fallback.
    """

    def __init__(
        self,
        client: OpenAI,
        model_name: str,
        prompt_builder,
        temperature: float = 0.3,
        max_tokens: int = 200,
        use_cot: bool = True,
    ):
        self.client = client
        self.model_name = model_name
        self.prompt_builder = prompt_builder
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_cot = use_cot
        self.episode_history: List[Dict] = []

    def reset(self):
        """Reset agent state for new episode"""
        self.episode_history = []

    def get_action(self, observation: Dict[str, Any], step: int) -> str:
        """
        Get triage decision from LLM.

        Returns:
            Action string in format "triage(X)" where X=0-3
        """
        try:
            prompt = self.prompt_builder.build_prompt(
                observation=observation,
                history=self.episode_history,
                step=step,
            )
            raw = self._call_llm_with_retry(prompt)
            action_str = self._parse_and_validate(raw)

            self.episode_history.append({
                'step': step,
                'observation': observation,
                'action': action_str,
            })
            return action_str

        except Exception:
            return self._fallback_action(observation)

    def _call_llm_with_retry(self, prompt: str, max_retries: int = 3) -> str:
        """Call LLM with exponential backoff retry"""
        last_exc: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self.prompt_builder.get_system_prompt()},
                        {"role": "user",   "content": prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return response.choices[0].message.content or ""
            except Exception as exc:
                last_exc = exc
                if attempt < max_retries - 1:
                    time.sleep((2 ** attempt) * 0.5)
        raise last_exc  # type: ignore[misc]

    def _parse_and_validate(self, llm_output: str) -> str:
        """
        Parse LLM output to extract valid action.
        Tries multiple patterns in order of specificity.
        """
        llm_output = llm_output.strip()

        # Pattern 1: "Action: triage(X)"
        m = re.search(r'[Aa]ction:\s*triage\(([0-3])\)', llm_output)
        if m:
            return f"triage({m.group(1)})"

        # Pattern 2: Direct "triage(X)"
        m = re.search(r'triage\(([0-3])\)', llm_output, re.IGNORECASE)
        if m:
            return f"triage({m.group(1)})"

        # Pattern 3: "Priority: X"
        m = re.search(r'[Pp]riority:\s*([0-3])', llm_output)
        if m:
            return f"triage({m.group(1)})"

        # Pattern 4: standalone digit 0-3
        m = re.search(r'\b([0-3])\b', llm_output)
        if m:
            return f"triage({m.group(1)})"

        return "triage(2)"  # default semi-urgent

    def _fallback_action(self, obs: Dict[str, Any]) -> str:
        """Rule-based fallback when LLM fails"""
        severity = obs.get('severity', 2)
        spo2 = float(obs.get('oxygen_saturation', [95])[0])
        hr   = float(obs.get('heart_rate',         [80])[0])
        sbp  = float(obs.get('systolic_bp',        [120])[0])

        if spo2 < 85 or hr > 150 or hr < 45 or sbp < 85:
            return "triage(0)"
        if severity >= 4:
            return "triage(0)"
        elif severity == 3:
            return "triage(1)"
        elif severity == 2:
            return "triage(2)"
        else:
            return "triage(3)"
