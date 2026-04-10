"""
Medical Triage Environment - Client

Connects to a running Medical Triage environment server via HTTP.
Follows the OpenEnv client/server convention used by reference environments.

Usage:
    from client import MedicalTriageClient

    client = MedicalTriageClient(base_url="http://localhost:7860")

    # Start episode
    result = client.reset(task="medium", seed=42)
    env_id = result["env_id"]

    # Step
    action = {
        "action_type": "triage",
        "patient_id": "P0001",
        "priority_level": 2,
        "reasoning": "Low SpO2 and elevated HR"
    }
    result = client.step(env_id, action)

    # Grade
    score = client.grade(env_id, task="medium")
"""

import os
from typing import Any, Dict, Optional

import httpx


class MedicalTriageClient:
    """
    HTTP client for the Medical Triage Environment server.

    Wraps the REST API exposed by server/app.py so external agents
    can interact with the environment without importing Python modules directly.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:7860",
        timeout: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=timeout)

    # ── Core OpenEnv interface ────────────────────────────────────────────────

    def reset(self, task: str = "medium", seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Start a new episode.

        Returns:
            {env_id, observation, state}
        """
        payload: Dict[str, Any] = {"task": task}
        if seed is not None:
            payload["seed"] = seed
        resp = self._client.post(f"{self.base_url}/reset", json=payload)
        resp.raise_for_status()
        return resp.json()

    def step(self, env_id: str, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute one action in the environment.

        Args:
            env_id:  Episode ID returned by reset()
            action:  Dict with action_type, patient_id, priority_level, etc.

        Returns:
            {observation, reward, done, info, state}
        """
        resp = self._client.post(
            f"{self.base_url}/step",
            json={"env_id": env_id, "action": action},
        )
        resp.raise_for_status()
        return resp.json()

    def state(self, env_id: str) -> Dict[str, Any]:
        """Get current episode state."""
        resp = self._client.get(f"{self.base_url}/state/{env_id}")
        resp.raise_for_status()
        return resp.json()

    def grade(self, env_id: str, task: str) -> Dict[str, Any]:
        """
        Grade a completed episode.

        Returns:
            {score, metrics}
        """
        resp = self._client.get(
            f"{self.base_url}/grader",
            params={"env_id": env_id, "task": task},
        )
        resp.raise_for_status()
        return resp.json()

    # ── Utility ───────────────────────────────────────────────────────────────

    def health(self) -> Dict[str, Any]:
        """Check server health."""
        resp = self._client.get(f"{self.base_url}/health")
        resp.raise_for_status()
        return resp.json()

    def tasks(self) -> Dict[str, Any]:
        """List available tasks and action schema."""
        resp = self._client.get(f"{self.base_url}/tasks")
        resp.raise_for_status()
        return resp.json()

    def close(self):
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ── Quick demo ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    base_url = os.getenv("ENV_BASE_URL", "http://localhost:7860")

    with MedicalTriageClient(base_url=base_url) as client:
        print(f"Health: {client.health()}")
        print(f"Tasks:  {[t['id'] for t in client.tasks()['tasks']]}")

        result = client.reset(task="easy", seed=42)
        env_id = result["env_id"]
        print(f"\nEpisode started: {env_id}")

        obs = result["observation"]
        waiting = obs.get("waiting_patients", [])
        print(f"Waiting patients: {len(waiting)}")

        if waiting:
            p = waiting[0]
            action = {
                "action_type": "triage",
                "patient_id": p["id"],
                "priority_level": 2,
                "reasoning": "Demo triage action",
            }
            step_result = client.step(env_id, action)
            print(f"Reward: {step_result['reward']['total']:.2f}")
            print(f"Done:   {step_result['done']}")

        grade = client.grade(env_id, task="easy")
        print(f"\nScore:  {grade['score']:.3f}")
        print(f"Metrics: {grade['metrics']}")
