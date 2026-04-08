---
title: Medical Triage Environment
emoji: 🏥
colorFrom: red
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Medical Triage Environment

A realistic emergency department triage simulation for training and evaluating AI agents on high-stakes clinical decision-making.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This OpenEnv environment simulates a busy emergency department where an AI agent acts as a triage nurse. The agent must assess incoming patients, interpret vital signs and symptoms, assign correct urgency levels, and manage limited resources under time pressure.

This is not a toy game. Triage is a real professional workflow performed in hospitals worldwide. Errors in priority assignment directly affect patient outcomes. The environment is designed to test structured clinical reasoning, prioritization under uncertainty, and resource-aware decision-making.

## Why This Matters

- Tests clinical reasoning — agents must interpret vital signs, complaints, and medical history
- Requires prioritization — limited beds and staff force difficult trade-offs
- Involves risk assessment — misclassifying a critical patient carries heavy penalties
- Simulates real operational constraints — patient arrivals, bed turnover, staff limits

## Live Demo

- Hugging Face Space: https://huggingface.co/spaces/makdiiimann/medical-triage-env
- API Endpoint: https://makdiiimann-medical-triage-env.hf.space
- Interactive API Docs: https://makdiiimann-medical-triage-env.hf.space/docs
- GitHub Repository: https://github.com/devtyagi096-boop/medical-triage-env

---

## Installation

```bash
git clone https://github.com/devtyagi096-boop/medical-triage-env.git
cd medical-triage-env
pip install -r requirements.txt
```

### Run the server locally

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Run inference (validator-facing script)

```bash
export API_BASE_URL=https://api-inference.huggingface.co/v1
export MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
export HF_TOKEN=your_hf_token_here

python inference.py
```

---

## Environment Interface

The environment follows the OpenEnv interface with typed Pydantic models.

### reset()

Starts a fresh episode. Returns `(Observation, State)`.

```python
from environment import MedicalTriageEnv
env = MedicalTriageEnv(task="medium", seed=42)
obs, state = env.reset()
```

### step(action)

Executes one action. Returns `(Observation, Reward, done, info, State)`.

```python
from models import Action
action = Action(action_type="triage", patient_id="P0001", priority_level=2, reasoning="Elevated HR and low SpO2")
obs, reward, done, info, state = env.step(action)
```

### get_episode_metrics()

Returns a dict with `patients_treated`, `avg_wait_time`, `critical_patient_wait`, `triage_accuracy`.

---

## Observation Space

| Field | Type | Description |
|---|---|---|
| current_time | float | Simulation time in minutes |
| waiting_patients | list[Patient] | Patients not yet triaged |
| triaged_patients | list[Patient] | Patients assigned priority, awaiting beds |
| available_beds | int | Currently free treatment beds |
| total_beds | int | Total bed capacity |
| staff_available | int | Medical staff on duty |
| recent_arrivals | list[str] | Patient IDs who just arrived |

Each `Patient` includes: `id`, `age`, `chief_complaint`, `vital_signs` (HR, BP, RR, Temp, SpO2, Pain), `medical_history`, `current_medications`, `allergies`.

---

## Action Space

| Field | Type | Description |
|---|---|---|
| action_type | string | `triage`, `reassess`, `wait`, `call_specialist` |
| patient_id | string or null | Target patient ID |
| priority_level | int 1–5 or null | 1=Critical, 5=Non-urgent |
| specialist_type | string or null | e.g. "cardiology", "neurology" |
| reasoning | string or null | Agent's explanation |

---

## Reward Function

Rewards are dense and provided at every step:

| Component | Description |
|---|---|
| triage_accuracy | Partial credit for near-correct priority assignment |
| patient_outcome_score | Bonus for correctly prioritizing critical patients; penalty for missing them |
| waiting_time_penalty | Penalty for critical patients waiting too long untriaged |
| efficiency_score | Reward for effective resource utilization |

Reward range: approximately -20.0 to +10.0 per step.

---

## Tasks

### Easy — Basic Triage
- Arrival rate: 0.3, Beds: 8, Staff: 4, Max patients: 15
- Pass criteria: ≥10 patients treated, avg wait < 20 min, accuracy > 70%

### Medium — Standard ED Operations
- Arrival rate: 0.5, Beds: 6, Staff: 3, Max patients: 25
- Pass criteria: ≥18 patients treated, critical wait < 15 min, accuracy > 75%

### Hard — Mass Casualty Scenario
- Arrival rate: 0.7, Beds: 5, Staff: 2, Max patients: 35
- Pass criteria: ≥25 patients treated, critical wait < 10 min, accuracy > 80%

All tasks use deterministic graders returning scores in [0.0, 1.0].

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | /health | Health check |
| GET | /tasks | List available tasks and action schema |
| POST | /reset | Start new episode |
| POST | /step | Execute action |
| GET | /state/{env_id} | Get current episode state |
| GET | /grader | Score a completed episode (`?env_id=...&task=...`) |
| GET | /docs | Interactive Swagger UI |

### Example: Full episode via API

```bash
# Reset
curl -X POST https://makdiiimann-medical-triage-env.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "medium", "seed": 42}'

# Step
curl -X POST https://makdiiimann-medical-triage-env.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"env_id": "<env_id>", "action": {"action_type": "triage", "patient_id": "P0001", "priority_level": 2, "reasoning": "Low SpO2 and elevated HR"}}'

# Grade
curl "https://makdiiimann-medical-triage-env.hf.space/grader?env_id=<env_id>&task=medium"
```

---

## Inference Script

`inference.py` is the validator-facing script. It reads three required environment variables:

- `API_BASE_URL` — OpenAI-compatible API base URL
- `MODEL_NAME` — Model identifier
- `HF_TOKEN` — Bearer token / API key

It emits structured JSON logs to stdout:

- `[START]` — emitted at the beginning of each task
- `[STEP]` — emitted after every environment step with action, reward, and info
- `[RESULT]` — emitted at the end of each task with score and metrics
- `[END]` — emitted once at the very end with full summary across all tasks

```bash
export API_BASE_URL=https://api-inference.huggingface.co/v1
export MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
export HF_TOKEN=hf_...

python inference.py
```

---

## Baseline Results

Baseline agent: `llama-3.3-70b-versatile` via Groq API, seed=42.

| Task | Score | Patients Treated | Triage Accuracy |
|---|---|---|---|
| Easy | 0.940 | 8 | 100% |
| Medium | 0.792 | 3 | 100% |
| Hard | incomplete | — | — (API rate limit) |

Full results in `baseline_results.json`.

---

## Docker Deployment

```bash
docker build -t medical-triage-env .
docker run -p 7860:7860 medical-triage-env
```

The Dockerfile copies all required files including `inference.py` and exposes port 7860.

---

## Project Structure

```
medical-triage-env/
├── models.py            # Pydantic models: Observation, Action, Reward, State
├── environment.py       # Core simulation logic
├── grader.py            # Deterministic task graders (easy/medium/hard)
├── inference.py         # Validator-facing inference script
├── baseline.py          # Baseline agent using Groq API
├── baseline_results.json
├── openenv.yaml         # Environment metadata
├── requirements.txt
├── Dockerfile
├── server/
│   └── app.py           # FastAPI server
└── README.md
```

---

## License

MIT License. See LICENSE for details.
