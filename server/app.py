# server/app.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
import sys
import os

# Add parent directory to path to import from root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import MedicalTriageEnv
from models import Action, Observation, State, Reward
from grader import grade_task
import json

app = FastAPI(
    title="Medical Triage Environment",
    description="Emergency department patient triage simulation",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active environments (in production, use Redis or similar)
environments: Dict[str, MedicalTriageEnv] = {}

class ResetRequest(BaseModel):
    task: str = "medium"
    seed: int = None

class StepRequest(BaseModel):
    env_id: str
    action: Dict[str, Any]

@app.get("/")
async def root():
    return {
        "name": "Medical Triage Environment",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/tasks")
async def list_tasks():
    """List available tasks"""
    return {
        "tasks": [
            {
                "id": "easy",
                "name": "Basic Triage",
                "description": "Low patient volume, adequate resources",
                "difficulty": "easy",
                "max_steps": 100
            },
            {
                "id": "medium",
                "name": "Standard ED Operations",
                "description": "Moderate patient volume with resource constraints",
                "difficulty": "medium",
                "max_steps": 100
            },
            {
                "id": "hard",
                "name": "Mass Casualty Scenario",
                "description": "High patient volume, limited resources",
                "difficulty": "hard",
                "max_steps": 100
            }
        ],
        "action_schema": {
            "action_type": "string (triage|reassess|wait|call_specialist)",
            "patient_id": "string (optional)",
            "priority_level": "integer 1-5 (optional)",
            "specialist_type": "string (optional)",
            "reasoning": "string (optional)"
        }
    }

@app.post("/reset")
async def reset(request: ResetRequest):
    """Reset environment and start new episode"""
    try:
        env = MedicalTriageEnv(task=request.task, seed=request.seed)
        obs, state = env.reset()
        
        env_id = state.episode_id
        environments[env_id] = env
        
        return {
            "env_id": env_id,
            "observation": obs.dict(),
            "state": state.dict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/step")
async def step(request: StepRequest):
    """Execute action in environment"""
    try:
        if request.env_id not in environments:
            raise HTTPException(status_code=404, detail="Environment not found")
        
        env = environments[request.env_id]
        action = Action(**request.action)
        
        obs, reward, done, info, state = env.step(action)
        
        return {
            "observation": obs.dict(),
            "reward": reward.dict(),
            "done": done,
            "info": info,
            "state": state.dict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/state/{env_id}")
async def get_state(env_id: str):
    """Get current state of environment"""
    if env_id not in environments:
        raise HTTPException(status_code=404, detail="Environment not found")
    
    env = environments[env_id]
    return env._get_state().dict()

@app.post("/grader")
async def run_grader(env_id: str, task: str):
    """Grade the completed episode"""
    if env_id not in environments:
        raise HTTPException(status_code=404, detail="Environment not found")
    
    env = environments[env_id]
    score = grade_task(env, task)
    metrics = env.get_episode_metrics()
    
    return {
        "score": score,
        "metrics": metrics
    }

@app.post("/baseline")
async def run_baseline():
    """Run baseline inference (returns pre-computed scores due to API limits)"""
    return {
        "message": "Baseline scores from test run",
        "scores": {
            "easy": 0.940,
            "medium": 0.792,
            "hard": 0.500
        },
        "note": "Actual baseline requires GROQ_API_KEY environment variable. See baseline.py for implementation."
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)