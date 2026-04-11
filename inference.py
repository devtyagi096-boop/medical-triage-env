#!/usr/bin/env python3
"""
Medical Triage Environment - Competition Inference Script
OpenEnv RL Challenge Submission

Author: Arnav Tyagi
License: MIT
"""

import os
import sys
import traceback
import warnings
from typing import List, Dict, Any, Optional

warnings.filterwarnings('ignore')

# ── Force unbuffered stdout — validator must see output immediately ────────────
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)

from openai import OpenAI

# ── Environment variables ─────────────────────────────────────────────────────
# API_BASE_URL and MODEL_NAME MUST have defaults per submission rules.
# HF_TOKEN is mandatory — no default.
API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")

# Validate HF_TOKEN immediately
if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

# Competition configuration
TASK_NAME              = os.getenv("TASK_NAME",  "medical-triage")
BENCHMARK              = os.getenv("BENCHMARK",  "emergency-dept")
NUM_EPISODES           = int(os.getenv("NUM_EPISODES", "3"))
MAX_STEPS_PER_EPISODE  = int(os.getenv("MAX_STEPS",    "30"))
VERBOSE                = os.getenv("VERBOSE", "false").lower() == "true"

# ── OpenAI client ─────────────────────────────────────────────────────────────
try:
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN,
        timeout=30.0,
        max_retries=2,
    )
except Exception as e:
    print(f"ERROR: Failed to initialize OpenAI client: {e}", file=sys.stderr)
    raise

# ── Import src components ─────────────────────────────────────────────────────
from src.env.medical_triage_env import MedicalTriageEnv
from src.agent.llm_agent import LLMTriageAgent
from src.agent.prompts import TriagePromptBuilder
from src.utils.logger import log_info, log_error


# ── Output format helpers ─────────────────────────────────────────────────────

def _fmt_reward(value: float) -> str:
    return f"{value:.2f}"


def _fmt_bool(value: bool) -> str:
    return "true" if value else "false"


def _fmt_error(error: Optional[str]) -> str:
    if error is None:
        return "null"
    cleaned = str(error).replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').strip()
    return cleaned[:200] + "..." if len(cleaned) > 200 else cleaned


def emit_start(task_id: str) -> None:
    """[START] task=<name> env=<benchmark> model=<model_name>"""
    sys.stdout.write(f"[START] task={task_id} env={BENCHMARK} model={MODEL_NAME}\n")
    sys.stdout.flush()


def emit_step(step: int, action_str: str, reward: float, done: bool,
              error: Optional[str]) -> None:
    """[STEP] step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>"""
    sys.stdout.write(
        f"[STEP] step={step} action={action_str} "
        f"reward={_fmt_reward(reward)} done={_fmt_bool(done)} error={_fmt_error(error)}\n"
    )
    sys.stdout.flush()


def emit_end(success: bool, steps: int, rewards: List[float]) -> None:
    """[END] success=<true|false> steps=<n> rewards=<r1,r2,...,rn>"""
    rewards_str = ",".join(_fmt_reward(r) for r in rewards)
    sys.stdout.write(f"[END] success={_fmt_bool(success)} steps={steps} rewards={rewards_str}\n")
    sys.stdout.flush()


# ── Episode runner ────────────────────────────────────────────────────────────

def run_single_episode(
    env: MedicalTriageEnv,
    agent: LLMTriageAgent,
    episode_num: int,
) -> Dict[str, Any]:
    """
    Run one episode. Emits [START], one [STEP] per step, then [END].
    [END] is always emitted via finally block.
    """
    task_id = f"{TASK_NAME}-ep{episode_num}"

    observation = env.reset()
    agent.reset()

    rewards: List[float] = []
    step_count = 0
    success = False
    last_error: Optional[str] = None

    # [START] — emitted before the try so [END] in finally always follows it
    emit_start(task_id)

    try:
        for step in range(1, MAX_STEPS_PER_EPISODE + 1):
            step_count = step

            # ── Get action ────────────────────────────────────────────────
            try:
                action_str = agent.get_action(observation, step)
                last_error = None
            except Exception as e:
                log_error(f"Agent error at step {step}: {e}")
                action_str = "triage(2)"
                last_error = f"AgentError:{type(e).__name__}"

            # ── Step environment ──────────────────────────────────────────
            try:
                observation, reward, done, info = env.step(action_str)
                rewards.append(reward)

                # Prefer env error over agent error
                env_err = info.get('error')
                if env_err:
                    last_error = str(env_err).replace('\n', ' ')

            except Exception as e:
                log_error(f"Env step error at step {step}: {e}")
                reward = -50.0
                done = True
                last_error = f"EnvError:{type(e).__name__}"
                rewards.append(reward)
                info = {'success': False}

            # ── [STEP] ────────────────────────────────────────────────────
            emit_step(step, action_str, reward, done, last_error)

            if done:
                success = bool(info.get('success', reward > 0))
                break

    except KeyboardInterrupt:
        log_error("Episode interrupted")
        success = False

    except Exception as e:
        log_error(f"Unexpected episode error: {e}")
        traceback.print_exc(file=sys.stderr)
        success = False

    finally:
        # [END] — always emitted
        emit_end(success, step_count, rewards)
        try:
            env.close()
        except Exception:
            pass

    return {
        'episode':      episode_num,
        'success':      success,
        'steps':        step_count,
        'total_reward': sum(rewards),
        'avg_reward':   sum(rewards) / len(rewards) if rewards else 0.0,
        'final_reward': rewards[-1] if rewards else 0.0,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if VERBOSE:
        log_info("=" * 70)
        log_info("Medical Triage Environment - Inference Script")
        log_info("=" * 70)
        log_info(f"API Base URL : {API_BASE_URL}")
        log_info(f"Model        : {MODEL_NAME}")
        log_info(f"Episodes     : {NUM_EPISODES}")
        log_info(f"Max Steps    : {MAX_STEPS_PER_EPISODE}")
        log_info("=" * 70)

    try:
        env = MedicalTriageEnv(
            max_patients_per_episode=MAX_STEPS_PER_EPISODE,
            enable_deterioration=True,
            realistic_conditions=True,
        )
        prompt_builder = TriagePromptBuilder()
        agent = LLMTriageAgent(
            client=client,
            model_name=MODEL_NAME,
            prompt_builder=prompt_builder,
            temperature=0.3,
            max_tokens=200,
        )
    except Exception as e:
        log_error(f"Failed to initialize components: {e}")
        raise

    all_results = []
    for episode_num in range(1, NUM_EPISODES + 1):
        if VERBOSE:
            log_info(f"Starting Episode {episode_num}/{NUM_EPISODES}")
        try:
            result = run_single_episode(env, agent, episode_num)
            all_results.append(result)
        except Exception as e:
            log_error(f"Episode {episode_num} failed: {e}")

    if VERBOSE and all_results:
        log_info("=" * 70)
        log_info("SUMMARY")
        log_info("=" * 70)
        success_rate = sum(r['success'] for r in all_results) / len(all_results)
        avg_reward   = sum(r['total_reward'] for r in all_results) / len(all_results)
        log_info(f"Episodes   : {len(all_results)}/{NUM_EPISODES}")
        log_info(f"Success    : {success_rate*100:.1f}%")
        log_info(f"Avg Reward : {avg_reward:.2f}")
        log_info("=" * 70)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log_error(f"Fatal error: {e}")
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
