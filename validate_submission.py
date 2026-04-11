#!/usr/bin/env python3
"""
Pre-Submission Validation Script
Run this before submitting to the competition to verify everything is correct.

Usage:
    python validate_submission.py
"""

import os
import re
import sys
import subprocess
from pathlib import Path


def check(label: str, passed: bool) -> bool:
    print(f"  {'✅' if passed else '❌'} {label}")
    return passed


def main() -> int:
    print("=" * 60)
    print("🔍 COMPETITION SUBMISSION VALIDATION")
    print("=" * 60)
    all_passed = True

    # ── 1. Required files ─────────────────────────────────────────
    print("\n📁 Required Files")
    for f in ["inference.py", "requirements.txt", "Dockerfile", "README.md", "openenv.yaml"]:
        if not check(f, Path(f).exists()):
            all_passed = False

    # ── 2. Environment variables in inference.py ──────────────────
    print("\n🔑 Environment Variables in inference.py")
    try:
        src = Path("inference.py").read_text()
        if not check("API_BASE_URL has default", re.search(
                r'API_BASE_URL\s*=\s*os\.getenv\(["\']API_BASE_URL["\'],\s*["\'].+["\']\)', src)):
            all_passed = False
        if not check("MODEL_NAME has default", re.search(
                r'MODEL_NAME\s*=\s*os\.getenv\(["\']MODEL_NAME["\'],\s*["\'].+["\']\)', src)):
            all_passed = False
        if not check("HF_TOKEN is read", "HF_TOKEN" in src):
            all_passed = False
        if not check("OpenAI client used", "from openai import OpenAI" in src):
            all_passed = False
    except Exception as e:
        print(f"  ❌ Could not read inference.py: {e}")
        all_passed = False

    # ── 3. Output format test ─────────────────────────────────────
    print("\n📋 Output Format")
    env = os.environ.copy()
    env.update({"NUM_EPISODES": "1", "MAX_STEPS": "3",
                "HF_TOKEN": env.get("HF_TOKEN", "test_placeholder"),
                "VERBOSE": "false"})
    try:
        result = subprocess.run(
            [sys.executable, "inference.py"],
            capture_output=True, text=True, env=env, timeout=30
        )
        lines = [l for l in result.stdout.split("\n") if l.strip()]

        if not lines:
            print("  ❌ No stdout output produced")
            all_passed = False
        else:
            if not check("[START] line present", lines[0].startswith("[START]")):
                all_passed = False
            else:
                for field in ["task=", "env=", "model="]:
                    if not check(f"[START] has {field}", field in lines[0]):
                        all_passed = False

            step_lines = [l for l in lines if l.startswith("[STEP]")]
            if not check(f"{len(step_lines)} [STEP] line(s) found", len(step_lines) > 0):
                all_passed = False
            else:
                for sl in step_lines:
                    for field in ["step=", "action=", "reward=", "done=", "error="]:
                        if field not in sl:
                            print(f"  ❌ [STEP] missing {field}")
                            all_passed = False
                    if not re.search(r"reward=-?\d+\.\d{2}", sl):
                        print("  ❌ reward not 2 decimal places")
                        all_passed = False
                    if not re.search(r"done=(true|false)", sl):
                        print("  ❌ done not lowercase bool")
                        all_passed = False
                if all_passed:
                    print(f"  ✅ All [STEP] fields and formats correct")

            if not check("[END] line present", lines[-1].startswith("[END]")):
                all_passed = False
            else:
                for field in ["success=", "steps=", "rewards="]:
                    if not check(f"[END] has {field}", field in lines[-1]):
                        all_passed = False
                if not re.search(r"success=(true|false)", lines[-1]):
                    print("  ❌ success not lowercase bool")
                    all_passed = False

    except subprocess.TimeoutExpired:
        print("  ❌ inference.py timed out (30s)")
        all_passed = False
    except Exception as e:
        print(f"  ❌ Error running inference.py: {e}")
        all_passed = False

    # ── Result ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL CHECKS PASSED — ready to submit")
        print("=" * 60)
        print("\nReminder: ensure your HuggingFace Space is in Running state before submitting.")
        return 0
    else:
        print("❌ SOME CHECKS FAILED — fix issues above before submitting")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
