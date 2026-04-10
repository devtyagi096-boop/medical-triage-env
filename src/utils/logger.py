"""
Logging utilities for inference script — all output goes to stderr,
never stdout (stdout is reserved for the competition output format).
"""

import sys
from datetime import datetime


def log_info(message: str):
    """Log info message to stderr"""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[INFO {ts}] {message}", file=sys.stderr, flush=True)


def log_error(message: str):
    """Log error message to stderr"""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[ERROR {ts}] {message}", file=sys.stderr, flush=True)


def log_step(step: int, action: str, reward: float):
    """Log step details to stderr"""
    print(f"  Step {step}: {action} -> reward={reward:.2f}", file=sys.stderr, flush=True)


def log_debug(message: str, verbose: bool = False):
    """Log debug message if verbose enabled"""
    if verbose:
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[DEBUG {ts}] {message}", file=sys.stderr, flush=True)
