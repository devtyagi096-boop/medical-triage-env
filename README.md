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

A realistic emergency department triage simulation for training and evaluating AI agents on critical healthcare decision-making.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This OpenEnv environment simulates an emergency department where an AI agent acts as a triage nurse. The agent must:

- **Assess incoming patients** based on vital signs, symptoms, and medical history
- **Assign appropriate priority levels** (1=Critical to 5=Non-urgent)
- **Manage limited resources** (hospital beds, medical staff)
- **Minimize patient harm** and waiting times
- **Handle varying patient volumes** and acuity levels

## Why This Matters

Medical triage is a critical real-world task where decisions directly impact patient outcomes. This environment:

- ✅ **Tests clinical reasoning** - Agents must interpret vital signs and symptoms
- ✅ **Requires prioritization** - Limited resources demand difficult choices
- ✅ **Involves risk assessment** - Misclassifying critical patients can have severe consequences
- ✅ **Simulates real constraints** - Bed availability, staff limitations, patient arrivals

## Live Demo

Try the deployed environment:

- **🚀 HuggingFace Space:** https://huggingface.co/spaces/makdiiimann/medical-triage-env
- **🌐 API Endpoint:** https://makdiiimann-medical-triage-env.hf.space
- **📚 Interactive API Docs:** https://makdiiimann-medical-triage-env.hf.space/docs
- **💻 GitHub Repository:** https://github.com/devtyagi096-boop/medical-triage-env

## Installation

### Quick Start

```bash
# Clone from GitHub
git clone https://github.com/devtyagi096-boop/medical-triage-env.git
cd medical-triage-env

# Install dependencies
pip install -r requirements.txt