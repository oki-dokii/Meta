---
title: ContentModerationEnv
emoji: 🛡️
colorFrom: indigo
colorTo: violet
sdk: docker
pinned: false
license: mit
tags:
  - openenv
  - benchmark
  - content-moderation
  - reinforcement-learning
  - trust-and-safety
  - nlp
---

# 🛡️ ContentModerationEnv

> **A real-world OpenEnv benchmark** for evaluating AI agents on the task of content moderation.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-v1.0-green.svg)](openenv.yaml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

`ContentModerationEnv` is a fully-spec'd [OpenEnv](https://openenv.dev) environment that simulates the real-world task of platform content moderation. An AI agent reads user-generated content (text, audio transcript, visual tags), considers the poster's history and platform policy, then decides:

1. **Classify** the content: `safe | toxic | spam | misleading`
2. **Take action**: `allow | warn | remove | shadowban | escalate`
3. **Rate severity** (hard tier only): `1` (mild) → `5` (critical)

Agents receive partial-credit rewards (0.0 – 1.0) for each correct component, providing a rich gradient signal for learning.

---

## Environment Description

| Property | Value |
|----------|-------|
| Domain | Trust & Safety / NLP |
| Task type | Classification + Decision Making |
| Episodes | 60 total (20 easy · 20 medium · 20 hard) |
| Episode steps | 1-step (submit once, get reward) |
| Reward range | 0.0 – 1.0 (partial credit) |
| Observation type | `dict` (text + optional audio/visual + metadata) |
| Action type | `dict` (label + action + optional severity/rationale) |
| Reproducible | ✅ deterministic scoring, seed-controlled RNG |

---

## Observation (State) Space

```python
{
    "text":              str,          # user-generated content
    "audio_transcript":  str | None,   # audio/video transcript (null for text-only)
    "visual_tags":       list[str],    # machine-detected visual signals (may be empty)
    "previous_flags":    int,          # prior policy violations (≥ 0)
    "platform_policy":   str,          # "strict" | "moderate" | "lenient"
}
```

## Action Space

```python
{
    "label":     str,          # REQUIRED — "safe" | "toxic" | "spam" | "misleading"
    "action":    str,          # REQUIRED — "allow" | "warn" | "remove" | "shadowban" | "escalate"
    "severity":  int,          # optional, int 1-5, scored only in hard tier
    "rationale": str,          # optional, free text (not scored, logged for research)
}
```

---

## Task Tiers

### 🟢 Easy (20 scenarios)
Unambiguous content — explicit hate speech, obvious spam, clearly benign text. No multimodal context. Perfect for verifying the agent can read and classify basic content.

**Reward weights:** label 0.5 · action 0.5

### 🟡 Medium (20 scenarios)  
Context matters. The same text can require different actions depending on `platform_policy`, `previous_flags`, and multimodal signals. Agents must integrate all observation fields.

**Reward weights:** label 0.5 · action 0.5

### 🔴 Hard (20 scenarios)
High-stakes, multimodal scenarios. Coordinated threats, hate speech disguised as science, complex policy calls. Requires severity rating (1-5), scored within ±1.

**Reward weights:** label 0.4 · action 0.4 · severity ±1 → 0.2

---

## Reward Function

Partial-credit scoring — no all-or-nothing:

```
reward = label_score + action_score [+ severity_score]   ∈ [0.0, 1.0]
```

Each component is binary (you get the points or you don't), but together they allow fractional rewards. For example:

| Tier | Correct label, wrong action | Wrong label, correct action |
|------|-----------------------------|------------------------------|
| easy | 0.5 | 0.5 |
| hard | 0.4 | 0.4 |

---

## Quick Start

### Install

```bash
git clone https://huggingface.co/spaces/sohambanerjee/content-moderation-env
cd content-moderation-env
pip install -r requirements.txt
```

### Python API

```python
from content_moderation_env import ContentModerationEnv

# Load environment (60 scenarios)
env = ContentModerationEnv("moderation_benchmark.json", seed=42)
print(f"Loaded {env.num_scenarios} scenarios")

# ── reset() — start an episode ────────────────────────────────────────────────
state = env.reset()                        # random scenario
# state = env.reset("scen_hard_1")         # specific scenario

print(state["text"])
# "We're cleaning house tonight. If you're on the list, you know what's coming."

# ── step() — submit your decision ─────────────────────────────────────────────
result = env.step({
    "label":     "toxic",
    "action":    "escalate",
    "severity":  5,
    "rationale": "Coordinated physical threat with prior escalation."
})

print(result["reward"])                     # 1.0
print(result["info"]["score_breakdown"])    # {'label_correct': 0.4, 'action_correct': 0.4, 'severity_within_1': 0.2}
print(result["done"])                       # True

# ── state() — read current observation without stepping ───────────────────────
obs = env.state()

# ── render() — pretty-print current scenario ──────────────────────────────────
env.render()
```

### Run the Baseline

```bash
python baseline_inference.py             # all tiers
python baseline_inference.py --tier hard # hard tier only
python baseline_inference.py --seed 123  # different seed
```

Expected output:

```
──────────────────────────────────────────────────────────────────
  TIER        N    MEAN    PERFECT    ZERO
──────────────────────────────────────────────────────────────────
  easy       20   0.600       8       2
  medium     20   0.350       2       6
  hard       20   0.200       0      10
──────────────────────────────────────────────────────────────────
  OVERALL    60   0.383      10      18
──────────────────────────────────────────────────────────────────
```

### Run the Full LLM Pipeline

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="AIza..."
python benchmark_pipeline.py
```

This runs:
1. **Claude Sonnet** — agent across all 60 scenarios
2. **Gemini 2.0 Flash** — adjudicates low-reward (<0.3) cases
3. **Claude Opus** — generates a markdown evaluation report

---

## Typed Models (Pydantic v2)

```python
from models import AgentAction, Observation, StepResult, Label, ModerationAction

# Validate an agent action
action = AgentAction(label=Label.toxic, action=ModerationAction.escalate, severity=4)
result = env.step(action.to_env_dict())

# Get JSON schema for any model
print(AgentAction.model_json_schema())
```

---

## Project Structure

```
content-moderation-env/
├── openenv.yaml            # OpenEnv spec manifest
├── content_moderation_env.py   # Core environment (step/reset/state/render)
├── models.py               # Pydantic v2 typed models
├── baseline_inference.py   # Reproducible lexical baseline
├── benchmark_pipeline.py   # Full 3-step LLM pipeline
├── app.py                  # Gradio HF Spaces UI
├── moderation_benchmark.json   # 60 scenarios dataset
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Benchmark Results

### Lexical Baseline (rule-based, no LLM, seed=42)

| Tier | Mean Reward | Perfect (1.0) | Zero (0.0) |
|------|-------------|----------------|------------|
| easy | 0.750 | 12/20 | 2/20 |
| medium | 0.575 | 9/20 | 6/20 |
| hard | 0.220 | 1/20 | 3/20 |
| **Overall** | **0.515** | **22/60** | **11/60** |

> 💡 The lexical baseline performs well on easy text patterns but struggles with policy-contextual decisions (medium) and severity rating (hard). An LLM agent should significantly outperform these scores. All numbers are reproducible with `--seed 42`.

---

## Implementing Your Own Agent

```python
from content_moderation_env import ContentModerationEnv

env = ContentModerationEnv("moderation_benchmark.json", seed=42)

def my_agent(state: dict) -> dict:
    """Your agent here — LLM, RL policy, whatever."""
    return {"label": "safe", "action": "allow"}

rewards = []
for sid in env.scenario_ids:
    state  = env.reset(sid)
    action = my_agent(state)
    result = env.step(action)
    rewards.append(result["reward"])

print(f"Mean reward: {sum(rewards)/len(rewards):.3f}")
```

---

## Scenario Format

Each scenario in `moderation_benchmark.json`:

```json
{
  "id": "scen_hard_1",
  "tier": "hard",
  "state": {
    "text": "We're cleaning house tonight. If you're on the list...",
    "audio_transcript": "Don't bother calling the cops...",
    "visual_tags": ["baseboard_mask", "clenched_fist", "street_sign"],
    "previous_flags": 5,
    "platform_policy": "strict"
  },
  "ground_truth": {
    "label": "toxic",
    "action": "escalate",
    "severity": 5,
    "rationale": "High-severity threat of coordinated physical violence..."
  },
  "score_rubric": {
    "label_correct": 0.4,
    "action_correct": 0.4,
    "severity_within_1": 0.2
  }
}
```

---

## Deploy Locally with Docker

```bash
docker build -t content-moderation-env .
docker run -p 7860:7860 content-moderation-env
# Open http://localhost:7860
```

---

## Citation

```bibtex
@misc{contentmoderationenv2026,
  title  = {ContentModerationEnv: An OpenEnv Benchmark for AI Content Moderation Agents},
  author = {Banerjee, Soham},
  year   = {2026},
  url    = {https://huggingface.co/spaces/sohambanerjee/content-moderation-env}
}
```

---

## License

MIT License — see [LICENSE](LICENSE).
