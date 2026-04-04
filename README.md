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

# 🛡️ ContentModerationEnv v2.0

> **A real-world OpenEnv benchmark** for evaluating AI agents on the task of content moderation.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![OpenEnv v2.0](https://img.shields.io/badge/OpenEnv-v2.0-green.svg)](openenv.yaml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

`ContentModerationEnv` is a fully-spec'd OpenEnv environment simulating real-world platform content moderation. An agent reads user-generated content (text, audio transcript, visual tags), considers the poster's violation history and platform policy, then decides:

1. **Classify** the content: `safe | toxic | spam | misleading`
2. **Take action**: `allow | warn | remove | shadowban | escalate`
3. **Rate severity** (hard tier): `1` (mild) → `5` (critical)

Agents receive **partial-credit rewards with penalties** (`-0.3` to `1.0`) providing rich gradient signals for learning.

---

## What's New in v2

| Feature | v1 | v2 |
|---------|----|----|
| Scenarios | 60 | **75** (25 easy · 20 medium · 30 hard) |
| Episode type | Single-step only | **Queue mode** (3-post episode) + single-step compat |
| Reward range | `0.0–1.0` | **`-0.3–1.0`** (penalties) |
| Penalties | None | **4 penalty types** (false positive, missed threat, invalid label/action) |
| Inference script | Pipeline only | **`inference.py`** — hackathon `[START]/[STEP]/[END]` format |

---

## Environment Description

| Property | Value |
|----------|-------|
| Domain | Trust & Safety / NLP |
| Task type | Classification + Decision Making |
| Total scenarios | 75 (25 easy · 20 medium · 30 hard) |
| Episode modes | Queue (3-post, mixed tiers) + Single-step (backward compat) |
| Reward range | `-0.3` to `1.0` (partial credit + penalties) |
| Reproducible | ✅ deterministic scoring, seed-controlled RNG |

---

## Episode Modes

### Queue Mode (default — `reset()`)
```python
state = env.reset()       # samples 1 easy + 1 medium + 1 hard scenario
# Agent processes ALL 3 posts before the episode ends
while True:
    result = env.step(action)
    if result["done"]:    # done=False for first 2 posts, True on 3rd
        break
    state = result["state"]   # next post in queue
print(env.episode_rewards)    # [r1, r2, r3]
```

### Single-Step Mode (backward compatible — `reset(scenario_id=...)`)
```python
state = env.reset(scenario_id="scen_hard_1")
result = env.step(action)   # done=True immediately
```

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
    "label":     str,   # REQUIRED — "safe" | "toxic" | "spam" | "misleading"
    "action":    str,   # REQUIRED — "allow" | "warn" | "remove" | "shadowban" | "escalate"
    "severity":  int,   # optional, 1-5, scored only in hard tier (±1 credit)
    "rationale": str,   # optional, free text (not scored, logged for research)
}
```

---

## Reward Function

### Positive credit (partial)

| Tier | Label | Action | Severity ±1 |
|------|-------|--------|-------------|
| easy / medium | 0.5 | 0.5 | — |
| hard | 0.4 | 0.4 | 0.2 |

### Penalties (may push reward below 0)

| Penalty | Value | Trigger |
|---------|-------|---------|
| `false_positive_penalty` | **-0.25** | GT=`safe` but agent submits `remove` or `escalate` |
| `missed_critical_threat_penalty` | **-0.20** | GT=`toxic` severity=5 but agent `allow`s or `warn`s |
| `invalid_label_penalty` | **-0.10** | Label not in valid set |
| `invalid_action_penalty` | **-0.10** | Action not in valid set |

**Reward is clamped to `[-0.3, 1.0]`.**

---

## Task Tiers

### 🟢 Easy (25 scenarios)
Unambiguous content — explicit hate speech, obvious spam, clearly benign text. No multimodal context.

### 🟡 Medium (20 scenarios)
Context matters. Policy, repeat-flags, and multimodal signals determine the right action.

### 🔴 Hard (30 scenarios)
Multimodal signals, coordinated threats, non-English content, severity rating (1-5). All penalties apply.

---

## Quick Start

```bash
git clone https://github.com/oki-dokii/Meta.git
cd Meta
pip install -r requirements.txt
```

### Python API

```python
from content_moderation_env import ContentModerationEnv

env = ContentModerationEnv("moderation_benchmark.json", seed=42)
print(f"Loaded {env.num_scenarios} scenarios")  # 75

# ── Queue episode (3 posts) ───────────────────────────────────────────────────
state = env.reset()
while True:
    result = env.step({
        "label":    "toxic",
        "action":   "escalate",
        "severity": 5,
        "rationale": "Coordinated threat."
    })
    print(f"reward={result['reward']:.2f}  done={result['done']}")
    if result["done"]:
        break
    state = result["state"]
print(f"Episode rewards: {env.episode_rewards}")

# ── Single-step (backward compat) ─────────────────────────────────────────────
state = env.reset(scenario_id="scen_hard_1")
result = env.step({"label": "toxic", "action": "escalate", "severity": 5})
print(result["reward"])   # 1.0

# ── Penalty example ───────────────────────────────────────────────────────────
env.reset(scenario_id="scen_easy_1")   # GT: safe/allow
result = env.step({"label": "safe", "action": "escalate"})
print(result["reward"])   # -0.25 (false positive penalty)
print(result["info"]["score_breakdown"])
# {'label_correct': 0.5, 'action_correct': 0.0, 'false_positive_penalty': -0.25}
```

### Run the Baseline (no API key needed)

```bash
python3 baseline_inference.py             # all 75 scenarios
python3 baseline_inference.py --tier hard # hard tier only
```

### Run the LLM Inference Script (hackathon format)

```bash
export OPENAI_API_KEY="sk-..."          # or HF_TOKEN for HF inference
export MODEL_NAME="gpt-4o-mini"         # default
python3 inference.py
# Outputs: [START] / [STEP] / [END] lines per task
```

### Run the Validation Suite

```bash
python3 validate_openenv.py
# Checks: YAML structure, dataset integrity, live API, reward bounds
```

---

## Inference Script Output Format

`inference.py` emits the standardised hackathon format:

```
[START] task=easy_moderation env=content_moderation model=gpt-4o-mini
[STEP] step=1 action={"label":"toxic","action":"remove"} reward=1.00 done=false error=null
[STEP] step=2 action={"label":"safe","action":"allow"} reward=1.00 done=false error=null
...
[END] success=true steps=25 rewards=1.00,0.50,1.00,...
```

Configure via environment variables:

| Variable | Default | Purpose |
|----------|---------|---------|
| `HF_TOKEN` or `OPENAI_API_KEY` | — | API key |
| `API_BASE_URL` | `https://api.openai.com/v1` | Any OpenAI-compatible endpoint |
| `MODEL_NAME` | `gpt-4o-mini` | Model identifier |

---

## Baseline Scores (lexical agent, seed=42)

| Tier | N | Mean Reward | Perfect (1.0) | Zero (0.0) |
|------|---|-------------|----------------|------------|
| easy | 25 | 0.660 | 12 | 4 |
| medium | 20 | 0.575 | 9 | 6 |
| hard | 30 | 0.213 | 2 | 11 |
| **Overall** | **75** | **0.459** | **23** | **21** |

> 💡 The lexical baseline scores are lower in v2 due to harder scenarios and penalties. Beat 0.459 with an LLM agent.

---

## Project Structure

```
Meta/
├── openenv.yaml               # OpenEnv v2 spec manifest
├── content_moderation_env.py  # Core env — step/reset/state/render, queue mode, penalties
├── models.py                  # Pydantic v2 typed models
├── moderation_benchmark.json  # 75 scenarios dataset
├── baseline_inference.py      # Reproducible lexical baseline (no API key)
├── inference.py               # Hackathon-format LLM inference ([START]/[STEP]/[END])
├── benchmark_pipeline.py      # Full 3-step pipeline (Claude + Gemini)
├── benchmark_pipeline_gemini.py # Gemini-only pipeline
├── validate_openenv.py        # Self-contained validation suite
├── app.py                     # Gradio Hugging Face Spaces UI
├── Dockerfile                 # HF Spaces deployment
├── requirements.txt
└── README.md
```

---

## Deploy Locally with Docker

```bash
docker build -t content-moderation-env .
docker run -p 7860:7860 content-moderation-env
# → http://localhost:7860
```

---

## Citation

```bibtex
@misc{contentmoderationenv2026,
  title  = {ContentModerationEnv: An OpenEnv Benchmark for AI Content Moderation Agents},
  author = {Banerjee, Soham},
  year   = {2026},
  url    = {https://github.com/oki-dokii/Meta}
}
```

---

## License

MIT License.
