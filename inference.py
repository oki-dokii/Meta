"""
inference.py — OpenAI LLM agent for ContentModerationEnv
=========================================================
Hackathon-compliant inference script for the OpenEnv Content Moderation
benchmark. Uses the OpenAI client to drive an LLM agent through 3 task
episodes (easy / medium / hard moderation), then emits the exact stdout
format required for automated evaluation scoring.

Credentials (read from environment variables):
    API_BASE_URL  — LLM API endpoint (default: https://api.openai.com/v1)
    MODEL_NAME    — model identifier   (default: gpt-4o-mini)
    HF_TOKEN      — API key (falls back to OPENAI_API_KEY)

Stdout format (zero deviation allowed):
    [START] task=<name> env=content_moderation model=<model>
    [STEP]  step=<n> action=<json> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...>
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

from openai import OpenAI

# ── local import ──────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
from content_moderation_env import ContentModerationEnv

# ── Credentials ───────────────────────────────────────────────────────────────
API_BASE_URL: str = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-4o-mini")
API_KEY: Optional[str] = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")

# ── Constants ─────────────────────────────────────────────────────────────────
SCENARIOS_PATH = SCRIPT_DIR / "moderation_benchmark.json"
ENV_NAME = "content_moderation"

# Tasks and their scenario ID slices (5 scenarios each)
TASKS = [
    {
        "name": "easy_moderation",
        "tier": "easy",
        "scenario_ids": [f"scen_easy_{i}" for i in range(1, 6)],
    },
    {
        "name": "medium_moderation",
        "tier": "medium",
        "scenario_ids": [f"scen_medium_{i}" for i in range(1, 6)],
    },
    {
        "name": "hard_moderation",
        "tier": "hard",
        "scenario_ids": [f"scen_hard_{i}" for i in range(1, 6)],
    },
]

# Fallback action used when the LLM response cannot be parsed
FALLBACK_ACTION: Dict = {"label": "safe", "action": "allow"}

# ── Logging helpers (exact stdout format) ─────────────────────────────────────

def log_start(task_name: str, model: str) -> None:
    """Emit [START] line."""
    print(f"[START] task={task_name} env={ENV_NAME} model={model}", flush=True)


def log_step(
    step: int,
    action: Dict,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    """Emit [STEP] line.

    Rules:
    - action serialised as compact JSON (no spaces)
    - reward: exactly 2 decimal places
    - done: lowercase string 'true' or 'false'
    - error: raw string or the literal word  null  (not quoted 'null', not None)
    """
    action_str = json.dumps(action, separators=(",", ":"))
    reward_str = f"{reward:.2f}"
    done_str = "true" if done else "false"
    error_str = error if error is not None else "null"
    print(
        f"[STEP] step={step} action={action_str} reward={reward_str} "
        f"done={done_str} error={error_str}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    """Emit [END] line.

    Rules:
    - success: lowercase string 'true' or 'false'
    - rewards: comma-separated, each exactly 2 decimal places
    """
    success_str = "true" if success else "false"
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={success_str} steps={steps} rewards={rewards_str}",
        flush=True,
    )


# ── Prompt builder ────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are an expert content moderation agent for a social media platform.
You will be shown a piece of content and must make a moderation decision.

You must respond with ONLY a valid JSON object, nothing else.
No explanation, no markdown, no code blocks. Just the raw JSON.

For easy and medium scenarios respond with:
{"label": "<label>", "action": "<action>"}

For hard scenarios also include severity:
{"label": "<label>", "action": "<action>", "severity": <1-5>, "rationale": "<brief reason>"}

Valid labels: safe, toxic, spam, misleading
Valid actions: allow, warn, remove, shadowban, escalate
severity: integer 1 (mild) to 5 (critical) — include only for hard tier\
"""


def build_prompt(state: Dict, tier: str) -> str:
    """Build the user-turn content from an environment state dict."""
    audio = state.get("audio_transcript") or "None"
    vtags = state.get("visual_tags") or []
    vtags_str = ", ".join(vtags) if vtags else "None"
    flags = state.get("previous_flags", 0)
    policy = state.get("platform_policy", "moderate")

    return (
        f"Content to moderate:\n"
        f"Text: {state['text']}\n"
        f"Audio transcript: {audio}\n"
        f"Visual tags: {vtags_str}\n"
        f"Previous violations by this account: {flags}\n"
        f"Platform policy: {policy}\n"
        f"Tier: {tier}\n\n"
        f"Make your moderation decision."
    )


# ── LLM action getter ─────────────────────────────────────────────────────────

def get_llm_action(
    client: OpenAI,
    state: Dict,
    tier: str,
) -> tuple[Dict, Optional[str]]:
    """
    Call the LLM and parse its JSON response into an action dict.

    Returns
    -------
    action : dict   — parsed action (or FALLBACK_ACTION on failure)
    error  : str | None — error message if something went wrong, else None
    """
    user_content = build_prompt(state, tier)
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.1,
            max_tokens=150,
        )
        raw = response.choices[0].message.content or ""
        raw = raw.strip()

        # Strip markdown fences if the model returns them despite instructions
        if raw.startswith("```"):
            lines = raw.splitlines()
            # Remove opening fence (```json or ```)
            lines = lines[1:] if lines else lines
            # Remove closing fence
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            raw = "\n".join(lines).strip()

        parsed = json.loads(raw)

        # Validate required keys; fall back silently if malformed
        label = str(parsed.get("label", "")).strip().lower()
        action = str(parsed.get("action", "")).strip().lower()

        valid_labels = {"safe", "toxic", "spam", "misleading"}
        valid_actions = {"allow", "warn", "remove", "shadowban", "escalate"}

        if label not in valid_labels:
            label = FALLBACK_ACTION["label"]
        if action not in valid_actions:
            action = FALLBACK_ACTION["action"]

        result: Dict = {"label": label, "action": action}

        # Include severity for hard tier (or if the model provided it)
        if "severity" in parsed:
            try:
                sev = int(parsed["severity"])
                sev = max(1, min(5, sev))
                result["severity"] = sev
            except (ValueError, TypeError):
                pass

        # Include rationale if provided (not scored, but useful for logging)
        if "rationale" in parsed and isinstance(parsed["rationale"], str):
            result["rationale"] = parsed["rationale"][:500]  # cap length

        return result, None

    except Exception as exc:  # noqa: BLE001
        # Return fallback and surface the error message in the [STEP] log
        return dict(FALLBACK_ACTION), str(exc)


# ── Task runner ───────────────────────────────────────────────────────────────

def run_task(
    client: OpenAI,
    env: ContentModerationEnv,
    task_name: str,
    scenario_ids: List[str],
    tier: str,
) -> None:
    """
    Run one complete task episode and emit [START] / [STEP]* / [END] lines.

    Each scenario is executed in single-step mode (scenario_id=...) for
    reproducibility. The while-not-done loop handles the queue episode
    structure, where done=False signals more posts remain in the episode.

    Parameters
    ----------
    client       : OpenAI client instance
    env          : initialised ContentModerationEnv
    task_name    : name string used in log lines
    scenario_ids : list of scenario IDs to include in this task
    tier         : "easy" | "medium" | "hard"
    """
    rewards: List[float] = []
    steps_taken = 0
    success = False

    log_start(task_name, MODEL_NAME)

    try:
        for sid in scenario_ids:
            # Single-step mode per scenario (reproducible, deterministic order)
            try:
                state = env.reset(scenario_id=sid)
            except Exception as exc:  # noqa: BLE001
                # Can't even reset — log a zero-reward step and move on
                steps_taken += 1
                rewards.append(0.0)
                log_step(steps_taken, dict(FALLBACK_ACTION), 0.0, True,
                         f"reset_error:{exc}")
                continue

            done = False
            while not done:
                step_error: Optional[str] = None

                # Get LLM decision (returns tuple: action, error_or_None)
                action, llm_error = get_llm_action(client, state, tier)
                if llm_error:
                    step_error = f"llm_error:{llm_error}"

                # Submit to environment
                reward = 0.0
                try:
                    result = env.step(action)
                    reward = result["reward"]
                    done = result["done"]
                    # Advance state if episode continues
                    if not done:
                        state = result["state"]
                except Exception as exc:  # noqa: BLE001
                    done = True  # treat step failure as terminal
                    step_error = (step_error or "") + f" step_error:{exc}"

                steps_taken += 1
                rewards.append(reward)
                log_step(steps_taken, action, reward, done, step_error)

        mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
        success = mean_reward >= 0.1

    except Exception as exc:  # noqa: BLE001
        print(f"[DEBUG] Task fatal error: {exc}", flush=True)

    finally:
        log_end(success, steps_taken, rewards)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """Entry point — run all 3 task episodes."""
    if not API_KEY:
        # Still emit output so the format check passes; auth will fail at LLM call
        sys.stderr.write(
            "WARNING: No API key found. Set HF_TOKEN or OPENAI_API_KEY.\n"
        )

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "no-key")
    env = ContentModerationEnv(str(SCENARIOS_PATH), seed=42)

    for task in TASKS:
        run_task(
            client=client,
            env=env,
            task_name=task["name"],
            scenario_ids=task["scenario_ids"],
            tier=task["tier"],
        )


if __name__ == "__main__":
    main()
