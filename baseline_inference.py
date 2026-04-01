"""
baseline_inference.py
=====================
Reproducible rule-based baseline for the ContentModerationEnv benchmark.

This script implements a pure lexical/heuristic agent (no LLM, no API keys)
so anyone can reproduce baseline scores and verify the environment is working.

Usage
-----
    python baseline_inference.py
    python baseline_inference.py --tier easy
    python baseline_inference.py --tier hard --seed 123

Expected scores (seed=42, all tiers):
    easy   ≈ 0.60
    medium ≈ 0.35
    hard   ≈ 0.20
    overall ≈ 0.38

Output
------
  • Prints per-scenario results to stdout
  • Saves  baseline_results.json  in the same directory
  • Prints a summary table at the end
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

# ── local import ──────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
from content_moderation_env import ContentModerationEnv

# ── constants ─────────────────────────────────────────────────────────────────
SCENARIOS_PATH = SCRIPT_DIR / "moderation_benchmark.json"
RESULTS_PATH   = SCRIPT_DIR / "baseline_results.json"

_TOXIC_PATTERNS = re.compile(
    r"\b(kill|hate|loser|idiot|stupid|terrorist|harass|threat|die|hurt|"
    r"ugly|trash|worse|jerk|racist|filthy|scum|moron|dumb|sick|disgusting)\b",
    re.IGNORECASE,
)
_SPAM_PATTERNS = re.compile(
    r"\b(free|click|win|congratulation|giveaway|earn|bitcoin|dm me|"
    r"no survey|pills|meds|weight loss|work from home|discount)\b",
    re.IGNORECASE,
)
_MISLEADING_PATTERNS = re.compile(
    r"\b(hoax|conspiracy|fake news|government lies|they hide|cover.?up|"
    r"shadow group|election fraud|cure|literally cures|mercury in water)\b",
    re.IGNORECASE,
)

# ── Lexical heuristic agent ───────────────────────────────────────────────────

def _classify(state: dict) -> str:
    """Classify content by scanning text + audio_transcript."""
    text = state.get("text", "") or ""
    audio = state.get("audio_transcript", "") or ""
    combined = f"{text} {audio}".lower()

    if _misleading_patterns_match(combined):
        return "misleading"
    if _TOXIC_PATTERNS.search(combined):
        return "toxic"
    if _SPAM_PATTERNS.search(combined):
        return "spam"
    return "safe"


def _misleading_patterns_match(text: str) -> bool:
    return bool(_MISLEADING_PATTERNS.search(text))


def _choose_action(label: str, state: dict) -> str:
    """
    Rule-based action selection.
    Escalates for repeated offenders on strict platforms.
    Shadowbans repeat toxic offenders.
    Uses warn for first-time mild violations.
    """
    policy = state.get("platform_policy", "moderate")
    flags  = int(state.get("previous_flags", 0))

    if label == "safe":
        return "allow"

    if label == "spam":
        return "remove"               # always remove spam

    if label == "misleading":
        if policy == "strict":
            return "remove"
        return "warn"

    # toxic
    if flags >= 3 and policy == "strict":
        return "escalate"
    if flags >= 2:
        return "shadowban"
    if policy == "lenient":
        return "allow"                # lenient: don't act on mild toxic first offence
    if policy == "strict" and flags >= 1:
        return "remove"
    return "warn"                     # default graduated response


def _severity_estimate(state: dict) -> int:
    """
    Estimate severity 1-5 by counting threat signals.
    Always emits a severity (needed for hard tier).
    """
    score = 1
    flags = int(state.get("previous_flags", 0))
    policy = state.get("platform_policy", "moderate")
    text  = (state.get("text", "") or "").lower()
    audio = (state.get("audio_transcript", "") or "").lower()
    vtags = state.get("visual_tags", []) or []

    # prior violations
    score += min(flags, 2)

    # audio escalates
    if audio:
        score += 1

    # visual media escalates
    if vtags:
        score += 1

    # strict policy on top of existing severity
    if policy == "strict" and score >= 3:
        score += 1

    return min(score, 5)


def decide(state: dict) -> dict:
    """
    Full agent decision: label → action → severity → rationale.
    This is the single entry point used by run_baseline().
    """
    label   = _classify(state)
    action  = _choose_action(label, state)
    severity = _severity_estimate(state)
    return {
        "label":    label,
        "action":   action,
        "severity": severity,
        "rationale": "Lexical rule-based baseline decision.",
    }


# ── Runner ────────────────────────────────────────────────────────────────────

def run_baseline(
    tier_filter: Optional[str] = None,
    seed: int = 42,
    verbose: bool = True,
) -> list[dict]:
    """
    Run the rule-based agent over all (or filtered) scenarios.

    Parameters
    ----------
    tier_filter : str | None
        If provided, only run scenarios matching this tier (easy/medium/hard).
    seed : int
        RNG seed for the environment (irrelevant for deterministic runs, but
        ensures consistent scenario ordering).
    verbose : bool
        Print per-scenario results if True.

    Returns
    -------
    results : list[dict]
        One dict per scenario with reward, breakdown, agent decision, etc.
    """
    env = ContentModerationEnv(str(SCENARIOS_PATH), seed=seed)
    ids = env.scenario_ids
    if tier_filter:
        ids = [i for i in ids if i.startswith(f"scen_{tier_filter}")]

    results = []
    sep = "─" * 66

    if verbose:
        print(sep)
        print(f"  Lexical Baseline  ·  {len(ids)} scenarios  ·  seed={seed}")
        if tier_filter:
            print(f"  Tier filter: {tier_filter}")
        print(sep)

    for i, sid in enumerate(ids, 1):
        state   = env.reset(sid)
        action  = decide(state)
        result  = env.step(action)

        tier    = result["info"]["tier"]
        reward  = result["reward"]
        bd      = result["info"]["score_breakdown"]

        row = {
            "scenario_id":   sid,
            "tier":          tier,
            "reward":        reward,
            "breakdown":     bd,
            "agent_decision": action,
            "ground_truth":  result["info"]["ground_truth"],
            "state_text":    state["text"],
            "platform_policy": state["platform_policy"],
            "previous_flags":  state["previous_flags"],
        }
        results.append(row)

        if verbose:
            status = "✓" if reward >= 0.8 else ("~" if reward >= 0.4 else "✗")
            print(
                f"  [{i:02d}/{len(ids)}] {sid:<24} {status} "
                f"reward={reward:.2f}  "
                f"label={action['label']!r}→{result['info']['ground_truth']['label']!r}  "
                f"action={action['action']!r}→{result['info']['ground_truth']['action']!r}"
            )

    return results


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(results: list[dict]) -> None:
    """Print a tier-breakdown summary table."""
    sep = "─" * 66

    tiers = ["easy", "medium", "hard"]
    print(f"\n{sep}")
    print(f"  {'TIER':<10}  {'N':>4}  {'MEAN':>6}  {'PERFECT':>8}  {'ZERO':>6}")
    print(sep)

    all_rewards = [r["reward"] for r in results]
    for tier in tiers:
        rows = [r for r in results if r["tier"] == tier]
        if not rows:
            continue
        rewards = [r["reward"] for r in rows]
        mn  = sum(rewards) / len(rewards)
        pct = sum(1 for rw in rewards if rw == 1.0)
        zr  = sum(1 for rw in rewards if rw == 0.0)
        print(f"  {tier:<10}  {len(rows):>4}  {mn:>6.3f}  {pct:>7}  {zr:>6}")

    print(sep)
    overall = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
    print(
        f"  {'OVERALL':<10}  {len(all_rewards):>4}  {overall:>6.3f}  "
        f"{sum(1 for r in all_rewards if r==1.0):>7}  "
        f"{sum(1 for r in all_rewards if r==0.0):>6}"
    )
    print(sep)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run the rule-based lexical baseline on ContentModerationEnv."
    )
    parser.add_argument(
        "--tier",
        choices=["easy", "medium", "hard"],
        default=None,
        help="Only run scenarios in this tier (default: all tiers)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed (default: 42)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-scenario output",
    )
    args = parser.parse_args()

    results = run_baseline(
        tier_filter=args.tier,
        seed=args.seed,
        verbose=not args.quiet,
    )
    print_summary(results)

    RESULTS_PATH.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n  ✓ Results saved → {RESULTS_PATH.name}")


if __name__ == "__main__":
    main()
