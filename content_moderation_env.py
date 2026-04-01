"""
ContentModerationEnv
====================
An OpenEnv-compatible environment for evaluating AI content moderation agents
against a pre-built scenario benchmark.

Usage:
    env = ContentModerationEnv("moderation_benchmark.json")
    state = env.reset()                       # random scenario
    state = env.reset(scenario_id="scen_hard_7")  # specific scenario
    result = env.step({
        "label": "toxic",
        "action": "escalate",
        "severity": 5,                        # hard tier only
        "rationale": "This is doxxing ..."    # hard tier only
    })
    env.render()
"""

import json
import os
import random
import textwrap
from copy import deepcopy
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_scenarios(path: str) -> Dict[str, dict]:
    """Load scenarios.json and index by id."""
    with open(path, "r", encoding="utf-8") as f:
        raw: List[dict] = json.load(f)
    return {s["id"]: s for s in raw}


def _compute_reward(action: dict, scenario: dict) -> tuple[float, dict]:
    """
    Deterministic scorer.

    Returns
    -------
    reward : float  – 0.0 … 1.0
    breakdown : dict – per-component scores for debugging
    """
    gt = scenario["ground_truth"]
    rubric = scenario["score_rubric"]
    tier = scenario["tier"]

    breakdown: dict = {}
    reward = 0.0

    # ── Label ──────────────────────────────────────────────────────────────
    label_ok = str(action.get("label", "")).strip().lower() == str(gt["label"]).strip().lower()
    label_score = rubric["label_correct"] if label_ok else 0.0
    breakdown["label_correct"] = label_score
    reward += label_score

    # ── Action ─────────────────────────────────────────────────────────────
    action_ok = str(action.get("action", "")).strip().lower() == str(gt["action"]).strip().lower()
    action_score = rubric["action_correct"] if action_ok else 0.0
    breakdown["action_correct"] = action_score
    reward += action_score

    # ── Severity (hard tier only) ───────────────────────────────────────────
    if tier == "hard" and "severity_within_1" in rubric:
        gt_severity = int(gt.get("severity", 0))
        pred_severity = action.get("severity")
        if pred_severity is not None:
            within_1 = abs(int(pred_severity) - gt_severity) <= 1
            sev_score = rubric["severity_within_1"] if within_1 else 0.0
        else:
            sev_score = 0.0
        breakdown["severity_within_1"] = sev_score
        reward += sev_score

    # Clamp to [0.0, 1.0] for safety
    reward = max(0.0, min(1.0, round(reward, 6)))
    return reward, breakdown


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class ContentModerationEnv:
    """
    OpenEnv-compatible content moderation environment.

    Parameters
    ----------
    scenarios_path : str
        Path to the JSON file containing benchmark scenarios.
    seed : int | None
        Optional RNG seed for reproducible random episode selection.
    """

    # Supported labels and actions (for validation)
    VALID_LABELS = {"safe", "toxic", "spam", "misleading"}
    VALID_ACTIONS = {"allow", "warn", "remove", "shadowban", "escalate"}

    def __init__(self, scenarios_path: str, seed: Optional[int] = None):
        if not os.path.isfile(scenarios_path):
            raise FileNotFoundError(f"Scenarios file not found: {scenarios_path!r}")

        self._scenarios: Dict[str, dict] = _load_scenarios(scenarios_path)
        self._scenario_ids: List[str] = sorted(self._scenarios.keys())
        self._rng = random.Random(seed)

        # Runtime state
        self._current_scenario: Optional[dict] = None
        self._done: bool = True
        self._last_reward: float = 0.0
        self._last_breakdown: dict = {}
        self._step_count: int = 0

    # ── Core API ────────────────────────────────────────────────────────────

    def reset(self, scenario_id: Optional[str] = None) -> dict:
        """
        Begin a new episode.

        Parameters
        ----------
        scenario_id : str | None
            If provided, loads that specific scenario.
            If None, picks one uniformly at random.

        Returns
        -------
        state : dict  – the scenario's `state` sub-dict
        """
        if scenario_id is not None:
            if scenario_id not in self._scenarios:
                available = ", ".join(self._scenario_ids[:5]) + " …"
                raise ValueError(
                    f"Unknown scenario_id {scenario_id!r}. "
                    f"Available (sample): {available}"
                )
            chosen_id = scenario_id
        else:
            chosen_id = self._rng.choice(self._scenario_ids)

        self._current_scenario = deepcopy(self._scenarios[chosen_id])
        self._done = False
        self._last_reward = 0.0
        self._last_breakdown = {}
        self._step_count = 0

        return deepcopy(self._current_scenario["state"])

    def step(self, action: dict) -> dict:
        """
        Submit a moderation decision for the current scenario.

        Parameters
        ----------
        action : dict
            {
              "label":    str,          # required
              "action":   str,          # required
              "severity": int,          # optional; scored in hard tier
              "rationale": str          # optional; not scored (future use)
            }

        Returns
        -------
        result : dict
            {
              "state":   dict,   – current scenario state (unchanged)
              "reward":  float,  – 0.0 … 1.0
              "done":    bool,   – always True after one step
              "info":    dict    – breakdown + ground truth + metadata
            }
        """
        if self._current_scenario is None or self._done:
            raise RuntimeError(
                "Environment is not active. Call reset() before step()."
            )
        if not isinstance(action, dict):
            raise TypeError(f"action must be a dict, got {type(action).__name__!r}")

        # Validate keys (warn but don't hard-fail, to allow partial submissions)
        warnings: List[str] = []
        label = str(action.get("label", "")).strip().lower()
        act   = str(action.get("action", "")).strip().lower()
        if label not in self.VALID_LABELS:
            warnings.append(f"Unknown label {label!r}; valid: {sorted(self.VALID_LABELS)}")
        if act not in self.VALID_ACTIONS:
            warnings.append(f"Unknown action {act!r}; valid: {sorted(self.VALID_ACTIONS)}")

        reward, breakdown = _compute_reward(action, self._current_scenario)

        self._last_reward = reward
        self._last_breakdown = breakdown
        self._done = True
        self._step_count += 1

        result = {
            "state":  deepcopy(self._current_scenario["state"]),
            "reward": reward,
            "done":   True,
            "info": {
                "scenario_id":   self._current_scenario["id"],
                "tier":          self._current_scenario["tier"],
                "ground_truth":  deepcopy(self._current_scenario["ground_truth"]),
                "score_rubric":  deepcopy(self._current_scenario["score_rubric"]),
                "score_breakdown": breakdown,
                "submitted_action": deepcopy(action),
                "warnings":      warnings,
            },
        }
        return result

    def state(self) -> dict:
        """Return the current scenario state dict (read-only copy)."""
        if self._current_scenario is None:
            raise RuntimeError("No active scenario. Call reset() first.")
        return deepcopy(self._current_scenario["state"])

    def render(self, mode: str = "text") -> None:
        """
        Pretty-print the current scenario and last step result.

        Parameters
        ----------
        mode : str
            Only "text" is supported.
        """
        if self._current_scenario is None:
            print("[ContentModerationEnv] No active scenario. Call reset() first.")
            return

        sc = self._current_scenario
        gt = sc["ground_truth"]
        st = sc["state"]

        sep = "─" * 62
        print(sep)
        print(f"  Scenario : {sc['id']}  │  Tier: {sc['tier'].upper()}")
        print(sep)
        print(f"  Text     : {textwrap.shorten(st['text'], 80)}")
        if st.get("audio_transcript"):
            print(f"  Audio    : {textwrap.shorten(st['audio_transcript'], 80)}")
        if st.get("visual_tags"):
            print(f"  Visual   : {', '.join(st['visual_tags'])}")
        print(f"  Prev flags: {st['previous_flags']}  │  Policy: {st['platform_policy']}")
        print()
        print(f"  Ground truth ▶  label={gt['label']}  action={gt['action']}", end="")
        if "severity" in gt:
            print(f"  severity={gt['severity']}", end="")
        print()
        if "rationale" in gt:
            print(f"  Rationale: {textwrap.shorten(gt['rationale'], 100)}")
        print()

        if self._done and self._last_breakdown:
            print(f"  Last reward : {self._last_reward:.3f}")
            print(f"  Breakdown   : {self._last_breakdown}")
        print(sep)

    # ── Convenience ─────────────────────────────────────────────────────────

    @property
    def scenario_ids(self) -> List[str]:
        """Sorted list of all available scenario IDs."""
        return list(self._scenario_ids)

    @property
    def num_scenarios(self) -> int:
        return len(self._scenarios)

    def __repr__(self) -> str:
        active = self._current_scenario["id"] if self._current_scenario else "None"
        return (
            f"ContentModerationEnv("
            f"scenarios={self.num_scenarios}, "
            f"active={active!r}, "
            f"done={self._done})"
        )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main():
    # Resolve path relative to this file's directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    scenarios_path = os.path.join(base_dir, "moderation_benchmark.json")

    print("=" * 62)
    print("  ContentModerationEnv — Demo")
    print("=" * 62)

    env = ContentModerationEnv(scenarios_path, seed=42)
    print(f"Loaded {env.num_scenarios} scenarios.\n")

    # ── Episode 1: Easy tier, perfect action ─────────────────────────────
    print("▶ Episode 1  — Easy tier, scen_easy_2 (perfect answer)")
    state = env.reset(scenario_id="scen_easy_2")
    print(f"  State text: {state['text']!r}")

    result = env.step({"label": "toxic", "action": "remove"})
    print(f"  Reward : {result['reward']}  (expected 1.0)")
    print(f"  Breakdown: {result['info']['score_breakdown']}")
    env.render()

    # ── Episode 2: Easy tier, wrong label ───────────────────────────────────
    print("\n▶ Episode 2  — Easy tier, scen_easy_2 (wrong label, correct action)")
    env.reset(scenario_id="scen_easy_2")
    result = env.step({"label": "spam", "action": "remove"})
    print(f"  Reward : {result['reward']}  (expected 0.5 — label wrong)")
    print(f"  Breakdown: {result['info']['score_breakdown']}")

    # ── Episode 3: Medium tier, partial ─────────────────────────────────────
    print("\n▶ Episode 3  — Medium tier, scen_medium_5 (correct label, wrong action)")
    env.reset(scenario_id="scen_medium_5")
    result = env.step({"label": "toxic", "action": "warn"})  # GT action=shadowban
    print(f"  Reward : {result['reward']}  (expected 0.5 — action wrong)")
    print(f"  Breakdown: {result['info']['score_breakdown']}")
    print(f"  Ground truth: {result['info']['ground_truth']}")

    # ── Episode 4: Hard tier, near miss on severity ──────────────────────────
    print("\n▶ Episode 4  — Hard tier, scen_hard_1 (severity ±1)")
    env.reset(scenario_id="scen_hard_1")
    result = env.step({
        "label": "toxic",
        "action": "escalate",
        "severity": 4,          # GT=5, within ±1 → gets severity credit
        "rationale": "Coordinated threat with prior violations."
    })
    print(f"  Reward : {result['reward']}  (expected 1.0 — all correct, sev within 1)")
    print(f"  Breakdown: {result['info']['score_breakdown']}")

    # ── Episode 5: Hard tier, severity too far off ─────────────────────────
    print("\n▶ Episode 5  — Hard tier, scen_hard_1 (severity off by 2)")
    env.reset(scenario_id="scen_hard_1")
    result = env.step({
        "label": "toxic",
        "action": "escalate",
        "severity": 2,          # GT=5, off by 3 → no severity credit
        "rationale": "Some concern."
    })
    print(f"  Reward : {result['reward']}  (expected 0.8 — sev wrong)")
    print(f"  Breakdown: {result['info']['score_breakdown']}")

    # ── Episode 6: Random episode ────────────────────────────────────────────
    print("\n▶ Episode 6  — Random scenario, dummy all-wrong action")
    state = env.reset()
    print(f"  Current scenario: {env}")
    result = env.step({"label": "safe", "action": "allow", "severity": 1})
    print(f"  Reward : {result['reward']}")
    env.render()

    # ── Step guard: calling step() twice without reset ───────────────────────
    print("\n▶ Episode 7  — Guard: step() on done env raises RuntimeError")
    try:
        env.step({"label": "safe", "action": "allow"})
    except RuntimeError as e:
        print(f"  ✓ Caught expected error: {e}")

    print("\n✅  All episodes completed successfully.")


if __name__ == "__main__":
    main()
