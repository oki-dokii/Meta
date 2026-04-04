"""
ContentModerationEnv
====================
An OpenEnv-compatible environment for evaluating AI content moderation agents
against a pre-built scenario benchmark.

Usage:
    env = ContentModerationEnv("moderation_benchmark.json")
    state = env.reset()                       # 3-post queue episode
    state = env.reset(scenario_id="scen_hard_7")  # single-step (backward compat)
    result = env.step({
        "label": "toxic",
        "action": "escalate",
        "severity": 5,
        "rationale": "This is doxxing ..."
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
    Deterministic scorer with penalty support.

    Returns
    -------
    reward : float  -- -0.3 ... 1.0 (penalties may make it negative)
    breakdown : dict -- per-component scores for debugging
    """
    gt = scenario["ground_truth"]
    rubric = scenario["score_rubric"]
    tier = scenario["tier"]

    breakdown: dict = {}
    reward = 0.0

    # -- Label ------------------------------------------------------------------
    label_submitted = str(action.get("label", "")).strip().lower()
    action_submitted = str(action.get("action", "")).strip().lower()
    gt_label = str(gt.get("label", "")).strip().lower()
    gt_action = str(gt.get("action", "")).strip().lower()

    label_ok = label_submitted == gt_label
    label_score = rubric["label_correct"] if label_ok else 0.0
    breakdown["label_correct"] = label_score
    reward += label_score

    # -- Action -----------------------------------------------------------------
    # Support valid_actions list for ambiguous scenarios (gap 3 fix).
    # If the ground_truth has a valid_actions list, any action in that list
    # earns full credit. Falls back to single gt_action for backward compat.
    valid_actions_gt: list = gt.get("valid_actions", [gt_action])
    action_ok = action_submitted in valid_actions_gt
    action_score = rubric["action_correct"] if action_ok else 0.0
    breakdown["action_correct"] = action_score
    reward += action_score

    # FP penalty also fires for any action in {escalate, remove} on safe GT
    # (unchanged — valid_actions on safe scenarios are not expected)

    # -- Severity (hard tier only) ----------------------------------------------
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

    # -- Penalties --------------------------------------------------------------

    # Penalty 1: False positive escalation (-0.25)
    # Removing/escalating content the GT says is safe is documented real-world harm.
    if gt_label == "safe" and action_submitted in {"escalate", "remove"}:
        breakdown["false_positive_penalty"] = -0.25
        reward += -0.25

    # Penalty 2: Missed critical threat (-0.20)
    # Missing a severity-5 threat is the worst moderation failure.
    if (
        gt_label == "toxic"
        and int(gt.get("severity", 0)) == 5
        and action_submitted in {"allow", "warn"}
    ):
        breakdown["missed_critical_threat_penalty"] = -0.20
        reward += -0.20

    # Penalty 3: Invalid label (-0.10)
    if label_submitted not in {"safe", "toxic", "spam", "misleading"}:
        breakdown["invalid_label_penalty"] = -0.10
        reward += -0.10

    # Penalty 4: Invalid action (-0.10)
    if action_submitted not in {"allow", "warn", "remove", "shadowban", "escalate"}:
        breakdown["invalid_action_penalty"] = -0.10
        reward += -0.10

    # Clamp to [-0.3, 1.0]
    reward = max(-0.3, min(1.0, round(reward, 6)))
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

    Episode modes
    -------------
    Queue mode  : reset()                    -- 3-post episode (1 easy+1 med+1 hard)
    Single-step : reset(scenario_id="...")   -- backward-compatible, done=True after 1 step
    """

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

        # Queue episode state
        self._queue: List[dict] = []
        self._queue_index: int = 0
        self._episode_rewards: List[float] = []

    # -- Core API ---------------------------------------------------------------

    def reset(self, scenario_id: Optional[str] = None) -> dict:
        """
        Begin a new episode.

        Parameters
        ----------
        scenario_id : str | None
            If provided, loads that specific scenario (single-step mode,
            backward compatible). If None, samples a mixed-tier queue of
            3 scenarios (1 easy + 1 medium + 1 hard) for a multi-step episode.

        Returns
        -------
        state : dict  -- the first scenario's observation
        """
        self._episode_rewards = []
        self._queue_index = 0
        self._last_reward = 0.0
        self._last_breakdown = {}
        self._step_count = 0
        self._done = False

        if scenario_id is not None:
            # -- Single-step mode (backward compatible) -------------------------
            if scenario_id not in self._scenarios:
                available = ", ".join(self._scenario_ids[:5]) + " ..."
                raise ValueError(
                    f"Unknown scenario_id {scenario_id!r}. "
                    f"Available (sample): {available}"
                )
            self._queue = [deepcopy(self._scenarios[scenario_id])]
        else:
            # -- Queue mode: try to build a coordinated cluster episode ---------
            # 25% chance: pick a random coordination_cluster and use its 3 posts
            # 75% chance: 1 easy + 1 medium + 1 hard (standard mixed queue)
            cluster_ids = self._rng.choice([None, None, None,
                                            "coord"])
            coord_scenarios = [
                s for s in self._scenarios.values()
                if "coordination_cluster" in s
            ]
            clusters: dict = {}
            for s in coord_scenarios:
                c = s["coordination_cluster"]
                clusters.setdefault(c, []).append(s)
            full_clusters = [k for k, v in clusters.items() if len(v) == 3]

            if full_clusters and cluster_ids == "coord":
                chosen_cluster = self._rng.choice(full_clusters)
                queue_scenarios = sorted(
                    clusters[chosen_cluster],
                    key=lambda s: {"easy": 0, "medium": 1, "hard": 2}.get(s["tier"], 1)
                )
                self._queue = [deepcopy(s) for s in queue_scenarios]
                self._active_cluster = chosen_cluster
            else:
                easy_ids   = [i for i in self._scenario_ids if i.startswith("scen_easy_")]
                medium_ids = [i for i in self._scenario_ids if i.startswith("scen_medium_")]
                hard_ids   = [i for i in self._scenario_ids if i.startswith("scen_hard_")]
                sampled = [
                    self._rng.choice(easy_ids),
                    self._rng.choice(medium_ids),
                    self._rng.choice(hard_ids),
                ]
                self._queue = [deepcopy(self._scenarios[sid]) for sid in sampled]
                self._active_cluster = None

        self._current_scenario = self._queue[0]
        if scenario_id is not None:
            self._active_cluster = None   # single-step: no cluster tracking
        return deepcopy(self._current_scenario["state"])

    def step(self, action: dict) -> dict:
        """
        Submit a moderation decision for the current scenario.

        Parameters
        ----------
        action : dict
            {
              "label":     str,   # required: safe/toxic/spam/misleading
              "action":    str,   # required: allow/warn/remove/shadowban/escalate
              "severity":  int,   # optional; scored in hard tier (1-5)
              "rationale": str    # optional; not scored
            }

        Returns
        -------
        result : dict
            {
              "state":   dict,   -- next observation (next queued post, or current if done)
              "reward":  float,  -- -0.3 ... 1.0 (with penalty support)
              "done":    bool,   -- False until final post in queue is processed
              "info":    dict    -- breakdown + ground truth + queue metadata
            }
        """
        if self._current_scenario is None or self._done:
            raise RuntimeError(
                "Environment is not active. Call reset() before step()."
            )
        if not isinstance(action, dict):
            raise TypeError(f"action must be a dict, got {type(action).__name__!r}")

        # Validate keys (warn but don't hard-fail, allows partial submissions)
        warnings: List[str] = []
        label = str(action.get("label", "")).strip().lower()
        act   = str(action.get("action", "")).strip().lower()
        if label not in self.VALID_LABELS:
            warnings.append(f"Unknown label {label!r}; valid: {sorted(self.VALID_LABELS)}")
        if act not in self.VALID_ACTIONS:
            warnings.append(f"Unknown action {act!r}; valid: {sorted(self.VALID_ACTIONS)}")

        # Capture scenario being processed BEFORE advancing queue
        processed_scenario = self._current_scenario
        reward, breakdown = _compute_reward(action, processed_scenario)

        self._last_reward = reward
        self._last_breakdown = breakdown
        self._episode_rewards.append(reward)
        self._step_count += 1
        self._queue_index += 1

        # Advance queue or mark episode done
        if self._queue_index >= len(self._queue):
            self._done = True
            next_state = deepcopy(processed_scenario["state"])
        else:
            self._done = False
            self._current_scenario = self._queue[self._queue_index]
            next_state = deepcopy(self._current_scenario["state"])

        # Coordination cluster bonus: if all posts in a cluster are correctly
        # labelled and actioned, add 0.1 bonus to the FINAL step's reward.
        cluster_bonus = 0.0
        if self._done and self._active_cluster:
            cluster_correct = all(
                r >= 0.8 for r in self._episode_rewards
            )
            if cluster_correct:
                cluster_bonus = 0.10
                breakdown["cluster_coordination_bonus"] = cluster_bonus
                reward = min(1.0, reward + cluster_bonus)
                self._last_reward = reward
                self._episode_rewards[-1] = reward  # update final step

        result = {
            "state":  next_state,
            "reward": reward,
            "done":   self._done,
            "info": {
                "scenario_id":         processed_scenario["id"],
                "tier":                processed_scenario["tier"],
                "queue_position":      self._queue_index,
                "queue_length":        len(self._queue),
                "episode_rewards":     list(self._episode_rewards),
                "ground_truth":        deepcopy(processed_scenario["ground_truth"]),
                "score_rubric":        deepcopy(processed_scenario["score_rubric"]),
                "score_breakdown":     breakdown,
                "submitted_action":    deepcopy(action),
                "warnings":            warnings,
                "coordination_cluster": processed_scenario.get("coordination_cluster"),
                "coordination_note":   processed_scenario.get("coordination_note"),
                "cluster_bonus_earned": cluster_bonus > 0,
            },
        }
        return result

    def state(self) -> dict:
        """Return the current scenario state dict (read-only copy)."""
        if self._current_scenario is None:
            raise RuntimeError("No active scenario. Call reset() first.")
        return deepcopy(self._current_scenario["state"])

    def render(self, mode: str = "text") -> None:
        """Pretty-print the current scenario and last step result."""
        if self._current_scenario is None:
            print("[ContentModerationEnv] No active scenario. Call reset() first.")
            return

        sc = self._current_scenario
        gt = sc["ground_truth"]
        st = sc["state"]

        sep = "-" * 62
        print(sep)
        print(f"  Scenario : {sc['id']}  |  Tier: {sc['tier'].upper()}")
        print(sep)
        print(f"  Text     : {textwrap.shorten(st['text'], 80)}")
        if st.get("audio_transcript"):
            print(f"  Audio    : {textwrap.shorten(st['audio_transcript'], 80)}")
        if st.get("visual_tags"):
            print(f"  Visual   : {', '.join(st['visual_tags'])}")
        print(f"  Prev flags: {st['previous_flags']}  |  Policy: {st['platform_policy']}")
        print()
        print(f"  Ground truth >  label={gt['label']}  action={gt['action']}", end="")
        if "severity" in gt:
            print(f"  severity={gt['severity']}", end="")
        print()
        if "rationale" in gt:
            print(f"  Rationale: {textwrap.shorten(gt['rationale'], 100)}")
        print()

        if self._done and self._last_breakdown:
            print(f"  Last reward : {self._last_reward:.3f}")
            print(f"  Breakdown   : {self._last_breakdown}")
        if self._episode_rewards:
            ep_total = sum(self._episode_rewards)
            print(f"  Episode rewards: {self._episode_rewards}  (total={ep_total:.3f})")
        print(sep)

    # -- Convenience ------------------------------------------------------------

    @property
    def scenario_ids(self) -> List[str]:
        """Sorted list of all available scenario IDs."""
        return list(self._scenario_ids)

    @property
    def num_scenarios(self) -> int:
        return len(self._scenarios)

    @property
    def episode_rewards(self) -> List[float]:
        """All rewards collected in the current episode so far."""
        return list(self._episode_rewards)

    def __repr__(self) -> str:
        active = self._current_scenario["id"] if self._current_scenario else "None"
        return (
            f"ContentModerationEnv("
            f"scenarios={self.num_scenarios}, "
            f"active={active!r}, "
            f"done={self._done}, "
            f"queue={self._queue_index}/{len(self._queue)})"
        )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    scenarios_path = os.path.join(base_dir, "moderation_benchmark.json")

    print("=" * 62)
    print("  ContentModerationEnv -- Demo")
    print("=" * 62)

    env = ContentModerationEnv(scenarios_path, seed=42)
    print(f"Loaded {env.num_scenarios} scenarios.\n")

    # -- Single-step mode (backward compat) ------------------------------------
    print("Episode 1 -- single-step (scen_easy_2, perfect)")
    state = env.reset(scenario_id="scen_easy_2")
    result = env.step({"label": "toxic", "action": "remove"})
    print(f"  Reward: {result['reward']}  done={result['done']}\n")

    # -- Multi-step queue mode ------------------------------------------------
    print("Episode 2 -- queue mode (3 posts, mixed tiers)")
    state = env.reset()
    step_n = 0
    while True:
        step_n += 1
        result = env.step({"label": "safe", "action": "allow", "severity": 1})
        print(f"  Step {step_n}: reward={result['reward']}  done={result['done']}")
        if result["done"]:
            break
        state = result["state"]
    print(f"  Episode rewards: {env.episode_rewards}\n")

    # -- Penalty test ----------------------------------------------------------
    print("Episode 3 -- penalty (escalating safe content)")
    env.reset(scenario_id="scen_easy_2")
    result = env.step({"label": "safe", "action": "escalate"})
    print(f"  Reward: {result['reward']}  (expect < 0)")
    print(f"  Breakdown: {result['info']['score_breakdown']}")


if __name__ == "__main__":
    main()
