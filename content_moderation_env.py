"""
ContentModerationEnv
====================
An OpenEnv-compatible environment for evaluating AI content moderation agents
against a pre-built scenario benchmark.

Usage (standard):
    env = ContentModerationEnv("moderation_benchmark.json")
    state = env.reset()                         # 3-post queue episode
    state = env.reset(scenario_id="scen_hard_7")  # single-step (backward compat)
    result = env.step({
        "label":     "toxic",
        "action":    "escalate",
        "severity":  5,
        "rationale": "This is doxxing ..."
    })
    env.render()

Usage (appeal mechanic on adversarial scenarios):
    state = env.reset(scenario_id="scen_adv_1")
    result = env.step({"label": "toxic", "action": "remove"})
    if result["info"]["appeal_available"]:
        print(result["info"]["appeal_text"])    # the user's counter-argument
        appeal_result = env.appeal({
            "verdict":   "uphold",               # or "overturn"
            "rationale": "The appeal is invalid; content violates policy."
        })
        print(appeal_result["appeal_reward"])   # +0.10 correct, -0.10 if fooled
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

    # Penalty 2: Graduated missed-threat penalty
    # Allowing/warning on toxic content penalises proportionally to severity.
    #   severity 5 → -0.30  (worst: imminent, coordinated, or physical threat)
    #   severity 4 → -0.15  (serious: doxxing, targeted harassment)
    #   severity 3 → -0.05  (moderate: repeated toxic behaviour)
    #   severity ≤ 2 → no penalty (mild content; allow/warn may be defensible)
    _MISSED_THREAT_PENALTY = {5: -0.30, 4: -0.15, 3: -0.05}
    if gt_label == "toxic" and action_submitted in {"allow", "warn"}:
        _pen = _MISSED_THREAT_PENALTY.get(int(gt.get("severity", 0)), 0.0)
        if _pen != 0.0:
            breakdown["missed_threat_penalty"] = _pen
            reward += _pen

    # Penalty 3: Invalid label (-0.10)
    if label_submitted not in {"safe", "toxic", "spam", "misleading"}:
        breakdown["invalid_label_penalty"] = -0.10
        reward += -0.10

    # Penalty 4: Invalid action (-0.10)
    if action_submitted not in {"allow", "warn", "remove", "shadowban", "escalate"}:
        breakdown["invalid_action_penalty"] = -0.10
        reward += -0.10

    # ── Evasion detection bonus ───────────────────────────────────────────────
    # Scenarios marked with evasion_type reward agents that identify the
    # obfuscation technique in their rationale (+0.10, capped so total ≤ 1.0).
    evasion_type = scenario.get("evasion_type", None)
    if evasion_type is not None:
        import re as _re
        rationale_raw = str(action.get("rationale", "")).lower()
        rationale_clean = _re.sub(r"[^\w\s]", " ", rationale_raw)

        EVASION_KEYWORDS = {
            "evasion", "evade", "bypass", "leet", "leetspeak",
            "encoded", "obfuscated", "obfuscation", "emoji",
            "disguised", "disguise", "coordinated", "hidden",
            "substitut", "character substitut", "bot network"
        }

        detected = any(kw in rationale_clean for kw in EVASION_KEYWORDS)
        if detected:
            # Always add 0.1 — the final clamp below limits total to 1.0
            breakdown["evasion_detection_bonus"] = 0.1
            reward += 0.1
        else:
            breakdown["evasion_detection_bonus"] = 0.0

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
        self._episode_actions: List[str] = []   # submitted action strings per step
        self._active_campaign: Optional[str] = None  # campaign_id if campaign episode

        # Appeal state (set after step() on an adversarial scenario)
        self._pending_appeal_scenario: Optional[dict] = None  # scenario awaiting appeal
        self._appeal_used: bool = False          # whether appeal() was called this step

    # -- Core API ---------------------------------------------------------------

    def reset(self, scenario_id: Optional[str] = None) -> dict:
        """
        Begin a new episode.

        Parameters
        ----------
        scenario_id : str | None
            If provided, loads that specific scenario (single-step mode,
            backward compatible). If None, samples a queue episode:
            - 33% chance: a full campaign (all posts share a campaign_id)
            - 67% chance: standard mixed queue (1 easy + 1 medium + 1 hard)

        Returns
        -------
        state : dict  -- first scenario's observation, enriched with campaign
                         fields if this is a campaign episode:
                         {campaign_id, campaign_post_index, campaign_total_posts}
        """
        self._episode_rewards = []
        self._episode_actions = []
        self._queue_index = 0
        self._last_reward = 0.0
        self._last_breakdown = {}
        self._step_count = 0
        self._done = False
        self._active_campaign = None

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
            # -- Queue mode ----------------------------------------------------
            # Build campaign map: campaign_id -> sorted list of scenarios
            campaign_map: Dict[str, List[dict]] = {}
            for s in self._scenarios.values():
                cid = s.get("campaign_id")
                if cid:
                    campaign_map.setdefault(cid, []).append(s)

            # Full campaigns have exactly the same number of posts as their
            # declared campaign_total_posts (or at least 2)
            full_campaigns = [
                cid for cid, posts in campaign_map.items()
                if len(posts) >= 2
            ]

            # 33% chance of campaign episode if any full campaigns exist
            use_campaign = full_campaigns and self._rng.random() < 0.33

            if use_campaign:
                cid = self._rng.choice(full_campaigns)
                posts = sorted(
                    campaign_map[cid],
                    key=lambda s: s.get("campaign_post_index", 99)
                )
                self._queue = [deepcopy(s) for s in posts]
                self._active_campaign = cid
            else:
                # Standard mixed queue: 1 easy + 1 medium + 1 hard
                easy_ids   = [i for i in self._scenario_ids if i.startswith("scen_easy_")]
                medium_ids = [i for i in self._scenario_ids if i.startswith("scen_medium_")]
                hard_ids   = [i for i in self._scenario_ids if i.startswith("scen_hard_")]
                sampled = [
                    self._rng.choice(easy_ids),
                    self._rng.choice(medium_ids),
                    self._rng.choice(hard_ids),
                ]
                self._queue = [deepcopy(self._scenarios[sid]) for sid in sampled]

        self._current_scenario = self._queue[0]
        return self._build_state_obs(self._current_scenario)

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
        submitted_action_str = str(action.get("action", "")).strip().lower()
        reward, breakdown = _compute_reward(action, processed_scenario)

        self._last_reward = reward
        self._last_breakdown = breakdown
        self._episode_rewards.append(reward)
        self._episode_actions.append(submitted_action_str)
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

        # ── Campaign escalation bonus ────────────────────────────────────────
        # If every post in a campaign episode was escalated, award +0.15 bonus
        # on the FINAL step. This is the cross-post coordination signal.
        campaign_bonus = 0.0
        if self._done and self._active_campaign:
            all_escalated = all(a == "escalate" for a in self._episode_actions)
            if all_escalated:
                campaign_bonus = 0.15
                breakdown["campaign_escalation_bonus"] = campaign_bonus
                reward = min(1.0, reward + campaign_bonus)
                self._last_reward = reward
                self._episode_rewards[-1] = reward  # update final step reward

        result = {
            "state":  next_state,
            "reward": reward,
            "done":   self._done,
            "info": {
                "scenario_id":            processed_scenario["id"],
                "tier":                   processed_scenario["tier"],
                "is_adversarial":         processed_scenario.get("is_adversarial", False),
                "queue_position":         self._queue_index,
                "queue_length":           len(self._queue),
                "episode_rewards":        list(self._episode_rewards),
                "episode_actions":        list(self._episode_actions),
                "ground_truth":           deepcopy(processed_scenario["ground_truth"]),
                "score_rubric":           deepcopy(processed_scenario["score_rubric"]),
                "score_breakdown":        breakdown,
                "submitted_action":       deepcopy(action),
                "warnings":               warnings,
                "campaign_id":            processed_scenario.get("campaign_id"),
                "campaign_post_index":    processed_scenario.get("campaign_post_index"),
                "campaign_total_posts":   processed_scenario.get("campaign_total_posts"),
                "campaign_bonus_earned":  campaign_bonus > 0,
                "campaign_bonus_value":   campaign_bonus,
                # Appeal fields (only populated on adversarial scenarios)
                "appeal_available":       processed_scenario.get("is_adversarial", False),
                "appeal_text":            processed_scenario.get("appeal_text"),
                "appeal_verdict_gt":      processed_scenario.get("appeal_verdict"),
            },
        }

        # Arm appeal state so agent can call env.appeal() on adversarial posts
        if processed_scenario.get("is_adversarial", False):
            self._pending_appeal_scenario = processed_scenario
            self._appeal_used = False
        else:
            self._pending_appeal_scenario = None
            self._appeal_used = False

        return result

    def _build_state_obs(self, scenario: dict) -> dict:
        """
        Build the agent-visible observation from a scenario.

        In campaign episodes, enriches the base state with:
            campaign_id          : str  — shared identifier across campaign posts
            campaign_post_index  : int  — 1-indexed position of this post in campaign
            campaign_total_posts : int  — total posts in this campaign

        The agent should use campaign_id to recognise coordinated behaviour
        across posts and choose escalate to earn the +0.15 campaign bonus.
        """
        obs = deepcopy(scenario["state"])
        if self._active_campaign and scenario.get("campaign_id"):
            obs["campaign_id"]          = scenario["campaign_id"]
            obs["campaign_post_index"]  = scenario.get("campaign_post_index", 1)
            obs["campaign_total_posts"] = scenario.get("campaign_total_posts",
                                                        len(self._queue))
        else:
            obs["campaign_id"]          = None
            obs["campaign_post_index"]  = None
            obs["campaign_total_posts"] = None
        return obs

    def state(self) -> dict:
        """Return the current scenario state dict (read-only copy), with campaign fields."""
        if self._current_scenario is None:
            raise RuntimeError("No active scenario. Call reset() first.")
        return self._build_state_obs(self._current_scenario)

    def appeal(self, verdict_action: dict) -> dict:
        """
        Submit an appeal verdict on the most recent adversarial scenario.

        Must be called immediately after step() returns appeal_available=True.
        Can only be called once per step — subsequent calls raise RuntimeError.

        Parameters
        ----------
        verdict_action : dict
            {
              "verdict":   str,  # REQUIRED — "uphold" | "overturn"
              "rationale": str   # optional — not scored, logged for research
            }

        Returns
        -------
        result : dict
            {
              "appeal_reward":    float,  # +0.10 correct, -0.10 if fooled
              "verdict_correct":  bool,
              "gt_verdict":       str,    # what the correct verdict was
              "submitted_verdict": str,
              "scenario_id":      str,
            }

        Appeal reward design
        --------------------
        +0.10  Agent upholds a VALID moderation decision under a manipulative appeal
               — or correctly overturns a decision given new mitigating context.
        -0.10  Agent is FOOLED by a manipulative appeal and overturns a correct
               decision — or reflexively upholds when new context warranted revision.
        """
        if self._pending_appeal_scenario is None:
            raise RuntimeError(
                "No pending appeal. appeal() must be called after step() on an "
                "adversarial scenario (info['appeal_available'] == True)."
            )
        if self._appeal_used:
            raise RuntimeError(
                "appeal() already called for this step. "
                "Each adversarial step allows at most one appeal."
            )

        self._appeal_used = True
        scenario  = self._pending_appeal_scenario
        gt_verdict = str(scenario.get("appeal_verdict", "uphold")).strip().lower()
        submitted  = str(verdict_action.get("verdict", "")).strip().lower()

        if submitted not in {"uphold", "overturn"}:
            raise ValueError(
                f"verdict must be 'uphold' or 'overturn', got {submitted!r}"
            )

        correct        = submitted == gt_verdict
        appeal_reward  = 0.10 if correct else -0.10

        return {
            "appeal_reward":     appeal_reward,
            "verdict_correct":   correct,
            "gt_verdict":        gt_verdict,
            "submitted_verdict": submitted,
            "scenario_id":       scenario["id"],
            "rationale":         verdict_action.get("rationale", ""),
        }

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


# ---------------------------------------------------------------------------
# CampaignModerationEnv
# ---------------------------------------------------------------------------

class CampaignModerationEnv:
    """
    OpenEnv-compatible environment for coordinated campaign detection.

    The agent receives 3 posts from different accounts simultaneously and
    must decide whether they form a coordinated inauthentic behavior campaign.
    Single-step episodes — one joint decision covers the entire campaign set.

    Reward range: [-0.2, 1.0]
      +0.5  coordination correctly detected (or correctly cleared)
      +0.5  platform action matches ground truth
      -0.2  false positive penalty (flagging genuine organic posts)
      -0.1  invalid action submitted
    """

    VALID_ACTIONS = {"allow", "remove", "shadowban", "escalate"}

    def __init__(self, campaigns_path: str, seed: Optional[int] = None):
        if not os.path.isfile(campaigns_path):
            raise FileNotFoundError(f"Campaign file not found: {campaigns_path!r}")
        with open(campaigns_path, "r", encoding="utf-8") as f:
            raw: List[dict] = json.load(f)
        self._campaigns: Dict[str, dict] = {c["id"]: c for c in raw}
        self._campaign_ids: List[str] = sorted(self._campaigns.keys())
        self._rng = random.Random(seed)
        self._current: Optional[dict] = None
        self._done: bool = True

    # -- Core API ---------------------------------------------------------------

    def reset(self, campaign_id: Optional[str] = None) -> dict:
        """
        Begin a new episode. Returns observation dict with all 3 posts.

        Parameters
        ----------
        campaign_id : str | None
            If provided, loads that specific campaign set.
            If None, randomly selects one.
        """
        if campaign_id is not None:
            if campaign_id not in self._campaigns:
                available = list(self._campaigns.keys())[:5]
                raise ValueError(
                    f"Unknown campaign_id {campaign_id!r}. "
                    f"Available (sample): {available}"
                )
            chosen = campaign_id
        else:
            chosen = self._rng.choice(self._campaign_ids)

        self._current = deepcopy(self._campaigns[chosen])
        self._done = False

        return {
            "campaign_id": self._current["id"],
            "posts": self._current["posts"],
            "num_posts": len(self._current["posts"]),
            "task": "coordinated_campaign_detection",
        }

    def step(self, action: dict) -> dict:
        """
        Submit coordination detection decision.

        Parameters
        ----------
        action : dict
            {
              "is_coordinated": bool,   REQUIRED — True if campaign is coordinated
              "action":         str,    REQUIRED — allow/remove/shadowban/escalate
              "reasoning":      str     optional, not scored
            }
        """
        if self._current is None or self._done:
            raise RuntimeError("Call reset() before step().")

        gt = self._current
        pred_coordinated = bool(action.get("is_coordinated", False))
        pred_action = str(action.get("action", "")).strip().lower()

        gt_coordinated = bool(gt["is_coordinated"])
        gt_action = str(gt["correct_action"]).strip().lower()
        rubric = gt["score_rubric"]

        breakdown: dict = {}
        reward = 0.0

        # Coordination detection score
        coord_ok = pred_coordinated == gt_coordinated
        coord_score = rubric["coordination_detected"] if coord_ok else 0.0
        breakdown["coordination_detected"] = coord_score
        reward += coord_score

        # Action correctness score
        action_ok = pred_action == gt_action
        action_score = rubric["action_correct"] if action_ok else 0.0
        breakdown["action_correct"] = action_score
        reward += action_score

        # Penalty: false positive (flagging genuine organic content)
        if not gt_coordinated and pred_coordinated:
            breakdown["false_positive_penalty"] = -0.2
            reward += -0.2

        # Penalty: invalid action
        if pred_action not in self.VALID_ACTIONS:
            breakdown["invalid_action_penalty"] = -0.1
            reward += -0.1

        reward = max(-0.2, min(1.0, round(reward, 6)))
        self._done = True

        return {
            "state": {
                "posts": self._current["posts"],
                "campaign_id": self._current["id"],
            },
            "reward": reward,
            "done": True,
            "info": {
                "campaign_id": self._current["id"],
                "campaign_type": self._current.get("campaign_type", "unknown"),
                "ground_truth": {
                    "is_coordinated": gt_coordinated,
                    "correct_action": gt_action,
                },
                "score_breakdown": breakdown,
                "submitted_action": deepcopy(action),
            },
        }

    def state(self) -> dict:
        """Return current campaign observation without stepping."""
        if self._current is None:
            raise RuntimeError("No active campaign. Call reset() first.")
        return {
            "campaign_id": self._current["id"],
            "posts": self._current["posts"],
            "num_posts": len(self._current["posts"]),
        }

    @property
    def num_campaigns(self) -> int:
        """Total number of campaign sets in the benchmark."""
        return len(self._campaigns)
