"""
validate_openenv.py
===================
Self-contained validator for openenv.yaml and content_moderation_env.py.
Checks all fields required by the OpenEnv spec and confirms the live
environment behaves correctly. Prints PASS/FAIL per check.

Run:
    python3 validate_openenv.py
"""

import json
import sys
from pathlib import Path

import yaml  # pip install pyyaml

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
from content_moderation_env import ContentModerationEnv

YAML_PATH = SCRIPT_DIR / "openenv.yaml"
JSON_PATH = SCRIPT_DIR / "moderation_benchmark.json"

PASS = "✅ PASS"
FAIL = "❌ FAIL"
WARN = "⚠️  WARN"

checks_passed = 0
checks_failed = 0


def check(name: str, condition: bool, detail: str = "", warn: bool = False) -> bool:
    global checks_passed, checks_failed
    status = PASS if condition else (WARN if warn else FAIL)
    suffix = f"  ({detail})" if detail else ""
    print(f"  {status}  {name}{suffix}")
    if condition:
        checks_passed += 1
    else:
        checks_failed += 1
    return condition


# ── 1. YAML structure ─────────────────────────────────────────────────────────
print("\n── openenv.yaml structure ──────────────────────────────────────────")
with open(YAML_PATH) as f:
    spec = yaml.safe_load(f)

check("name field present",       "name" in spec)
check("version field present",    "version" in spec)
check("description field present","description" in spec)
check("tasks field present",      "tasks" in spec)
check("observation_space present","observation_space" in spec)
check("action_space present",     "action_space" in spec)
check("reward field present",     "reward" in spec)
check("api field present",        "api" in spec)
check("baseline field present",   "baseline" in spec)
check("deployment field present", "deployment" in spec)

tasks = spec.get("tasks", [])
check("at least 3 tasks defined", len(tasks) >= 3, f"found {len(tasks)}")

task_names = [t.get("name") for t in tasks]
for name in ["Easy Content Moderation", "Medium Content Moderation", "Hard Content Moderation"]:
    check(f"task '{name}' present", name in task_names)

difficulties = [t.get("difficulty") for t in tasks]
check("easy difficulty present",   "easy"   in difficulties)
check("medium difficulty present", "medium" in difficulties)
check("hard difficulty present",   "hard"   in difficulties)

reward = spec.get("reward", {})
check("reward range [0.0, 1.0]",  reward.get("range") == [0.0, 1.0],
      f"got {reward.get('range')}")
check("partial_progress = true",   reward.get("partial_progress") is True)

api = spec.get("api", {})
check("reset() documented",  "reset" in api)
check("step() documented",   "step"  in api)
check("state() documented",  "state" in api)

# ── 2. Dataset integrity ──────────────────────────────────────────────────────
print("\n── moderation_benchmark.json integrity ─────────────────────────────")
data = json.loads(JSON_PATH.read_text())
check("≥ 60 scenarios", len(data) >= 60, f"found {len(data)}")
check("≥ 75 scenarios", len(data) >= 75, f"found {len(data)}")

tiers = {"easy": 0, "medium": 0, "hard": 0}
ids_seen = set()
all_ok = True
for s in data:
    if s["id"] in ids_seen:
        all_ok = False
    ids_seen.add(s["id"])
    tiers[s.get("tier", "?")] = tiers.get(s.get("tier", "?"), 0) + 1
check("no duplicate IDs", all_ok)
check("easy tier count ≥ 20",   tiers["easy"]   >= 20, f"found {tiers['easy']}")
check("medium tier count ≥ 20", tiers["medium"] >= 20, f"found {tiers['medium']}")
check("hard tier count ≥ 20",   tiers["hard"]   >= 20, f"found {tiers['hard']}")

# Check all hard scenarios have severity in ground_truth
hard_with_sev = sum(1 for s in data if s["tier"]=="hard" and "severity" in s.get("ground_truth",{}))
hard_total = tiers["hard"]
check("hard scenarios have severity", hard_with_sev == hard_total,
      f"{hard_with_sev}/{hard_total}")

# Easy-tier GT coverage: all labels + all actions must be represented,
# and the 4 previously missing combos must each have ≥ 2 examples.
from collections import Counter as _C
easy_s = [s for s in data if s["tier"] == "easy"]
e_labels  = _C(s["ground_truth"]["label"]  for s in easy_s)
e_actions = _C(s["ground_truth"]["action"] for s in easy_s)
e_combos  = _C((s["ground_truth"]["label"], s["ground_truth"]["action"]) for s in easy_s)
for lbl in ["safe", "toxic", "spam", "misleading"]:
    check(f"easy label '{lbl}' covered", e_labels[lbl] >= 2, f"count={e_labels[lbl]}")
for act in ["allow", "warn", "remove", "shadowban", "escalate"]:
    check(f"easy action '{act}' covered", e_actions[act] >= 2, f"count={e_actions[act]}")
for lbl, act in [("misleading","shadowban"),("toxic","shadowban"),
                 ("toxic","warn"),("misleading","escalate")]:
    check(f"easy {lbl}/{act} ≥ 2 examples", e_combos[(lbl,act)] >= 2,
          f"count={e_combos[(lbl,act)]}")

print("\n── ContentModerationEnv live API ───────────────────────────────────")
env = ContentModerationEnv(str(JSON_PATH), seed=42)

check("env loads all scenarios", env.num_scenarios == len(data),
      f"{env.num_scenarios} loaded, {len(data)} in JSON")
check("scenario_ids property works", len(env.scenario_ids) == env.num_scenarios)

# Single-step mode
state = env.reset(scenario_id="scen_easy_2")
check("reset(scenario_id) returns dict", isinstance(state, dict))
check("state has 'text' field", "text" in state)
check("state has 'platform_policy' field", "platform_policy" in state)
check("state has 'previous_flags' field", "previous_flags" in state)

# Perfect action
result = env.step({"label": "toxic", "action": "remove"})
check("step() returns dict with 4 keys",
      all(k in result for k in ["state","reward","done","info"]))
check("single-step done=True", result["done"] is True)
check("perfect reward = 1.0", result["reward"] == 1.0, f"got {result['reward']}")

# Queue mode
state = env.reset()
check("queue reset() returns state", isinstance(state, dict))
step_n = 0
rewards = []
while True:
    r = env.step({"label": "safe", "action": "allow", "severity": 1})
    rewards.append(r["reward"])
    step_n += 1
    if r["done"]:
        break
check("queue mode runs 3 steps", step_n == 3, f"ran {step_n}")
check("episode_rewards accumulates", len(env.episode_rewards) == 3)

# Penalty: false positive escalation — submit WRONG label so no +0.5 offset
env.reset(scenario_id="scen_easy_1")   # GT: safe/allow
result = env.step({"label": "toxic", "action": "escalate"})   # wrong label, FP penalty
check("false_positive_penalty fires", "false_positive_penalty" in result["info"]["score_breakdown"])
check("reward drops to zero from penalty", result["reward"] == 0.0, f"got {result['reward']}")

# Guard: step on done env
try:
    env.step({"label": "safe", "action": "allow"})
    check("step() on done env raises RuntimeError", False)
except RuntimeError:
    check("step() on done env raises RuntimeError", True)

# state() method
env.reset(scenario_id="scen_hard_1")
s = env.state()
check("state() returns dict", isinstance(s, dict))

# ── valid_actions: ambiguous scenario scoring ─────────────────────────────────
from content_moderation_env import _compute_reward as _cr

# Find a scenario with valid_actions: [remove, shadowban]
rs_scenario = next(
    (sc for sc in env._scenarios.values()
     if sc.get("ground_truth", {}).get("valid_actions") == ["remove", "shadowban"]
     or sc.get("ground_truth", {}).get("valid_actions") == ["shadowban", "remove"]),
    None
)
if rs_scenario:
    gt_label = rs_scenario["ground_truth"]["label"]
    gt_sev   = rs_scenario["ground_truth"].get("severity", 3)
    r_rem,  _ = _cr({"label": gt_label, "action": "remove",    "severity": gt_sev}, rs_scenario)
    r_sha,  _ = _cr({"label": gt_label, "action": "shadowban", "severity": gt_sev}, rs_scenario)
    r_bad,  _ = _cr({"label": gt_label, "action": "allow",     "severity": gt_sev}, rs_scenario)
    check("valid_actions: remove scores full credit",    r_rem  >= 0.8, f"got {r_rem:.2f}")
    check("valid_actions: shadowban scores full credit", r_sha  >= 0.8, f"got {r_sha:.2f}")
    check("valid_actions: remove == shadowban reward",   abs(r_rem - r_sha) < 0.01,
          f"remove={r_rem:.2f} shadowban={r_sha:.2f}")
    check("valid_actions: allow does NOT score full",    r_bad  < r_rem, f"allow={r_bad:.2f}")
else:
    check("valid_actions: remove/shadowban scenario exists", False,
          "none found — run _add_ambiguous_scenarios.py")

ambig_count = sum(1 for sc in env._scenarios.values() if "valid_actions" in sc.get("ground_truth", {}))
check("ambiguous scenarios (valid_actions) ≥ 10", ambig_count >= 10, f"found {ambig_count}")

# ── Campaign mechanic ──────────────────────────────────────────────────────────
print("\n── Campaign mechanic (cross-post coordination) ──────────────────────")

# Count campaigns
camp_map: dict = {}
for sc in env._scenarios.values():
    cid = sc.get("campaign_id")
    if cid:
        camp_map.setdefault(cid, []).append(sc)
full_camps = {k: v for k, v in camp_map.items() if len(v) >= 2}
check("campaigns ≥ 3 defined", len(full_camps) >= 3, f"found {len(full_camps)}: {list(full_camps)[:3]}")

# Force a campaign episode using the first known full campaign
first_camp_id = sorted(full_camps.keys())[0]
first_camp_posts = sorted(full_camps[first_camp_id], key=lambda s: s.get("campaign_post_index", 99))

# Manually build env into campaign mode to test deterministically
camp_env = ContentModerationEnv(str(JSON_PATH), seed=99)
camp_env._queue = [__import__("copy").deepcopy(s) for s in first_camp_posts]
camp_env._active_campaign = first_camp_id
camp_env._current_scenario = camp_env._queue[0]
camp_env._done = False
camp_env._queue_index = 0
camp_env._episode_rewards = []
camp_env._episode_actions = []

# Check state has campaign fields
obs = camp_env._build_state_obs(camp_env._queue[0])
check("campaign_id present in state obs",         obs.get("campaign_id") == first_camp_id)
check("campaign_post_index present in state obs", obs.get("campaign_post_index") == 1)
check("campaign_total_posts present in state obs",obs.get("campaign_total_posts") is not None)

# Run campaign episode: escalate all → bonus should fire
step_rewards = []
for i, post in enumerate(first_camp_posts):
    camp_env._current_scenario = __import__("copy").deepcopy(post)
    camp_env._done = False
    # Get the GT label so we score it correctly (max reward) + escalate
    gt_label = post["ground_truth"]["label"]
    gt_sev   = post["ground_truth"].get("severity", 3)
    action   = {"label": gt_label, "action": "escalate", "severity": gt_sev}
    result   = camp_env.step(action)
    step_rewards.append(result["reward"])

bonus_fired = result["info"]["campaign_bonus_earned"]
bonus_val   = result["info"]["campaign_bonus_value"]
check("campaign bonus fires when all escalated",  bonus_fired is True)
check("campaign bonus value = 0.15",              abs(bonus_val - 0.15) < 0.001, f"got {bonus_val}")
check("final step reward ≥ baseline (bonus added)", step_rewards[-1] > 0.8)

# Now test: NOT escalating all = NO bonus
camp_env2 = ContentModerationEnv(str(JSON_PATH), seed=99)
camp_env2._queue = [__import__("copy").deepcopy(s) for s in first_camp_posts]
camp_env2._active_campaign = first_camp_id
camp_env2._current_scenario = camp_env2._queue[0]
camp_env2._done = False
camp_env2._queue_index = 0
camp_env2._episode_rewards = []
camp_env2._episode_actions = []

for i, post in enumerate(first_camp_posts):
    camp_env2._current_scenario = __import__("copy").deepcopy(post)
    camp_env2._done = False
    gt_label = post["ground_truth"]["label"]
    # First post: warn (not escalate) — should break bonus
    action = {"label": gt_label, "action": ("warn" if i == 0 else "escalate")}
    r2 = camp_env2.step(action)

check("campaign bonus does NOT fire if any action ≠ escalate",
      r2["info"]["campaign_bonus_earned"] is False)

# Check episode_actions tracked correctly
check("episode_actions tracked in info", "episode_actions" in result["info"])

# ── 4. Reward bounds ─────────────────────────────────────────────────────────

# ── is_adversarial in state obs ───────────────────────────────────────────────
adv_obs_env = ContentModerationEnv(str(JSON_PATH), seed=99)
obs_adv    = adv_obs_env.reset(scenario_id="scen_adv_1")
obs_normal = adv_obs_env.reset(scenario_id="scen_easy_1")
check("is_adversarial=True in state obs on adv scenario",
      obs_adv.get("is_adversarial") is True)
check("is_adversarial=False in state obs on normal scenario",
      obs_normal.get("is_adversarial") is False)

# ── reset(campaign_id=...) deterministic campaign mode ─────────────────────────
print("\n── reset(campaign_id) deterministic mode ────────────────────────────")
camp_reset_env = ContentModerationEnv(str(JSON_PATH), seed=11)
obs_c = camp_reset_env.reset(campaign_id="camp_crypto_001")
check("reset(campaign_id) returns state", isinstance(obs_c, dict))
check("reset(campaign_id) sets active_campaign",
      camp_reset_env._active_campaign == "camp_crypto_001")
check("reset(campaign_id) queues all 3 posts",
      len(camp_reset_env._queue) == 3)
check("reset(campaign_id) orders by campaign_post_index",
      [s.get("campaign_post_index") for s in camp_reset_env._queue] == [1, 2, 3])
try:
    camp_reset_env.reset(campaign_id="nonexistent_xyz")
    check("reset(bad campaign_id) raises ValueError", False)
except ValueError:
    check("reset(bad campaign_id) raises ValueError", True)
try:
    camp_reset_env.reset(scenario_id="scen_easy_1", campaign_id="camp_crypto_001")
    check("reset(scenario_id+campaign_id) raises ValueError", False)
except ValueError:
    check("reset(scenario_id+campaign_id) raises ValueError", True)

# ── Appeal mechanic (adversarial scenarios) ───────────────────────────────────
print("\n── Appeal mechanic (adversarial scenarios) ──────────────────────────")

adv_scenarios = [s for s in data if s.get("is_adversarial")]
check("adversarial scenarios ≥ 10 defined", len(adv_scenarios) >= 10, f"found {len(adv_scenarios)}")
uphold_count   = sum(1 for s in adv_scenarios if s.get("appeal_verdict") == "uphold")
overturn_count = sum(1 for s in adv_scenarios if s.get("appeal_verdict") == "overturn")
check("uphold verdicts ≥ 4",   uphold_count   >= 4, f"found {uphold_count}")
check("overturn verdicts ≥ 2", overturn_count >= 2, f"found {overturn_count}")

# Pick first uphold and overturn scenario for live tests
uphold_sc   = next(s for s in adv_scenarios if s["appeal_verdict"] == "uphold")
overturn_sc = next(s for s in adv_scenarios if s["appeal_verdict"] == "overturn")

adv_env = ContentModerationEnv(str(JSON_PATH), seed=77)

# -- Test 1: step() on adversarial scenario exposes appeal fields
adv_env.reset(scenario_id=uphold_sc["id"])
gt = uphold_sc["ground_truth"]
r = adv_env.step({"label": gt["label"], "action": gt["action"],
                  "severity": gt.get("severity", 3)})
check("is_adversarial present in info",  r["info"]["is_adversarial"] is True)
check("appeal_available=True on adv",    r["info"]["appeal_available"] is True)
check("appeal_text present",             isinstance(r["info"]["appeal_text"], str)
                                         and len(r["info"]["appeal_text"]) > 10)

# -- Test 2: correct uphold verdict → +0.10
ar = adv_env.appeal({"verdict": "uphold", "rationale": "appeal is manipulative"})
check("correct uphold verdict → +0.10",  abs(ar["appeal_reward"] - 0.10) < 0.001,
      f"got {ar['appeal_reward']}")
check("verdict_correct=True on uphold",  ar["verdict_correct"] is True)

# -- Test 3: double-call guard
try:
    adv_env.appeal({"verdict": "uphold"})
    check("second appeal() raises RuntimeError", False)
except RuntimeError:
    check("second appeal() raises RuntimeError", True)

# -- Test 4: wrong verdict → -0.10
adv_env.reset(scenario_id=uphold_sc["id"])
adv_env.step({"label": gt["label"], "action": gt["action"], "severity": gt.get("severity", 3)})
ar_wrong = adv_env.appeal({"verdict": "overturn"})   # wrong — should uphold
check("wrong verdict → -0.10", abs(ar_wrong["appeal_reward"] - (-0.10)) < 0.001,
      f"got {ar_wrong['appeal_reward']}")
check("verdict_correct=False on wrong", ar_wrong["verdict_correct"] is False)

# -- Test 5: overturn case
adv_env.reset(scenario_id=overturn_sc["id"])
gt2 = overturn_sc["ground_truth"]
adv_env.step({"label": gt2["label"], "action": gt2["action"], "severity": gt2.get("severity", 3)})
ar_ov = adv_env.appeal({"verdict": "overturn"})
check("correct overturn verdict → +0.10", abs(ar_ov["appeal_reward"] - 0.10) < 0.001,
      f"got {ar_ov['appeal_reward']}")

# -- Test 6: appeal() on non-adversarial scenario raises
adv_env.reset(scenario_id="scen_easy_1")
adv_env.step({"label": "safe", "action": "allow"})
try:
    adv_env.appeal({"verdict": "uphold"})
    check("appeal() on non-adversarial raises RuntimeError", False)
except RuntimeError:
    check("appeal() on non-adversarial raises RuntimeError", True)

# Build a synthetic toxic scenario at each severity level to verify penalties
def _make_toxic(severity: int) -> dict:
    return {
        "id": f"_synth_sev{severity}", "tier": "hard",
        "state": {"text": "x", "audio_transcript": None, "visual_tags": [],
                  "previous_flags": 0, "platform_policy": "strict"},
        "ground_truth": {"label": "toxic", "action": "escalate", "severity": severity},
        "score_rubric": {"label_correct": 0.4, "action_correct": 0.4, "severity_within_1": 0.2},
    }

from content_moderation_env import _compute_reward as _cr2
# allow on sev-5 → -0.30
r5, b5 = _cr2({"label": "toxic", "action": "allow", "severity": 5}, _make_toxic(5))
check("sev-5 miss penalty = -0.30",
      b5.get("missed_threat_penalty") == -0.30, f"got {b5.get('missed_threat_penalty')}")
# allow on sev-4 → -0.15
r4, b4 = _cr2({"label": "toxic", "action": "allow", "severity": 4}, _make_toxic(4))
check("sev-4 miss penalty = -0.15",
      b4.get("missed_threat_penalty") == -0.15, f"got {b4.get('missed_threat_penalty')}")
# allow on sev-3 → -0.05
r3, b3 = _cr2({"label": "toxic", "action": "allow", "severity": 3}, _make_toxic(3))
check("sev-3 miss penalty = -0.05",
      b3.get("missed_threat_penalty") == -0.05, f"got {b3.get('missed_threat_penalty')}")
# allow on sev-2 → no penalty
r2, b2 = _cr2({"label": "toxic", "action": "allow", "severity": 2}, _make_toxic(2))
check("sev-2 miss → no penalty",
      "missed_threat_penalty" not in b2, f"got {b2.get('missed_threat_penalty')}")
# old key name gone
check("old 'missed_critical_threat_penalty' key replaced",
      "missed_critical_threat_penalty" not in b5)

print("\n── Reward range validation ─────────────────────────────────────────")
from content_moderation_env import _compute_reward
import json as _json

all_rewards = []
for scenario in data:
    # Worst possible action
    r, _ = _compute_reward({"label": "safe", "action": "escalate", "severity": 1}, scenario)
    all_rewards.append(r)
    # Best possible action
    gt = scenario["ground_truth"]
    r2, _ = _compute_reward({
        "label": gt["label"],
        "action": gt["action"],
        "severity": gt.get("severity", 3),
    }, scenario)
    all_rewards.append(r2)

check("min reward ≥ 0.0", min(all_rewards) >= 0.0, f"min={min(all_rewards):.3f}")
check("max reward ≤ 1.0",  max(all_rewards) <= 1.0,  f"max={max(all_rewards):.3f}")

# ── Summary ───────────────────────────────────────────────────────────────────
total = checks_passed + checks_failed
print(f"\n{'═'*62}")
print(f"  RESULT: {checks_passed}/{total} checks passed")
if checks_failed == 0:
    print("  ✅  ALL CHECKS PASSED — openenv.yaml is valid")
else:
    print(f"  ❌  {checks_failed} check(s) FAILED — fix before submission")
print(f"{'═'*62}\n")
sys.exit(0 if checks_failed == 0 else 1)
