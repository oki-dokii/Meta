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
check("reward range [-0.3, 1.0]",  reward.get("range") == [-0.3, 1.0],
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

# ── 3. Live environment API ───────────────────────────────────────────────────
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
check("reward can be negative", result["reward"] < 0, f"got {result['reward']}")

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

check("min reward ≥ -0.3", min(all_rewards) >= -0.3, f"min={min(all_rewards):.3f}")
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
