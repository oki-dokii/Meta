"""
benchmark_pipeline_gemini.py
============================
3-step content moderation benchmark — Gemini-only version.

Uses Google Gemini for all 3 steps (no Anthropic key required):

Step 1 — Gemini 2.0 Flash   : live agent across all 60 scenarios
Step 2 — Gemini 2.0 Flash   : adjudicates low-reward cases (reward < 0.3)
Step 3 — Gemini 1.5 Pro     : generates markdown evaluation report

Requirements:
    pip install google-genai

Environment variables:
    GOOGLE_API_KEY
"""

import json
import os
import re
import sys
import time
from pathlib import Path

try:
    from google import genai as google_genai
except ImportError:
    sys.exit("Missing dependency: pip install google-genai")

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
from content_moderation_env import ContentModerationEnv

SCENARIOS_PATH = SCRIPT_DIR / "moderation_benchmark.json"
RESULTS_PATH   = SCRIPT_DIR / "benchmark_results_gemini.json"
REPORT_PATH    = SCRIPT_DIR / "benchmark_report_gemini.md"

LOW_REWARD_THRESHOLD = 0.3
RATE_LIMIT_DELAY     = 1.0   # seconds between calls


# ── Utilities ─────────────────────────────────────────────────────────────────

def _extract_json(text: str) -> dict:
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON found:\n{text[:300]}")
    blob = re.sub(r",\s*([}\]])", r"\1", match.group(0))
    return json.loads(blob)


def _retry(fn, retries: int = 3, backoff: float = 2.0):
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            if attempt == retries - 1:
                raise
            wait = backoff ** (attempt + 1)
            print(f"    ⚠ Attempt {attempt+1} failed: {e}. Retrying in {wait:.0f}s …")
            time.sleep(wait)


# ── Step 1 — Gemini Agent ─────────────────────────────────────────────────────

AGENT_PROMPT = """\
You are a content moderation AI. Review the content state below and make a decision.

POLICY GUIDE:
- strict  : zero tolerance; prefer removal/escalation over warnings.
- moderate: graduated response; context and repeat offence matter.
- lenient : allow edge cases; only act on clear violations.

A repeat offender (previous_flags > 2) on a strict platform should receive
escalated action even for borderline content.

STATE:
{state_json}

OUTPUT — respond ONLY with valid JSON, no preamble, no markdown fences:
{{
  "label": "safe|toxic|spam|misleading",
  "action": "allow|warn|remove|shadowban|escalate",
  "severity": 1,
  "rationale": "1-2 sentence explanation"
}}"""


def run_agent_gemini(client, state: dict) -> dict:
    prompt = AGENT_PROMPT.format(state_json=json.dumps(state, indent=2))

    def _call():
        resp = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        return _extract_json(resp.text)

    return _retry(_call)


def run_full_benchmark(env: ContentModerationEnv, client) -> list[dict]:
    results = []
    ids = env.scenario_ids
    total = len(ids)

    print(f"\n{'─'*62}")
    print(f"  STEP 1 — Gemini 2.0 Flash agent  ({total} scenarios)")
    print(f"{'─'*62}")

    for i, sid in enumerate(ids, 1):
        state = env.reset(sid)
        tier  = env._current_scenario["tier"]

        try:
            action = run_agent_gemini(client, state)
        except Exception as e:
            print(f"  [{i:02}/{total}] {sid}: ✗ agent error — {e}")
            action = {"label": "safe", "action": "allow", "severity": 1,
                      "rationale": f"[ERROR] {e}"}

        result = env.step(action)

        row = {
            "scenario_id":     sid,
            "tier":            tier,
            "agent_decision":  action,
            "reward":          result["reward"],
            "breakdown":       result["info"]["score_breakdown"],
            "ground_truth":    result["info"]["ground_truth"],
            "state_text":      state["text"],
            "platform_policy": state["platform_policy"],
            "previous_flags":  state["previous_flags"],
        }
        results.append(row)

        status = "✓" if row["reward"] >= 0.8 else ("~" if row["reward"] >= 0.4 else "✗")
        print(f"  [{i:02}/{total}] {sid}: {status} "
              f"reward={row['reward']:.2f}  "
              f"label={action.get('label','?')} / action={action.get('action','?')}")

        time.sleep(RATE_LIMIT_DELAY)

    return results


# ── Step 2 — Gemini Adjudicator ───────────────────────────────────────────────

ADJUDICATOR_TMPL = """\
You are a senior trust & safety policy expert reviewing a content moderation decision.

The automated scorer flagged this case as LOW REWARD ({reward:.2f}/1.0).
Determine if the decision was reasonable given the platform policy.

CONTENT STATE:
{state_json}

AI DECISION:
{agent_decision_json}

GROUND TRUTH:
{ground_truth_json}

Score on these dimensions (0.0-1.0 each):
1. policy_alignment: Does the action match the platform_policy level?
2. proportionality: Is the severity appropriate?
3. consistency: Would a human expert likely agree?

Output ONLY valid JSON:
{{
  "override_reward": 0.0,
  "override_rationale": "1 sentence explanation",
  "human_agreement": 0.0,
  "dimensions": {{
    "policy_alignment": 0.0,
    "proportionality": 0.0,
    "consistency": 0.0
  }}
}}"""


def adjudicate_low_reward(low_reward_rows: list[dict], client) -> list[dict]:
    adjudicated = []
    total = len(low_reward_rows)

    print(f"\n{'─'*62}")
    print(f"  STEP 2 — Gemini 2.0 Flash adjudicator  ({total} low-reward cases)")
    print(f"{'─'*62}")

    for i, row in enumerate(low_reward_rows, 1):
        state_for_prompt = {
            "id": row["scenario_id"], "tier": row["tier"],
            "text": row["state_text"],
            "platform_policy": row["platform_policy"],
            "previous_flags": row["previous_flags"],
        }
        prompt = ADJUDICATOR_TMPL.format(
            reward=row["reward"],
            state_json=json.dumps(state_for_prompt, indent=2),
            agent_decision_json=json.dumps(row["agent_decision"], indent=2),
            ground_truth_json=json.dumps(row["ground_truth"], indent=2),
        )

        try:
            def _call(p=prompt):
                resp = client.models.generate_content(model="gemini-2.0-flash", contents=p)
                return _extract_json(resp.text)

            verdict = _retry(_call)
            override_reward = float(verdict.get("override_reward", 0.0))
            verdict["override_reward"] = max(0.0, min(1.0, override_reward))
            verdict.setdefault("override_rationale", "")
            verdict.setdefault("human_agreement", 0.0)
            verdict.setdefault("dimensions", {})

        except Exception as e:
            print(f"  [{i:02}/{total}] {row['scenario_id']}: ✗ adjudicator error — {e}")
            verdict = {
                "override_reward": row["reward"],
                "override_rationale": f"[ERROR] {e}",
                "human_agreement": 0.0,
                "dimensions": {},
            }

        row["adjudication"] = verdict
        adjudicated.append(row)

        orr = verdict["override_reward"]
        ha  = verdict.get("human_agreement", 0.0)
        dims = verdict.get("dimensions", {})
        dims_str = "  ".join(f"{k[:3]}={v:.1f}" for k, v in dims.items()) if dims else "n/a"
        print(f"  [{i:02}/{total}] {row['scenario_id']}: "
              f"raw={row['reward']:.2f} → override={orr:.2f}  agree={ha:.2f}  [{dims_str}]")

        time.sleep(RATE_LIMIT_DELAY)

    return adjudicated


# ── Step 3 — Gemini Report Generator ─────────────────────────────────────────

REPORT_PROMPT = """\
You are an expert AI evaluator. Write a comprehensive benchmark evaluation report in markdown.

An AI agent (Gemini 2.0 Flash) was evaluated across 60 content moderation scenarios
(easy / medium / hard tier). Low-reward cases (< 0.3) were adjudicated by a second
Gemini instance using three policy dimensions: policy_alignment, proportionality,
and consistency — each scored 0.0-1.0.

RESULTS_JSON:
{results_json}

Write a full evaluation report covering:
1. Executive Summary (key metrics, headline finding, raw + override-adjusted mean rewards)
2. Per-tier performance table (mean reward, % perfect, % zero)
3. Label & action accuracy breakdown
4. Analysis of hard-tier severity scoring
5. Low-reward case analysis with dimension scores table
6. Top-3 most interesting errors (quote scenario text, GT vs agent decision, override score)
7. Recommendations for agent improvement
8. Appendix: full score table (scenario_id | tier | raw_reward | override_reward | label✓ | action✓)

Write clear, structured markdown. Be precise with statistics. 3-5 pages."""


def generate_report(results: list[dict], client) -> str:
    print(f"\n{'─'*62}")
    print("  STEP 3 — Gemini 1.5 Pro report generation")
    print(f"{'─'*62}")

    slim = []
    for r in results:
        row = {k: r[k] for k in (
            "scenario_id", "tier", "reward", "breakdown",
            "ground_truth", "agent_decision", "state_text",
            "platform_policy", "previous_flags"
        ) if k in r}
        if "adjudication" in r:
            adj = r["adjudication"]
            row["adjudication"]      = adj
            row["override_reward"]   = adj.get("override_reward", r["reward"])
            row["override_rationale"] = adj.get("override_rationale", "")
            row["human_agreement"]   = adj.get("human_agreement", 0.0)
            row["dimensions"]        = adj.get("dimensions", {})
        slim.append(row)

    def _call():
        resp = client.models.generate_content(
            model="gemini-1.5-pro",
            contents=REPORT_PROMPT.format(results_json=json.dumps(slim, indent=2)),
        )
        return resp.text

    report = _retry(_call)
    print("  Report generated successfully.\n")
    return report


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    google_key = os.environ.get("GOOGLE_API_KEY")
    if not google_key:
        sys.exit("Set GOOGLE_API_KEY environment variable.")

    client = google_genai.Client(api_key=google_key)
    env = ContentModerationEnv(str(SCENARIOS_PATH), seed=0)
    print(f"Loaded {env.num_scenarios} scenarios from {SCENARIOS_PATH.name}")

    # STEP 1
    all_results = run_full_benchmark(env, client)
    RESULTS_PATH.write_text(json.dumps(all_results, indent=2))

    rewards = [r["reward"] for r in all_results]
    mean_r  = sum(rewards) / len(rewards)
    perfect = sum(1 for r in rewards if r == 1.0)
    zero    = sum(1 for r in rewards if r == 0.0)
    print(f"\n  Mean reward   : {mean_r:.3f}")
    print(f"  Perfect (1.0) : {perfect}/{len(rewards)}")
    print(f"  Zero    (0.0) : {zero}/{len(rewards)}")
    print(f"  ✓ Raw results saved → {RESULTS_PATH.name}")

    # STEP 2
    low_reward = [r for r in all_results if r["reward"] < LOW_REWARD_THRESHOLD]
    print(f"\n  {len(low_reward)} scenarios below threshold ({LOW_REWARD_THRESHOLD})")

    if low_reward:
        adjudicated = adjudicate_low_reward(low_reward, client)
        adj_map = {r["scenario_id"]: r for r in adjudicated}
        for row in all_results:
            if row["scenario_id"] in adj_map:
                adj = adj_map[row["scenario_id"]]["adjudication"]
                row["adjudication"]    = adj
                row["override_reward"] = adj.get("override_reward", row["reward"])

        mean_override = sum(
            r.get("adjudication", {}).get("override_reward", r["reward"])
            for r in adjudicated
        ) / len(adjudicated)
        defensible = sum(
            1 for r in adjudicated
            if r.get("adjudication", {}).get("override_reward", 0.0) >= 0.5
        )
        print(f"\n  Defensible override ≥ 0.5 : {defensible}")
        print(f"  Mean override reward       : {mean_override:.3f}")

        RESULTS_PATH.write_text(json.dumps(all_results, indent=2))
        print(f"  ✓ Updated results saved → {RESULTS_PATH.name}")
    else:
        print("  No low-reward cases — all scores ≥ 0.3 ✓")

    # STEP 3
    report_md = generate_report(all_results, client)
    REPORT_PATH.write_text(report_md, encoding="utf-8")
    print(f"  ✓ Report saved → {REPORT_PATH.name}")

    adj_rewards = [r.get("override_reward", r["reward"]) for r in all_results]
    mean_adj = sum(adj_rewards) / len(adj_rewards)
    print(f"\nBenchmark complete: {len(all_results)} scenarios · "
          f"mean reward {mean_r:.3f} (adj {mean_adj:.3f}) · "
          f"report → {REPORT_PATH.name}")
    print("\n" + report_md)


if __name__ == "__main__":
    main()
