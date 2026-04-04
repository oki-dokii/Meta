"""
app.py — Gradio UI for ContentModerationEnv (Hugging Face Spaces)
=================================================================
Live interactive demo + API endpoint for the OpenEnv benchmark.

Tabs
----
  1. Try It — step through individual scenarios
  2. Run Baseline — run the lexical agent over all 60 scenarios
  3. API Docs — curl / Python examples
"""

import json
import sys
from pathlib import Path

import gradio as gr

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from content_moderation_env import ContentModerationEnv, CampaignModerationEnv
from baseline_inference import decide, run_baseline, print_summary

# ── env singletons ────────────────────────────────────────────────────────────
SCENARIOS_PATH = SCRIPT_DIR / "moderation_benchmark.json"
env = ContentModerationEnv(str(SCENARIOS_PATH), seed=42)
ALL_IDS = env.scenario_ids

CAMPAIGN_PATH = SCRIPT_DIR / "campaign_benchmark.json"
campaign_env = CampaignModerationEnv(str(CAMPAIGN_PATH), seed=42)

# ── helpers ───────────────────────────────────────────────────────────────────

def _fmt_state(s: dict) -> str:
    lines = [f"**Text:** {s['text']}"]
    if s.get("audio_transcript"):
        lines.append(f"**Audio:** {s['audio_transcript']}")
    if s.get("visual_tags"):
        lines.append(f"**Visual tags:** {', '.join(s['visual_tags'])}")
    lines.append(f"**Previous flags:** {s['previous_flags']}  |  **Policy:** {s['platform_policy']}")
    return "\n\n".join(lines)


def _reward_bar(reward: float) -> str:
    filled = int(reward * 20)
    bar = "█" * filled + "░" * (20 - filled)
    emoji = "✅" if reward >= 0.8 else ("🟡" if reward >= 0.4 else "❌")
    return f"{emoji} [{bar}] {reward:.2f}"


# ── Tab 1: Try It ─────────────────────────────────────────────────────────────

def load_scenario(scenario_id: str):
    try:
        state = env.reset(scenario_id)
    except Exception as e:
        return f"Error: {e}", "", gr.update(visible=False)
    tier = env._current_scenario["tier"]
    show_sev = tier == "hard"
    return _fmt_state(state), f"**Tier:** `{tier}`", gr.update(visible=show_sev)


def submit_action(scenario_id: str, label: str, action: str, severity: int, rationale: str):
    try:
        env.reset(scenario_id)
    except Exception as e:
        return f"Error resetting: {e}", ""

    act_dict = {"label": label, "action": action, "severity": severity, "rationale": rationale}
    try:
        result = env.step(act_dict)
    except Exception as e:
        return f"Error in step(): {e}", ""

    info = result["info"]
    gt   = info["ground_truth"]
    bd   = info["score_breakdown"]
    reward = result["reward"]

    out_md = f"""
### Result

{_reward_bar(reward)}

| Component | Score |
|-----------|-------|
| Label correct | `{bd.get('label_correct', 'n/a')}` |
| Action correct | `{bd.get('action_correct', 'n/a')}` |
| Severity ±1 | `{bd.get('severity_within_1', 'n/a')}` |

**Ground truth:** label=`{gt['label']}`  action=`{gt['action']}`  severity=`{gt.get('severity', 'n/a')}`

> {gt.get('rationale', '')}
"""
    raw = json.dumps(result, indent=2, default=str)
    return out_md, f"```json\n{raw}\n```"


# ── Tab 2: Baseline ───────────────────────────────────────────────────────────

def run_baseline_tab(tier_filter: str):
    tf = None if tier_filter == "all" else tier_filter
    results = run_baseline(tier_filter=tf, seed=42, verbose=False)

    tiers = ["easy", "medium", "hard"]
    rows = []
    for t in tiers:
        rs = [r for r in results if r["tier"] == t]
        if not rs:
            continue
        rw = [r["reward"] for r in rs]
        mn = sum(rw) / len(rw)
        pct = sum(1 for r in rw if r == 1.0)
        rows.append([t, len(rs), f"{mn:.3f}", pct, sum(1 for r in rw if r == 0.0)])

    all_rw = [r["reward"] for r in results]
    overall = sum(all_rw) / len(all_rw) if all_rw else 0.0
    rows.append(["**OVERALL**", len(all_rw), f"{overall:.3f}",
                 sum(1 for r in all_rw if r == 1.0), sum(1 for r in all_rw if r == 0.0)])

    headers = ["Tier", "N", "Mean Reward", "Perfect (1.0)", "Zero (0.0)"]
    return rows, f"Baseline complete. Overall mean reward: **{overall:.3f}**"


# ── Tab 3: API examples ───────────────────────────────────────────────────────

API_CURL = """\
# 1. Reset (load a random scenario)
STATE=$(python -c "
import json, sys
sys.path.insert(0, '.')
from content_moderation_env import ContentModerationEnv
env = ContentModerationEnv('moderation_benchmark.json', seed=42)
state = env.reset()
print(json.dumps(state, indent=2))
")

# 2. Step (submit your action)
python -c "
import json, sys
sys.path.insert(0, '.')
from content_moderation_env import ContentModerationEnv
env = ContentModerationEnv('moderation_benchmark.json', seed=42)
env.reset('scen_hard_1')
result = env.step({
    'label': 'toxic',
    'action': 'escalate',
    'severity': 5,
    'rationale': 'Coordinated physical threat.'
})
print(json.dumps(result, indent=2))
"
"""

API_PYTHON = """\
from content_moderation_env import ContentModerationEnv

# Instantiate
env = ContentModerationEnv("moderation_benchmark.json", seed=42)
print(f"Loaded {env.num_scenarios} scenarios")

# Episode
state = env.reset()                        # random
# state = env.reset("scen_hard_1")         # specific
print(state["text"])

result = env.step({
    "label": "toxic",
    "action": "escalate",
    "severity": 4,
    "rationale": "Threat indicators detected."
})
print(f"Reward: {result['reward']}")
print(f"Breakdown: {result['info']['score_breakdown']}")
"""

# ── Build UI ──────────────────────────────────────────────────────────────────

THEME = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="violet",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif"],
)

with gr.Blocks(
    theme=THEME,
    title="ContentModerationEnv — OpenEnv Benchmark",
    css="""
    .gr-box { border-radius: 12px !important; }
    .header { text-align: center; padding: 1.5rem 0 0.5rem; }
    """,
) as demo:

    gr.Markdown("""
# 🛡️ ContentModerationEnv
### An OpenEnv benchmark for evaluating AI content moderation agents

> **60 scenarios** across 3 difficulty tiers (easy / medium / hard) ·
> **Partial-credit scoring** (0.0 – 1.0) · **Full OpenEnv API**  
> `reset()` · `step()` · `state()` · typed Pydantic models · reproducible baseline
""", elem_classes=["header"])

    with gr.Tabs():

        # ── Tab 1: Try It ─────────────────────────────────────────────────────
        with gr.Tab("🎮 Try It"):
            with gr.Row():
                with gr.Column(scale=1):
                    sid_dd = gr.Dropdown(
                        choices=ALL_IDS,
                        value=ALL_IDS[0],
                        label="Scenario ID",
                        interactive=True,
                    )
                    load_btn = gr.Button("Load Scenario", variant="primary")
                    tier_md  = gr.Markdown()

                with gr.Column(scale=2):
                    state_md = gr.Markdown(label="Observation")

            gr.Markdown("### Your moderation decision")
            with gr.Row():
                label_dd  = gr.Dropdown(
                    choices=["safe", "toxic", "spam", "misleading"],
                    value="safe", label="Label"
                )
                action_dd = gr.Dropdown(
                    choices=["allow", "warn", "remove", "shadowban", "escalate"],
                    value="allow", label="Action"
                )
                sev_slider = gr.Slider(1, 5, value=3, step=1,
                                       label="Severity (hard tier)", visible=False)

            rationale_tb = gr.Textbox(label="Rationale (optional)", lines=2,
                                      placeholder="Brief explanation …")
            step_btn  = gr.Button("Submit → env.step()", variant="primary")
            result_md = gr.Markdown()
            result_raw = gr.Markdown()

            load_btn.click(
                load_scenario,
                inputs=[sid_dd],
                outputs=[state_md, tier_md, sev_slider],
            )
            step_btn.click(
                submit_action,
                inputs=[sid_dd, label_dd, action_dd, sev_slider, rationale_tb],
                outputs=[result_md, result_raw],
            )
            demo.load(load_scenario, inputs=[sid_dd], outputs=[state_md, tier_md, sev_slider])

        # ── Tab 2: Baseline ───────────────────────────────────────────────────
        with gr.Tab("📊 Baseline"):
            gr.Markdown("""
### Lexical Rule-Based Baseline

A deterministic, no-LLM agent that uses regex patterns to classify content
and policy-based rules to choose an action. Run it to verify the environment
and as a comparison floor for LLM agents.
""")
            tier_radio = gr.Radio(
                choices=["all", "easy", "medium", "hard"],
                value="all", label="Tier to evaluate"
            )
            run_btn    = gr.Button("Run Baseline", variant="primary")
            status_md  = gr.Markdown()
            result_tbl = gr.Dataframe(
                headers=["Tier", "N", "Mean Reward", "Perfect (1.0)", "Zero (0.0)"],
                interactive=False,
            )
            run_btn.click(
                run_baseline_tab,
                inputs=[tier_radio],
                outputs=[result_tbl, status_md],
            )

        # ── Tab 3: API Docs ───────────────────────────────────────────────────
        with gr.Tab("📖 API Docs"):
            gr.Markdown("""
## Quick Start

```bash
git clone https://huggingface.co/spaces/sohambanerjee/content-moderation-env
cd content-moderation-env
pip install -r requirements.txt
```

### Python API
""")
            gr.Code(API_PYTHON, language="python", label="Python usage")
            gr.Markdown("### Shell / curl equivalent")
            gr.Code(API_CURL, language="bash", label="Shell usage")

            gr.Markdown("""
## Action Space

| Field | Type | Required | Values |
|-------|------|----------|--------|
| `label` | str | ✅ | `safe` · `toxic` · `spam` · `misleading` |
| `action` | str | ✅ | `allow` · `warn` · `remove` · `shadowban` · `escalate` |
| `severity` | int 1-5 | ❌ (scored in hard) | `1` (mild) → `5` (critical) |
| `rationale` | str | ❌ | Free text explanation |

## Reward Function

| Tier | Label | Action | Severity ±1 |
|------|-------|--------|-------------|
| easy / medium | 0.5 | 0.5 | — |
| hard | 0.4 | 0.4 | 0.2 |

## Baseline Scores (lexical agent, seed=42)

| Tier | Mean Reward |
|------|-------------|
| easy | ~0.60 |
| medium | ~0.35 |
| hard | ~0.20 |
| **overall** | **~0.38** |
""")

        # ── Tab 4: Campaign Detection ────────────────────────────────────
        with gr.Tab("🎯 Campaign Detection"):
            gr.Markdown("""
### Coordinated Campaign Detection
Review **3 posts from different accounts** and determine whether they form
a coordinated inauthentic behavior campaign.

| Field | Description |
|-------|-------------|
| `is_coordinated` | True if posts are from a coordinated operation |
| `action` | `allow` / `remove` / `shadowban` / `escalate` |
| Reward | +0.5 coordination detected · +0.5 action correct · -0.2 false positive |
""")
            load_camp_btn = gr.Button("Load Campaign Set", variant="primary")
            camp_type_md  = gr.Markdown()
            camp_posts_md = gr.Markdown()

            with gr.Row():
                is_coord_dd = gr.Dropdown(
                    choices=["true", "false"],
                    value="false",
                    label="Is Coordinated?"
                )
                camp_action_dd = gr.Dropdown(
                    choices=["allow", "remove", "shadowban", "escalate"],
                    value="allow",
                    label="Action"
                )
            reasoning_tb = gr.Textbox(
                label="Reasoning (optional)", lines=2,
                placeholder="Explain your coordination assessment..."
            )
            camp_submit_btn = gr.Button(
                "Submit → campaign_env.step()", variant="primary"
            )
            camp_result_md = gr.Markdown()

            def load_campaign():
                state = campaign_env.reset()
                posts_md = ""
                for i, p in enumerate(state["posts"], 1):
                    posts_md += f"**Post {i}** — account: `{p['account_id']}`"
                    posts_md += f" &nbsp;|&nbsp; +{p['posted_at_offset_minutes']} min"
                    posts_md += f" &nbsp;|&nbsp; platform: `{p['platform']}`\n\n"
                    posts_md += f"> {p['text']}\n\n"
                    if p.get("visual_tags"):
                        posts_md += f"*Visual signals: {', '.join(p['visual_tags'])}*\n\n"
                    posts_md += "---\n\n"
                return (
                    f"**Campaign:** `{state['campaign_id']}` &nbsp;|"
                    f"&nbsp; {state['num_posts']} posts\n",
                    posts_md
                )

            def submit_campaign(is_coord_str, action, reasoning):
                # reset to fresh random campaign and step it
                campaign_env.reset()
                action_dict = {
                    "is_coordinated": is_coord_str == "true",
                    "action": action,
                    "reasoning": reasoning,
                }
                result = campaign_env.step(action_dict)
                r   = result["reward"]
                gt  = result["info"]["ground_truth"]
                bd  = result["info"]["score_breakdown"]
                filled = int(max(r, 0) * 20)
                bar = "█" * filled + "░" * (20 - filled)
                emoji = "✅" if r >= 0.8 else ("🟡" if r >= 0.4 else "❌")
                out = f"{emoji} [{bar}] {r:.2f}\n\n"
                out += f"**Ground truth:** coordinated=`{gt['is_coordinated']}`"
                out += f"  action=`{gt['correct_action']}`\n\n"
                out += f"**Score breakdown:**\n\n"
                for k, v in bd.items():
                    out += f"  - `{k}`: `{v}`\n"
                return out

            load_camp_btn.click(
                load_campaign,
                outputs=[camp_type_md, camp_posts_md]
            )
            camp_submit_btn.click(
                submit_campaign,
                inputs=[is_coord_dd, camp_action_dd, reasoning_tb],
                outputs=[camp_result_md]
            )
            demo.load(load_campaign, outputs=[camp_type_md, camp_posts_md])

    gr.Markdown("""
---
<p style="text-align:center; color: #888; font-size: 0.85rem;">
ContentModerationEnv v2.0 · OpenEnv · MIT License
</p>
""")


# ── OpenEnv HTTP API routes ───────────────────────────────────────────────────
# Added to the Gradio FastAPI instance so POST /reset returns HTTP 200,
# satisfying the HF Space validator check.

from fastapi.responses import JSONResponse
from fastapi import Request


@demo.app.post("/reset")
async def api_reset(request: Request):
    """POST /reset  →  initial observation, HTTP 200"""
    try:
        body: dict = {}
        if request.headers.get("content-type", "").startswith("application/json"):
            body = await request.json()
    except Exception:
        body = {}
    scenario_id = body.get("scenario_id", None) if isinstance(body, dict) else None
    try:
        state = env.reset(scenario_id=scenario_id)
        return JSONResponse({"state": state, "status": "ok"})
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


@demo.app.post("/step")
async def api_step(request: Request):
    """POST /step  →  takes action dict, returns result"""
    try:
        body: dict = await request.json()
    except Exception:
        body = {}
    action = body.get("action", {}) if isinstance(body, dict) else {}
    try:
        result = env.step(action)
        return JSONResponse(result)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


@demo.app.get("/state")
async def api_state():
    """GET /state  →  current environment state"""
    try:
        state = env.state()
        return JSONResponse({"state": state, "status": "ok"})
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)


@demo.app.post("/campaign/reset")
async def campaign_reset(request: Request):
    """POST /campaign/reset  →  observation with 3 campaign posts"""
    try:
        body = await request.json()
    except Exception:
        body = {}
    campaign_id = body.get("campaign_id", None) if isinstance(body, dict) else None
    try:
        state = campaign_env.reset(campaign_id=campaign_id)
        return JSONResponse({"state": state, "status": "ok"})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@demo.app.post("/campaign/step")
async def campaign_step(request: Request):
    """POST /campaign/step  →  submit coordination verdict, returns reward"""
    try:
        body = await request.json()
    except Exception:
        body = {}
    action = body.get("action", {}) if isinstance(body, dict) else {}
    try:
        result = campaign_env.step(action)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)
