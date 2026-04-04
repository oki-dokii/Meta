"""
_fill_easy_gaps.py
==================
Fills the 4 missing/thin GT combos in the easy tier:

  misleading/shadowban  → 0  (add 3)
  toxic/shadowban       → 0  (add 3)
  toxic/warn            → 1  (add 3 more)
  misleading/escalate   → 1  (add 2 more)

All scenarios are deliberately unambiguous (easy tier) but cover
action/label combos missing from the current dataset.

Run once:
    python3 _fill_easy_gaps.py
"""
import json, pathlib

HERE = pathlib.Path(__file__).parent
JSON_PATH = HERE / "moderation_benchmark.json"
RUBRIC = {"label_correct": 0.5, "action_correct": 0.5}

SCENARIOS = [

    # ── misleading / shadowban (3 new) ────────────────────────────────────────
    # shadowban = limit spread quietly without alerting the user;
    # used when removal would cause backlash but content must not amplify.
    {
        "id": "scen_easy_33", "tier": "easy",
        "state": {
            "text": "Drinking hydrogen peroxide kills cancer cells. Share this with anyone who needs it.",
            "audio_transcript": None, "visual_tags": [],
            "previous_flags": 1, "platform_policy": "moderate"
        },
        "ground_truth": {
            "label": "misleading", "action": "shadowban",
            "rationale": "Dangerous health misinformation with prior flag. Shadowban to "
                         "limit spread while keeping evidence for review on moderate platform."
        },
        "score_rubric": RUBRIC,
    },
    {
        "id": "scen_easy_34", "tier": "easy",
        "state": {
            "text": "Scientists CONFIRM: eating raw garlic every morning cures diabetes. "
                    "Big Pharma is furious — they're trying to get this post deleted!",
            "audio_transcript": None, "visual_tags": [],
            "previous_flags": 2, "platform_policy": "moderate"
        },
        "ground_truth": {
            "label": "misleading", "action": "shadowban",
            "rationale": "Repeat misinformation spreader (2 flags). Shadowban limits "
                         "algorithmic amplification; full removal held for appeals process."
        },
        "score_rubric": RUBRIC,
    },
    {
        "id": "scen_easy_35", "tier": "easy",
        "state": {
            "text": "The government is adding lithium to drinking water to make people docile. "
                    "Filter your water NOW.",
            "audio_transcript": None, "visual_tags": [],
            "previous_flags": 1, "platform_policy": "lenient"
        },
        "ground_truth": {
            "label": "misleading", "action": "shadowban",
            "rationale": "Conspiracy misinformation with prior flag. Lenient platform "
                         "prefers quiet suppression over outright removal."
        },
        "score_rubric": RUBRIC,
    },

    # ── toxic / shadowban (3 new) ─────────────────────────────────────────────
    # shadowban on toxic = suppress repeat low-grade toxicity without removing,
    # giving the user a silent warning period.
    {
        "id": "scen_easy_36", "tier": "easy",
        "state": {
            "text": "People who believe this stuff are genuinely the dumbest people alive.",
            "audio_transcript": None, "visual_tags": [],
            "previous_flags": 3, "platform_policy": "moderate"
        },
        "ground_truth": {
            "label": "toxic", "action": "shadowban",
            "rationale": "Repeated low-grade insults (3 flags). Shadowban to suppress "
                         "without alerting; moderate policy avoids hard removal for now."
        },
        "score_rubric": RUBRIC,
    },
    {
        "id": "scen_easy_37", "tier": "easy",
        "state": {
            "text": "Oh wow, another opinion from someone who clearly failed basic education. "
                    "Shocking. Not.",
            "audio_transcript": None, "visual_tags": [],
            "previous_flags": 4, "platform_policy": "lenient"
        },
        "ground_truth": {
            "label": "toxic", "action": "shadowban",
            "rationale": "Chronic low-grade toxicity (4 flags). Lenient policy shadowbans "
                         "before escalating to removal; content is mild but pattern is clear."
        },
        "score_rubric": RUBRIC,
    },
    {
        "id": "scen_easy_38", "tier": "easy",
        "state": {
            "text": "Imagine being this wrong and still having the audacity to post publicly. "
                    "Embarrassing for you and everyone who agrees with you.",
            "audio_transcript": None, "visual_tags": [],
            "previous_flags": 3, "platform_policy": "moderate"
        },
        "ground_truth": {
            "label": "toxic", "action": "shadowban",
            "rationale": "Targeted personal insults, repeat offender. Shadowban to quietly "
                         "limit reach; moderate platform reserves removal for direct threats."
        },
        "score_rubric": RUBRIC,
    },

    # ── toxic / warn (3 more, total → 4) ─────────────────────────────────────
    # warn = first-time mild toxicity on non-strict platform
    {
        "id": "scen_easy_39", "tier": "easy",
        "state": {
            "text": "This is the most brain-dead take I have ever seen on this platform.",
            "audio_transcript": None, "visual_tags": [],
            "previous_flags": 0, "platform_policy": "lenient"
        },
        "ground_truth": {
            "label": "toxic", "action": "warn",
            "rationale": "Mildly insulting, zero prior flags, lenient platform — warn is "
                         "the appropriate first-step action."
        },
        "score_rubric": RUBRIC,
    },
    {
        "id": "scen_easy_40", "tier": "easy",
        "state": {
            "text": "People like you are exactly why comment sections should be disabled.",
            "audio_transcript": None, "visual_tags": [],
            "previous_flags": 0, "platform_policy": "moderate"
        },
        "ground_truth": {
            "label": "toxic", "action": "warn",
            "rationale": "Condescending but non-threatening. First offence on a moderate "
                         "platform — warn before escalating."
        },
        "score_rubric": RUBRIC,
    },
    {
        "id": "scen_easy_41", "tier": "easy",
        "state": {
            "text": "Congratulations on posting the stupidest comment in this thread. "
                    "That's an achievement.",
            "audio_transcript": None, "visual_tags": [],
            "previous_flags": 0, "platform_policy": "moderate"
        },
        "ground_truth": {
            "label": "toxic", "action": "warn",
            "rationale": "Low-grade personal insult, no prior flags, moderate policy — "
                         "issue a warning first."
        },
        "score_rubric": RUBRIC,
    },

    # ── misleading / escalate (2 more, total → 3) ────────────────────────────
    # escalate = repeat misinformation with high-harm potential
    {
        "id": "scen_easy_42", "tier": "easy",
        "state": {
            "text": "Do NOT give your children the flu vaccine. Three kids in my town died "
                    "within 48 hours of getting it. The CDC is covering it up.",
            "audio_transcript": None, "visual_tags": [],
            "previous_flags": 5, "platform_policy": "strict"
        },
        "ground_truth": {
            "label": "misleading", "action": "escalate",
            "rationale": "Anti-vax misinformation with fabricated deaths, 5 prior flags, "
                         "strict platform — escalate for human review and account action."
        },
        "score_rubric": RUBRIC,
    },
    {
        "id": "scen_easy_43", "tier": "easy",
        "state": {
            "text": "BREAKING: WHO confirms that PCR tests were designed to produce false "
                    "positives and the entire pandemic was staged. Share before it's deleted!",
            "audio_transcript": None, "visual_tags": [],
            "previous_flags": 4, "platform_policy": "strict"
        },
        "ground_truth": {
            "label": "misleading", "action": "escalate",
            "rationale": "High-profile health misinformation, repeat offender (4 flags), "
                         "strict platform — escalate immediately."
        },
        "score_rubric": RUBRIC,
    },
]

# ── Write ─────────────────────────────────────────────────────────────────────
data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
existing = {s["id"] for s in data}
added = 0
for s in SCENARIOS:
    if s["id"] not in existing:
        data.append(s)
        added += 1

JSON_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

# Recount
from collections import Counter
easy = [s for s in data if s["tier"] == "easy"]
labels  = Counter(s["ground_truth"]["label"]  for s in easy)
actions = Counter(s["ground_truth"]["action"] for s in easy)
combos  = Counter((s["ground_truth"]["label"], s["ground_truth"]["action"]) for s in easy)

print(f"Added {added} scenarios. Easy total: {len(easy)}, Overall: {len(data)}")
print(f"\nLabels:  {dict(sorted(labels.items()))}")
print(f"Actions: {dict(sorted(actions.items()))}")
print("\nCombos:")
for (lbl, act), cnt in sorted(combos.items()):
    flag = "  ← NEW" if cnt >= 2 and (lbl, act) in [
        ("misleading","shadowban"),("toxic","shadowban"),
        ("toxic","warn"),("misleading","escalate")] else ""
    print(f"  {lbl:12} / {act:10} → {cnt}{flag}")
