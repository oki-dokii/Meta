"""
_add_ambiguous_scenarios.py
===========================
Adds 5 high-quality hard-tier ambiguous scenarios where remove and shadowban
are both defensible — the most realistic grey area in content moderation.

These scenarios specifically target the remove/shadowban decision boundary:
  - remove:    content is clearly harmful, visibility = 0 immediately
  - shadowban: content is limited (account sees their own posts, others don't)
               useful when: preserving evidence, uncertain severity, first offense

Run once:
    python3 _add_ambiguous_scenarios.py
"""
import json
import pathlib

HERE = pathlib.Path(__file__).parent
JSON_PATH = HERE / "moderation_benchmark.json"

HARD_RUBRIC = {"label_correct": 0.4, "action_correct": 0.4, "severity_within_1": 0.2}

AMBIGUOUS = [
    # ── 1. Implied dogwhistle — evidence preservation argument ────────────────
    {
        "id": "scen_hard_36", "tier": "hard",
        "state": {
            "text": "The real problem with this city is the 'diversity' they keep importing. "
                    "Our neighbourhood used to be safe.",
            "audio_transcript": None,
            "visual_tags": ["hate_speech_signal", "coded_language_detected"],
            "previous_flags": 1,
            "platform_policy": "strict"
        },
        "ground_truth": {
            "label": "toxic",
            "action": "shadowban",                       # primary GT
            "valid_actions": ["shadowban", "remove"],    # both defensible
            "severity": 3,
            "rationale": (
                "Coded xenophobia ('diversity imports'). shadowban limits spread while "
                "preserving evidence for pattern review; remove is also valid on a strict "
                "platform with a prior flag."
            )
        },
        "score_rubric": HARD_RUBRIC,
    },

    # ── 2. Repeat offender, ambiguous severity ────────────────────────────────
    {
        "id": "scen_hard_37", "tier": "hard",
        "state": {
            "text": "Nobody actually cares about these people. They should just disappear.",
            "audio_transcript": None,
            "visual_tags": ["community_targeting_signal"],
            "previous_flags": 3,
            "platform_policy": "moderate"
        },
        "ground_truth": {
            "label": "toxic",
            "action": "remove",
            "valid_actions": ["remove", "shadowban"],
            "severity": 3,
            "rationale": (
                "Dehumanising language targeting a community, 3 prior flags on a moderate "
                "platform. Remove is the primary action; shadowban is defensible if "
                "platform prefers quiet suppression for appeals purposes."
            )
        },
        "score_rubric": HARD_RUBRIC,
    },

    # ── 3. Non-English hate speech — context uncertainty ──────────────────────
    {
        "id": "scen_hard_38", "tier": "hard",
        "state": {
            "text": "Esos malditos inmigrantes arruinan todo. Habría que deportarlos a todos.",
            "audio_transcript": None,
            "visual_tags": ["translated_content", "nationalist_imagery"],
            "previous_flags": 0,
            "platform_policy": "strict"
        },
        "ground_truth": {
            "label": "toxic",
            "action": "shadowban",
            "valid_actions": ["shadowban", "remove"],
            "severity": 4,
            "rationale": (
                "Spanish: 'Those damned immigrants ruin everything. They should all be "
                "deported.' Clear xenophobia on strict platform. shadowban is primary to "
                "allow human review of translated context; remove is equally valid."
            )
        },
        "score_rubric": HARD_RUBRIC,
    },

    # ── 4. Coordinated inauthentic amplification ──────────────────────────────
    {
        "id": "scen_hard_39", "tier": "hard",
        "state": {
            "text": "Sharing this before it gets censored. These people are dangerous and "
                    "your children aren't safe around them.",
            "audio_transcript": None,
            "visual_tags": [
                "coordinated_amplification",
                "known_bot_network",
                "minority_targeting_signal"
            ],
            "previous_flags": 2,
            "platform_policy": "strict"
        },
        "ground_truth": {
            "label": "toxic",
            "action": "remove",
            "valid_actions": ["remove", "shadowban"],
            "severity": 5,
            "rationale": (
                "Coordinated hate amplification targeting a minority group, severity-5 due "
                "to bot-network signals. Remove is primary given the coordinated pattern; "
                "shadowban is defensible to preserve network forensics."
            )
        },
        "score_rubric": HARD_RUBRIC,
    },

    # ── 5. Satire/art intent claim with genuine harm signal ───────────────────
    {
        "id": "scen_hard_40", "tier": "hard",
        "state": {
            "text": "It's just a joke — if your group can't handle a little ribbing about "
                    "being subhuman, maybe you really are inferior.",
            "audio_transcript": None,
            "visual_tags": ["satire_label_claimed", "hate_speech_signal"],
            "previous_flags": 2,
            "platform_policy": "moderate"
        },
        "ground_truth": {
            "label": "toxic",
            "action": "shadowban",
            "valid_actions": ["shadowban", "remove"],
            "severity": 4,
            "rationale": (
                "Hate speech wrapped in claimed satire. 'Subhuman' is not protected speech "
                "even with satire framing. shadowban while claim is reviewed; remove is "
                "equally valid given the explicit dehumanising language."
            )
        },
        "score_rubric": HARD_RUBRIC,
    },
]

# ── Write ─────────────────────────────────────────────────────────────────────
data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
existing = {s["id"] for s in data}
added = 0
for s in AMBIGUOUS:
    if s["id"] not in existing:
        data.append(s)
        added += 1

JSON_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

total_ambig = sum(1 for s in data if "valid_actions" in s.get("ground_truth", {}))
tiers = {}
for s in data:
    tiers[s["tier"]] = tiers.get(s["tier"], 0) + 1

print(f"Added {added} scenarios. Total: {len(data)}")
print(f"Tiers: easy={tiers.get('easy',0)}, medium={tiers.get('medium',0)}, hard={tiers.get('hard',0)}")
print(f"Ambiguous scenarios (valid_actions): {total_ambig}")
