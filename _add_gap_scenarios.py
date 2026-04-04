"""
_add_gap_scenarios.py
=====================
Adds scenarios that close the 4 scoring gaps identified in the rubric:

Gap 1 (~2 pts) — Easy tier has no warn / escalate GT actions
Gap 2 (~1 pt)  — Easy tier has no misleading label
Gap 3 (~2 pts) — Hard tier has no ambiguous scenarios (single right answer)
Gap 4 (~1 pt)  — No cross-post coordination mechanic (handled via coordination_cluster)

Run once:
    python3 _add_gap_scenarios.py
"""
import json
import pathlib

HERE = pathlib.Path(__file__).parent
JSON_PATH = HERE / "moderation_benchmark.json"

HARD_RUBRIC   = {"label_correct": 0.4, "action_correct": 0.4, "severity_within_1": 0.2}
EASY_RUBRIC   = {"label_correct": 0.5, "action_correct": 0.5}

# ── Gap 1: Easy tier — warn / escalate GT actions ─────────────────────────────
GAP1 = [
    # warn actions
    {
        "id": "scen_easy_26", "tier": "easy",
        "state": {
            "text": "Your product designs look incredibly generic. Literally zero creativity.",
            "audio_transcript": None, "visual_tags": [],
            "previous_flags": 0, "platform_policy": "moderate"
        },
        "ground_truth": {
            "label": "toxic", "action": "warn",
            "rationale": "Mildly insulting, first offence on moderate platform — warn is appropriate."
        },
        "score_rubric": EASY_RUBRIC,
    },
    {
        "id": "scen_easy_27", "tier": "easy",
        "state": {
            "text": "JOIN NOW!! Get 1000 followers for FREE! DM us in the next 10 minutes!",
            "audio_transcript": None, "visual_tags": [],
            "previous_flags": 1, "platform_policy": "lenient"
        },
        "ground_truth": {
            "label": "spam", "action": "warn",
            "rationale": "Spam on a lenient platform with only 1 prior flag — warn before harder action."
        },
        "score_rubric": EASY_RUBRIC,
    },
    {
        "id": "scen_easy_28", "tier": "easy",
        "state": {
            "text": "This study proves vaccines cause autism — share widely before they delete it!",
            "audio_transcript": None, "visual_tags": [],
            "previous_flags": 0, "platform_policy": "moderate"
        },
        "ground_truth": {
            "label": "misleading", "action": "warn",
            "rationale": "First-time misinformation on a moderate platform — warn with fact-check label."
        },
        "score_rubric": EASY_RUBRIC,
    },
    # escalate actions
    {
        "id": "scen_easy_29", "tier": "easy",
        "state": {
            "text": "I know where you live. Don't think I won't show up.",
            "audio_transcript": None, "visual_tags": [],
            "previous_flags": 6, "platform_policy": "strict"
        },
        "ground_truth": {
            "label": "toxic", "action": "escalate",
            "rationale": "Explicit threat, 6 prior flags on strict platform — escalate to human review."
        },
        "score_rubric": EASY_RUBRIC,
    },
    {
        "id": "scen_easy_30", "tier": "easy",
        "state": {
            "text": "You're going to regret what you said. Trust me on that.",
            "audio_transcript": None, "visual_tags": [],
            "previous_flags": 5, "platform_policy": "strict"
        },
        "ground_truth": {
            "label": "toxic", "action": "escalate",
            "rationale": "Veiled threat by a repeat offender (5 flags) on strict platform."
        },
        "score_rubric": EASY_RUBRIC,
    },
]

# ── Gap 2: Easy tier — misleading label ──────────────────────────────────────
GAP2 = [
    {
        "id": "scen_easy_31", "tier": "easy",
        "state": {
            "text": "BREAKING: Scientists confirm 5G towers are transmitting viruses. Disabling your router protects you!",
            "audio_transcript": None, "visual_tags": [],
            "previous_flags": 0, "platform_policy": "strict"
        },
        "ground_truth": {
            "label": "misleading", "action": "remove",
            "rationale": "Clear health misinformation with potential to cause real harm — remove."
        },
        "score_rubric": EASY_RUBRIC,
    },
    {
        "id": "scen_easy_32", "tier": "easy",
        "state": {
            "text": "The moon landing was faked. Here's the original NASA blooper reel they hoped you'd never see!",
            "audio_transcript": None, "visual_tags": [],
            "previous_flags": 0, "platform_policy": "moderate"
        },
        "ground_truth": {
            "label": "misleading", "action": "warn",
            "rationale": "Conspiracy theory, no direct harm, first offence on moderate platform — warn."
        },
        "score_rubric": EASY_RUBRIC,
    },
]

# ── Gap 3: Hard tier — ambiguous scenarios (valid_actions list) ───────────────
# valid_actions: list of equally acceptable actions — agent scores for any
GAP3 = [
    {
        "id": "scen_hard_31", "tier": "hard",
        "state": {
            "text": "People like you shouldn't be allowed to reproduce. Just saying.",
            "audio_transcript": None,
            "visual_tags": ["hate_speech_signal"],
            "previous_flags": 1, "platform_policy": "moderate"
        },
        "ground_truth": {
            "label": "toxic",
            "action": "warn",                        # primary GT (backward compat)
            "valid_actions": ["warn", "remove"],     # both are defensible
            "severity": 3,
            "rationale": "Dehumanising comment, 1 prior flag, moderate policy — warn or remove are both valid."
        },
        "score_rubric": HARD_RUBRIC,
    },
    {
        "id": "scen_hard_32", "tier": "hard",
        "state": {
            "text": "This election was obviously stolen. The math simply doesn't add up when you look at the swing counties.",
            "audio_transcript": None,
            "visual_tags": ["political_content", "election_keywords"],
            "previous_flags": 0, "platform_policy": "moderate"
        },
        "ground_truth": {
            "label": "misleading",
            "action": "warn",
            "valid_actions": ["warn", "remove"],
            "severity": 2,
            "rationale": "Election misinformation, no prior offence, moderate policy — warn or remove both defensible."
        },
        "score_rubric": HARD_RUBRIC,
    },
    {
        "id": "scen_hard_33", "tier": "hard",
        "state": {
            "text": "If you see this person in your neighbourhood, make sure everyone knows what they did.",
            "audio_transcript": None,
            "visual_tags": ["personal_photo", "location_tag"],
            "previous_flags": 2, "platform_policy": "strict"
        },
        "ground_truth": {
            "label": "toxic",
            "action": "remove",
            "valid_actions": ["remove", "shadowban"],
            "severity": 4,
            "rationale": "Potential doxxing incitement, strict policy — remove or shadowban are both valid."
        },
        "score_rubric": HARD_RUBRIC,
    },
    {
        "id": "scen_hard_34", "tier": "hard",
        "state": {
            "text": "Try mixing bleach and ammonia — it'll clear your sinuses immediately. Worked for me!",
            "audio_transcript": None,
            "visual_tags": ["home_remedy_tag"],
            "previous_flags": 0, "platform_policy": "lenient"
        },
        "ground_truth": {
            "label": "misleading",
            "action": "remove",
            "valid_actions": ["remove", "warn"],
            "severity": 4,
            "rationale": "Dangerous health misinformation on a lenient platform — remove is primary but warn is defensible."
        },
        "score_rubric": HARD_RUBRIC,
    },
    {
        "id": "scen_hard_35", "tier": "hard",
        "state": {
            "text": "You're all pathetic losers. Every single one of you. Get a real job.",
            "audio_transcript": "Aggressive shouting tone throughout the clip.",
            "visual_tags": ["aggressive_gesture"],
            "previous_flags": 4, "platform_policy": "moderate"
        },
        "ground_truth": {
            "label": "toxic",
            "action": "shadowban",
            "valid_actions": ["shadowban", "remove"],
            "severity": 3,
            "rationale": "Repeated toxic behaviour (4 flags), moderate policy — shadowban or remove both defensible."
        },
        "score_rubric": HARD_RUBRIC,
    },
]

# ── Gap 4: Cross-post coordination clusters ───────────────────────────────────
# coordination_cluster: tag links posts that form a coordinated campaign.
# The env groups these when building queue episodes and awards a coordination_bonus
# if all posts in the cluster are handled correctly.
GAP4 = [
    # Cluster A: coordinated spam campaign (3 posts, same actor/pattern)
    {
        "id": "scen_coord_easy_1", "tier": "easy",
        "coordination_cluster": "spam_cluster_A",
        "coordination_note": "Post 1/3 of coordinated crypto spam campaign. Standalone: spam/warn.",
        "state": {
            "text": "Huge crypto opportunity — 10x returns guaranteed! DM for access code.",
            "audio_transcript": None, "visual_tags": ["bulk_posting_pattern"],
            "previous_flags": 1, "platform_policy": "strict"
        },
        "ground_truth": {
            "label": "spam", "action": "warn",
            "rationale": "First confirmed post in a coordinated spam cluster — warn escalates to cluster-level action."
        },
        "score_rubric": EASY_RUBRIC,
    },
    {
        "id": "scen_coord_medium_1", "tier": "medium",
        "coordination_cluster": "spam_cluster_A",
        "coordination_note": "Post 2/3 of same campaign. Prior cluster post should raise suspicion.",
        "state": {
            "text": "Anyone else making insane gains with CosmoCoin? My portfolio is up 800% — ask me how!",
            "audio_transcript": None, "visual_tags": ["known_bot_network", "copied_profile_picture"],
            "previous_flags": 2, "platform_policy": "strict"
        },
        "ground_truth": {
            "label": "spam", "action": "shadowban",
            "rationale": "Second post from same coordinated campaign — shadowban to limit spread."
        },
        "score_rubric": {"label_correct": 0.5, "action_correct": 0.5},
    },
    {
        "id": "scen_coord_hard_1", "tier": "hard",
        "coordination_cluster": "spam_cluster_A",
        "coordination_note": "Post 3/3 — cluster confirmed. Full escalation warranted.",
        "state": {
            "text": "FINAL HOURS: CosmoCoin presale ending! This is your LAST chance before it's gone. Link in bio!",
            "audio_transcript": None,
            "visual_tags": ["known_bot_network", "bulk_posting_pattern", "affiliate_link_detected"],
            "previous_flags": 4, "platform_policy": "strict"
        },
        "ground_truth": {
            "label": "spam", "action": "escalate",
            "severity": 4,
            "rationale": "Third post in confirmed coordinated inauthentic campaign — escalate for account-level action."
        },
        "score_rubric": HARD_RUBRIC,
    },

    # Cluster B: coordinated harassment campaign
    {
        "id": "scen_coord_easy_2", "tier": "easy",
        "coordination_cluster": "harassment_cluster_B",
        "coordination_note": "Post 1/3 — begins as borderline comment from new account.",
        "state": {
            "text": "Interesting take. Very controversial choice for someone in your position.",
            "audio_transcript": None, "visual_tags": [],
            "previous_flags": 0, "platform_policy": "moderate"
        },
        "ground_truth": {
            "label": "safe", "action": "allow",
            "rationale": "Borderline but not actionable in isolation — allow."
        },
        "score_rubric": EASY_RUBRIC,
    },
    {
        "id": "scen_coord_medium_2", "tier": "medium",
        "coordination_cluster": "harassment_cluster_B",
        "coordination_note": "Post 2/3 — same user escalates. Pattern emerging.",
        "state": {
            "text": "People like you should stay out of public life. You clearly can't handle the heat.",
            "audio_transcript": None, "visual_tags": [],
            "previous_flags": 1, "platform_policy": "moderate"
        },
        "ground_truth": {
            "label": "toxic", "action": "warn",
            "rationale": "Targeting escalation from same cluster — warn to stop the pattern."
        },
        "score_rubric": {"label_correct": 0.5, "action_correct": 0.5},
    },
    {
        "id": "scen_coord_hard_2", "tier": "hard",
        "coordination_cluster": "harassment_cluster_B",
        "coordination_note": "Post 3/3 — full doxxing threat. Cluster context makes this critical.",
        "state": {
            "text": "We know your schedule. See you Thursday.",
            "audio_transcript": None,
            "visual_tags": ["location_tag", "doxxing_pattern"],
            "previous_flags": 2, "platform_policy": "moderate"
        },
        "ground_truth": {
            "label": "toxic", "action": "escalate",
            "severity": 5,
            "rationale": "Third post in harassment cluster — credible implied threat, escalate immediately."
        },
        "score_rubric": HARD_RUBRIC,
    },
]

# ── Write to JSON ─────────────────────────────────────────────────────────────
data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
existing = {s["id"] for s in data}
added = 0
for s in GAP1 + GAP2 + GAP3 + GAP4:
    if s["id"] not in existing:
        data.append(s)
        added += 1

JSON_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

tiers = {}
for s in data:
    tiers[s["tier"]] = tiers.get(s["tier"], 0) + 1

print(f"Added {added} scenarios. Total: {len(data)}")
print(f"Tiers: easy={tiers.get('easy',0)}, medium={tiers.get('medium',0)}, hard={tiers.get('hard',0)}")
clusters = {}
for s in data:
    if "coordination_cluster" in s:
        c = s["coordination_cluster"]
        clusters[c] = clusters.get(c, 0) + 1
print(f"Coordination clusters: {clusters}")
