"""
_add_campaign_scenarios.py
==========================
Adds 3 fully-formed campaign episodes to moderation_benchmark.json.
Each campaign has 3 posts that share a campaign_id. The posts are ordered
by campaign_post_index and escalate in severity.

The agent sees campaign_id in the state dict and should recognise coordinated
behaviour. Escalating ALL posts in a campaign episode earns +0.15 bonus.

Campaigns:
  camp_crypto_001  — coordinated crypto pump-and-dump spam
  camp_doxx_002    — escalating targeted harassment / doxxing threat
  camp_disinfo_003 — coordinated health misinformation (vaccine/cure hoax)

Run once:
    python3 _add_campaign_scenarios.py
"""
import json
import pathlib

HERE = pathlib.Path(__file__).parent
JSON_PATH = HERE / "moderation_benchmark.json"

HARD_RUBRIC   = {"label_correct": 0.4, "action_correct": 0.4, "severity_within_1": 0.2}
MEDIUM_RUBRIC = {"label_correct": 0.5, "action_correct": 0.5}
EASY_RUBRIC   = {"label_correct": 0.5, "action_correct": 0.5}

CAMPAIGNS = [

    # ── Campaign 1: Coordinated crypto pump-and-dump ──────────────────────────
    {
        "id": "camp_crypto_001_p1",
        "tier": "easy",
        "campaign_id": "camp_crypto_001",
        "campaign_post_index": 1,
        "campaign_total_posts": 3,
        "state": {
            "text": "CosmoCoin is the next big thing. Early investors are already up 300%. Don't miss out!",
            "audio_transcript": None,
            "visual_tags": ["affiliate_link_detected"],
            "previous_flags": 1,
            "platform_policy": "strict",
        },
        "ground_truth": {
            "label": "spam",
            "action": "escalate",
            "rationale": (
                "Post 1 of a coordinated crypto pump-and-dump campaign. "
                "Escalate immediately — campaign_id signals coordinated inauthentic behaviour."
            ),
        },
        "score_rubric": EASY_RUBRIC,
    },
    {
        "id": "camp_crypto_001_p2",
        "tier": "medium",
        "campaign_id": "camp_crypto_001",
        "campaign_post_index": 2,
        "campaign_total_posts": 3,
        "state": {
            "text": "My portfolio went from $500 to $15,000 in a week with CosmoCoin. "
                    "The listing is happening soon — last chance to buy before it explodes!",
            "audio_transcript": None,
            "visual_tags": ["known_bot_network", "bulk_posting_pattern", "affiliate_link_detected"],
            "previous_flags": 2,
            "platform_policy": "strict",
        },
        "ground_truth": {
            "label": "spam",
            "action": "escalate",
            "rationale": (
                "Post 2/3 of camp_crypto_001. Bot network + bulk posting confirmed. "
                "Escalate to stop coordinated financial fraud."
            ),
        },
        "score_rubric": MEDIUM_RUBRIC,
    },
    {
        "id": "camp_crypto_001_p3",
        "tier": "hard",
        "campaign_id": "camp_crypto_001",
        "campaign_post_index": 3,
        "campaign_total_posts": 3,
        "state": {
            "text": "FINAL ALERT: CosmoCoin presale closes in 2 hours. "
                    "100% guaranteed returns. DM RIGHT NOW before the devs close it.",
            "audio_transcript": "Voiceover: 'Send your crypto today, double it tomorrow.'",
            "visual_tags": ["known_bot_network", "bulk_posting_pattern",
                            "affiliate_link_detected", "urgency_language"],
            "previous_flags": 4,
            "platform_policy": "strict",
        },
        "ground_truth": {
            "label": "spam",
            "action": "escalate",
            "severity": 5,
            "rationale": (
                "Post 3/3 of camp_crypto_001. Full coordinated fraud confirmed across "
                "3 posts. Severity 5 — financial harm at scale. Escalate for account ban."
            ),
        },
        "score_rubric": HARD_RUBRIC,
    },

    # ── Campaign 2: Escalating targeted harassment → doxxing threat ───────────
    {
        "id": "camp_doxx_002_p1",
        "tier": "easy",
        "campaign_id": "camp_doxx_002",
        "campaign_post_index": 1,
        "campaign_total_posts": 3,
        "state": {
            "text": "Interesting how @journalist_name always seems to be at the right place "
                    "at the right time. Almost like they're tipped off.",
            "audio_transcript": None,
            "visual_tags": ["targeted_account_tag"],
            "previous_flags": 0,
            "platform_policy": "moderate",
        },
        "ground_truth": {
            "label": "toxic",
            "action": "escalate",
            "rationale": (
                "Post 1/3 of camp_doxx_002. Begins campaign of targeted harassment. "
                "Escalate to flag campaign for monitoring — even if individually mild."
            ),
        },
        "score_rubric": EASY_RUBRIC,
    },
    {
        "id": "camp_doxx_002_p2",
        "tier": "medium",
        "campaign_id": "camp_doxx_002",
        "campaign_post_index": 2,
        "campaign_total_posts": 3,
        "state": {
            "text": "Since @journalist_name wants to keep writing these hit pieces, "
                    "maybe people should know what neighbourhood they live in. "
                    "Hint: it starts with 'W'.",
            "audio_transcript": None,
            "visual_tags": ["targeted_account_tag", "location_hint"],
            "previous_flags": 1,
            "platform_policy": "moderate",
        },
        "ground_truth": {
            "label": "toxic",
            "action": "escalate",
            "rationale": (
                "Post 2/3 of camp_doxx_002. Incipient doxxing with geographic hint. "
                "Escalate — campaign context makes this a credible threat pattern."
            ),
        },
        "score_rubric": MEDIUM_RUBRIC,
    },
    {
        "id": "camp_doxx_002_p3",
        "tier": "hard",
        "campaign_id": "camp_doxx_002",
        "campaign_post_index": 3,
        "campaign_total_posts": 3,
        "state": {
            "text": "Found it. Here's @journalist_name's full address. "
                    "Someone should pay them a visit and explain how journalism works.",
            "audio_transcript": None,
            "visual_tags": ["doxxing_pattern", "personal_information", "targeted_account_tag"],
            "previous_flags": 2,
            "platform_policy": "moderate",
        },
        "ground_truth": {
            "label": "toxic",
            "action": "escalate",
            "severity": 5,
            "rationale": (
                "Post 3/3 of camp_doxx_002. Full doxxing with implied physical threat. "
                "Severity 5. Escalate immediately for law enforcement referral."
            ),
        },
        "score_rubric": HARD_RUBRIC,
    },

    # ── Campaign 3: Coordinated health misinformation ─────────────────────────
    {
        "id": "camp_disinfo_003_p1",
        "tier": "easy",
        "campaign_id": "camp_disinfo_003",
        "campaign_post_index": 1,
        "campaign_total_posts": 3,
        "state": {
            "text": "Doctors don't want you to know this — there's a natural cure "
                    "for most cancers that costs $3. Big Pharma is suppressing it.",
            "audio_transcript": None,
            "visual_tags": ["health_misinformation_signal"],
            "previous_flags": 0,
            "platform_policy": "strict",
        },
        "ground_truth": {
            "label": "misleading",
            "action": "escalate",
            "rationale": (
                "Post 1/3 of camp_disinfo_003. Coordinated cancer-cure misinformation. "
                "Escalate to prevent health harm — campaign_id signals coordinated spread."
            ),
        },
        "score_rubric": EASY_RUBRIC,
    },
    {
        "id": "camp_disinfo_003_p2",
        "tier": "medium",
        "campaign_id": "camp_disinfo_003",
        "campaign_post_index": 2,
        "campaign_total_posts": 3,
        "state": {
            "text": "I used the method from the previous post and my tumour shrank in 6 weeks. "
                    "My oncologist is furious. Share this before they delete it!",
            "audio_transcript": None,
            "visual_tags": ["health_misinformation_signal", "bulk_posting_pattern"],
            "previous_flags": 1,
            "platform_policy": "strict",
        },
        "ground_truth": {
            "label": "misleading",
            "action": "escalate",
            "rationale": (
                "Post 2/3 of camp_disinfo_003. False testimonial amplifying cancer "
                "misinformation. Escalate — could cause patients to abandon treatment."
            ),
        },
        "score_rubric": MEDIUM_RUBRIC,
    },
    {
        "id": "camp_disinfo_003_p3",
        "tier": "hard",
        "campaign_id": "camp_disinfo_003",
        "campaign_post_index": 3,
        "campaign_total_posts": 3,
        "state": {
            "text": "SHARE EVERYWHERE: The compound is apricot seed extract. 3 seeds/day. "
                    "Chemo is poison — they're killing you for profit. Pass this on!",
            "audio_transcript": "Narrator urges listeners to stop cancer treatment immediately.",
            "visual_tags": ["health_misinformation_signal", "urgency_language",
                            "bulk_posting_pattern", "known_bot_network"],
            "previous_flags": 3,
            "platform_policy": "strict",
        },
        "ground_truth": {
            "label": "misleading",
            "action": "escalate",
            "severity": 5,
            "rationale": (
                "Post 3/3 of camp_disinfo_003. Instructs cancer patients to stop chemo. "
                "Severity 5 — imminent risk of death. Escalate immediately."
            ),
        },
        "score_rubric": HARD_RUBRIC,
    },
]

# ── Write ─────────────────────────────────────────────────────────────────────
data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
existing = {s["id"] for s in data}
added = 0
for s in CAMPAIGNS:
    if s["id"] not in existing:
        data.append(s)
        added += 1

JSON_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

# Count campaigns
camp_map = {}
for s in data:
    cid = s.get("campaign_id")
    if cid:
        camp_map.setdefault(cid, []).append(s["id"])

tiers = {}
for s in data:
    tiers[s["tier"]] = tiers.get(s["tier"], 0) + 1

print(f"Added {added} scenarios. Total: {len(data)}")
print(f"Tiers: easy={tiers.get('easy',0)}, medium={tiers.get('medium',0)}, hard={tiers.get('hard',0)}")
print(f"\nCampaigns ({len(camp_map)}):")
for cid, ids in sorted(camp_map.items()):
    print(f"  {cid}: {len(ids)} posts — {ids}")
