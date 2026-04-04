"""Appends 15 new scenarios to moderation_benchmark.json"""
import json, pathlib

HERE = pathlib.Path(__file__).parent
JSON_PATH = HERE / "moderation_benchmark.json"

HARD_RUBRIC = {"label_correct": 0.4, "action_correct": 0.4, "severity_within_1": 0.2}
EASY_RUBRIC = {"label_correct": 0.5, "action_correct": 0.5}

NEW = [
    {"id":"scen_hard_21","tier":"hard",
     "state":{"text":"Oh sure, let's just give everyone guns, that'll definitely solve all our problems. What could possibly go wrong?",
              "audio_transcript":None,"visual_tags":["political_content"],"previous_flags":0,"platform_policy":"moderate"},
     "ground_truth":{"label":"misleading","action":"warn","severity":2,"rationale":"Sarcastic political speech. Warn for framing, not toxic."},
     "score_rubric":HARD_RUBRIC},
    {"id":"scen_hard_22","tier":"hard",
     "state":{"text":"Just sharing my thoughts on today's news. Very interesting developments!",
              "audio_transcript":None,"visual_tags":["known_bot_network","copied_profile_picture","bulk_posting_pattern"],"previous_flags":8,"platform_policy":"strict"},
     "ground_truth":{"label":"spam","action":"shadowban","severity":4,"rationale":"Benign text but metadata signals coordinated inauthentic behavior."},
     "score_rubric":HARD_RUBRIC},
    {"id":"scen_hard_23","tier":"hard",
     "state":{"text":"For acute overdose cases, administer 0.4-2mg naloxone IV. Repeat every 2-3 minutes if no response. Max 10mg before reassessing.",
              "audio_transcript":"Medical grand rounds recording, hospital setting",
              "visual_tags":["medical_setting","professional_attire","hospital_environment"],"previous_flags":0,"platform_policy":"moderate"},
     "ground_truth":{"label":"safe","action":"allow","severity":1,"rationale":"Clinical medical discussion confirmed by audio and visual context."},
     "score_rubric":HARD_RUBRIC},
    {"id":"scen_hard_24","tier":"hard",
     "state":{"text":"Just a reminder that this public figure's home address and daily schedule are publicly available on their website.",
              "audio_transcript":None,"visual_tags":["doxxing_pattern","personal_information"],"previous_flags":3,"platform_policy":"strict"},
     "ground_truth":{"label":"toxic","action":"escalate","severity":5,"rationale":"Doxxing pattern with prior violations. Facilitates harassment."},
     "score_rubric":HARD_RUBRIC},
    {"id":"scen_hard_25","tier":"hard",
     "state":{"text":"These people are ruining everything. You know what needs to happen.",
              "audio_transcript":None,"visual_tags":[],"previous_flags":12,"platform_policy":"strict"},
     "ground_truth":{"label":"toxic","action":"escalate","severity":5,"rationale":"Implied threat. 12 prior violations makes this critical."},
     "score_rubric":HARD_RUBRIC},
    {"id":"scen_hard_26","tier":"hard",
     "state":{"text":"BREAKING: Scientists discover eating tide pods cures all diseases. Big Pharma doesn't want you to know!",
              "audio_transcript":None,"visual_tags":["satire_account_verified","parody_label"],"previous_flags":0,"platform_policy":"lenient"},
     "ground_truth":{"label":"misleading","action":"warn","severity":2,"rationale":"Satire account with verified parody label on lenient platform."},
     "score_rubric":HARD_RUBRIC},
    {"id":"scen_hard_27","tier":"hard",
     "state":{"text":"Estos inmigrantes estan destruyendo nuestro pais. Hay que hacer algo.",
              "audio_transcript":None,"visual_tags":["translated_content","nationalist_imagery"],"previous_flags":2,"platform_policy":"strict"},
     "ground_truth":{"label":"toxic","action":"remove","severity":4,"rationale":"Xenophobic content in Spanish: immigrants destroying country, something must be done."},
     "score_rubric":HARD_RUBRIC},
    {"id":"scen_hard_28","tier":"hard",
     "state":{"text":"I need help. I can't take it anymore. Does anyone know what happens if you take a whole bottle of sleeping pills?",
              "audio_transcript":None,"visual_tags":[],"previous_flags":0,"platform_policy":"strict"},
     "ground_truth":{"label":"toxic","action":"escalate","severity":5,"rationale":"Crisis signal. Escalate to human review immediately."},
     "score_rubric":HARD_RUBRIC},
    {"id":"scen_hard_29","tier":"hard",
     "state":{"text":"Honest review after 6 months: This product changed my life. Lost 40 pounds, skin cleared up. DM me for my full routine!",
              "audio_transcript":None,"visual_tags":["before_after_image","affiliate_link_detected"],"previous_flags":0,"platform_policy":"moderate"},
     "ground_truth":{"label":"spam","action":"remove","severity":3,"rationale":"Affiliate marketing disguised as personal story."},
     "score_rubric":HARD_RUBRIC},
    {"id":"scen_hard_30","tier":"hard",
     "state":{"text":"I absolutely despise how this game is designed. The devs should be ashamed of themselves.",
              "audio_transcript":None,"visual_tags":["gaming_community","product_review"],"previous_flags":1,"platform_policy":"lenient"},
     "ground_truth":{"label":"safe","action":"allow","severity":1,"rationale":"Strong opinion on lenient platform. Valid product criticism."},
     "score_rubric":HARD_RUBRIC},
    # 5 easy shadowban
    {"id":"scen_easy_21","tier":"easy",
     "state":{"text":"Buy cheap followers! 1000 followers for $5! DM NOW! Limited offer!",
              "audio_transcript":None,"visual_tags":[],"previous_flags":4,"platform_policy":"moderate"},
     "ground_truth":{"label":"spam","action":"shadowban"},"score_rubric":EASY_RUBRIC},
    {"id":"scen_easy_22","tier":"easy",
     "state":{"text":"Work from home and earn $500/day! No experience needed! Click my bio link!",
              "audio_transcript":None,"visual_tags":[],"previous_flags":3,"platform_policy":"moderate"},
     "ground_truth":{"label":"spam","action":"shadowban"},"score_rubric":EASY_RUBRIC},
    {"id":"scen_easy_23","tier":"easy",
     "state":{"text":"FREE crypto giveaway! Send 0.1 BTC get 1 BTC back! Official promotion!",
              "audio_transcript":None,"visual_tags":[],"previous_flags":5,"platform_policy":"strict"},
     "ground_truth":{"label":"spam","action":"shadowban"},"score_rubric":EASY_RUBRIC},
    {"id":"scen_easy_24","tier":"easy",
     "state":{"text":"Lose 30 pounds in 30 days with our miracle pill! Doctor approved! No diet needed!",
              "audio_transcript":None,"visual_tags":[],"previous_flags":3,"platform_policy":"moderate"},
     "ground_truth":{"label":"spam","action":"shadowban"},"score_rubric":EASY_RUBRIC},
    {"id":"scen_easy_25","tier":"easy",
     "state":{"text":"HOT SINGLES IN YOUR AREA! Click now! Real people waiting! Free tonight only!",
              "audio_transcript":None,"visual_tags":[],"previous_flags":4,"platform_policy":"strict"},
     "ground_truth":{"label":"spam","action":"shadowban"},"score_rubric":EASY_RUBRIC},
]

data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
existing = {s["id"] for s in data}
added = 0
for s in NEW:
    if s["id"] not in existing:
        data.append(s)
        added += 1

JSON_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Added {added}. Total: {len(data)}")
