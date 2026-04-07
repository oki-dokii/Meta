import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
RESULTS_PATH   = SCRIPT_DIR / "benchmark_results_groq.json"
REPORT_PATH    = SCRIPT_DIR / "benchmark_report_groq.md"

def generate_local_report(results: list[dict]) -> str:
    md = ["# Content Moderation Benchmark Report\n"]
    
    # Exec Summary
    raw_rewards = [r["reward"] for r in results]
    adj_rewards = [r.get("override_reward", r["reward"]) for r in results]
    mean_raw = sum(raw_rewards) / len(raw_rewards)
    mean_adj = sum(adj_rewards) / len(adj_rewards)
    
    md.append("## 1. Executive Summary")
    md.append(f"- **Total Scenarios**: {len(results)}")
    md.append(f"- **Mean Raw Reward**: {mean_raw:.3f}")
    md.append(f"- **Mean Adjusted Reward**: {mean_adj:.3f}\n")
    
    # Per-tier
    md.append("## 2. Per-tier Performance")
    md.append("| Tier | Count | Mean Raw | Mean Adj | % Perfect |")
    md.append("|---|---|---|---|---|")
    for tier in ["easy", "medium", "hard"]:
        tier_res = [r for r in results if r["tier"] == tier]
        if not tier_res: continue
        t_raw = [r["reward"] for r in tier_res]
        t_adj = [r.get("override_reward", r["reward"]) for r in tier_res]
        perfect = sum(1 for r in t_raw if r >= 0.99)
        md.append(f"| {tier.capitalize()} | {len(tier_res)} | {sum(t_raw)/len(t_raw):.3f} | {sum(t_adj)/len(t_adj):.3f} | {perfect/len(tier_res)*100:.1f}% |")
    md.append("\n")
    
    # Accuracy
    md.append("## 3. Label & Action Accuracy")
    label_match = sum(1 for r in results if r["agent_decision"].get("label") == r["ground_truth"].get("label"))
    action_match = sum(1 for r in results if r["agent_decision"].get("action") == r["ground_truth"].get("action"))
    md.append(f"- **Label Accuracy**: {label_match/len(results)*100:.1f}%")
    md.append(f"- **Action Accuracy**: {action_match/len(results)*100:.1f}%\n")
    
    # Appendix
    md.append("## Appendix: Full Score Table")
    md.append("| ID | Tier | Raw | Adj | Label | Action |")
    md.append("|---|---|---|---|---|---|")
    for r in results:
        l_ok = "✅" if r["agent_decision"].get("label") == r["ground_truth"].get("label") else "❌"
        a_ok = "✅" if r["agent_decision"].get("action") == r["ground_truth"].get("action") else "❌"
        md.append(f"| {r['scenario_id']} | {r['tier']} | {r['reward']:.2f} | {r.get('override_reward', r['reward']):.2f} | {l_ok} | {a_ok} |")
        
    return "\n".join(md)

def main():
    if not RESULTS_PATH.exists():
        sys.exit("Results file not found. Run pipeline step 1 & 2 first.")
    
    results = json.loads(RESULTS_PATH.read_text())
    report_md = generate_local_report(results)
    REPORT_PATH.write_text(report_md, encoding="utf-8")
    print(f"  ✓ Local Report saved → {REPORT_PATH.name}")

if __name__ == "__main__":
    main()
