import json
import os
import sys
from pathlib import Path
from groq import Groq
from benchmark_pipeline_groq import generate_report

SCRIPT_DIR = Path(__file__).parent
RESULTS_PATH   = SCRIPT_DIR / "benchmark_results_groq.json"
REPORT_PATH    = SCRIPT_DIR / "benchmark_report_groq.md"

def main():
    groq_api_key = os.environ.get("GROQ_API_KEY")
    client = Groq(api_key=groq_api_key)
    results = json.loads(RESULTS_PATH.read_text())
    
    report_md = generate_report(results, client)
    REPORT_PATH.write_text(report_md, encoding="utf-8")
    print(f"  ✓ Report saved → {REPORT_PATH.name}")

if __name__ == "__main__":
    main()
