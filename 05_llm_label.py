"""
STEP 5: LLM-Assisted Labeling (Optional but Recommended)
==========================================================
For cases where rule-based labeling returned -1 (unknown),
we use Claude to label them from the case title/description.

This is OPTIONAL. Run this only if:
  - You have needs_llm_labeling.parquet from step 04
  - You want to expand your training data beyond rule-based labels
  - You want to validate the rule-based labels on a sample

Cost estimate:
  - ~500 tokens per case (title + description + prompt)
  - At Gemini API pricing, 1000 cases ≈ affordable for a project
  - Recommend batching: do 500–2000 cases for a good validation set

Output: llm_labeled_sample.parquet
"""

import pandas as pd
from google import generativeai as genai
import json
import os
import time
from pathlib import Path

OUTPUT_DIR = Path("compiled_dataset")
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))  # uses GEMINI_API_KEY env var
model = genai.GenerativeModel("gemini-1.5-flash")

LABELING_PROMPT = """You are a legal expert specializing in Indian dispute resolution law.

Given the following case information from an Indian court, determine if this case is suitable for:
1. ADR (Alternative Dispute Resolution) — includes arbitration, mediation, conciliation, Lok Adalat
2. ODR (Online Dispute Resolution) — a subset of ADR conducted digitally/online

Indian law context:
- CPC Section 89: Civil disputes can be referred to ADR
- Arbitration & Conciliation Act 1996: governs arbitrable disputes
- Non-compoundable criminal offences (serious crimes) CANNOT go to ADR
- Constitutional matters (fundamental rights, writ petitions) CANNOT go to ADR
- Consumer, motor accident, cheque bounce, small civil disputes — highly ADR/ODR suitable

Case information:
Court level: {court_level}
Case type: {case_type}
Act/Section: {act}
Title: {title}
Description: {description}
Disposal nature: {disposal_nature}

Respond ONLY with a JSON object, no other text:
{{
  "adr_suitable": true/false,
  "odr_suitable": true/false,
  "confidence": "high/medium/low",
  "reasoning": "one sentence explanation"
}}
"""


def label_case_with_llm(row: pd.Series) -> dict:
    prompt = LABELING_PROMPT.format(
        court_level=row.get("court_level", "Unknown"),
        case_type=row.get("case_type", "Unknown"),
        act=row.get("act", "Unknown"),
        title=str(row.get("title", ""))[:300],       # truncate
        description=str(row.get("description", ""))[:500],
        disposal_nature=row.get("disposal_nature", "Unknown"),
    )

    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        # Strip any markdown code fences
        text = text.replace("```json", "").replace("```", "").strip()
        result = json.loads(text)
        return {
            "adr_label": 1 if result.get("adr_suitable") else 0,
            "odr_label": 1 if result.get("odr_suitable") else 0,
            "final_label": 2 if result.get("odr_suitable") else (1 if result.get("adr_suitable") else 0),
            "label_reason": f"LLM ({result.get('confidence', '?')}): {result.get('reasoning', '')}",
            "llm_confidence": result.get("confidence", "unknown"),
        }
    except Exception as e:
        return {
            "adr_label": -1, "odr_label": -1, "final_label": -1,
            "label_reason": f"LLM error: {e}", "llm_confidence": "error"
        }


def main():
    print("=" * 60)
    print("LLM-Assisted Labeling (Optional Step)")
    print("=" * 60)

    unlabeled_path = OUTPUT_DIR / "needs_llm_labeling.parquet"
    if not unlabeled_path.exists():
        print(f"\n[ERROR] {unlabeled_path} not found. Run 04_label_adr.py first.")
        return

    df = pd.read_parquet(unlabeled_path)
    print(f"\nUnlabeled cases available: {len(df):,}")

    # ── Sample to label ───────────────────────────────────────────────────────
    # For a student project, 500–1000 LLM-labeled cases is sufficient as a
    # validation/augmentation set. Set SAMPLE_SIZE based on your API budget.
    SAMPLE_SIZE = 500
    sample = df.sample(min(SAMPLE_SIZE, len(df)), random_state=42).copy()
    print(f"Labeling {len(sample)} sampled cases with Gemini API ...")
    print("(This will make API calls — check your usage at Google AI Studio)\n")

    results = []
    for i, (idx, row) in enumerate(sample.iterrows()):
        if i % 50 == 0:
            print(f"  Progress: {i}/{len(sample)} ...")
        result = label_case_with_llm(row)
        results.append(result)
        time.sleep(0.3)  # rate limit safety

    results_df = pd.DataFrame(results, index=sample.index)
    # Update the sample with LLM labels
    for col in ["adr_label", "odr_label", "final_label", "label_reason", "llm_confidence"]:
        sample[col] = results_df[col]

    sample = sample[sample["final_label"] != -1]  # drop LLM errors

    out_path = OUTPUT_DIR / "llm_labeled_sample.parquet"
    sample.to_parquet(out_path, index=False)
    print(f"\nSaved → {out_path} ({len(sample):,} LLM-labeled rows)")

    print("\n── LLM Label distribution ──────────────────────────────")
    label_map = {0: "NOT eligible", 1: "ADR eligible", 2: "ADR+ODR eligible"}
    print(sample["final_label"].map(label_map).value_counts())

    print("\n── Confidence breakdown ────────────────────────────────")
    print(sample["llm_confidence"].value_counts())

    # ── Merge LLM labels back into training_data ──────────────────────────────
    training_path = OUTPUT_DIR / "training_data.parquet"
    if training_path.exists():
        print("\nMerging LLM labels into training_data.parquet ...")
        train = pd.read_parquet(training_path)
        combined = pd.concat([train, sample], ignore_index=True)
        combined.to_parquet(training_path, index=False)
        print(f"Updated training_data.parquet → {len(combined):,} rows total")

    print("\nDone ✓")


if __name__ == "__main__":
    main()
