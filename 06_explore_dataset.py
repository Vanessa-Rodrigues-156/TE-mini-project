"""
STEP 6: Dataset Explorer & Validation
=======================================
Run this after steps 01-04 to:
  - Understand the shape and quality of your compiled dataset
  - Check label distributions
  - Spot missing data
  - Preview sample rows per label class

Run: python 06_explore_dataset.py
"""

import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path("compiled_dataset")

LABEL_MAP = {
    -1: "Unknown (needs labeling)",
    0:  "NOT ADR/ODR eligible",
    1:  "ADR eligible only",
    2:  "ADR + ODR eligible",
}


def show_section(title: str):
    print(f"\n{'═' * 60}")
    print(f"  {title}")
    print('═' * 60)


def explore(path: Path, name: str):
    if not path.exists():
        print(f"  [SKIP] {path} not found.")
        return

    df = pd.read_parquet(path)
    show_section(f"{name}  ({len(df):,} rows)")

    print(f"\nColumns ({len(df.columns)}):")
    print("  " + ", ".join(df.columns.tolist()))

    print(f"\nData types:")
    print(df.dtypes.to_string())

    print(f"\nNull counts (top columns):")
    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0].sort_values(ascending=False)
    if len(nulls):
        print(nulls.head(15).to_string())
    else:
        print("  No nulls found.")

    if "final_label" in df.columns:
        print(f"\nLabel distribution:")
        counts = df["final_label"].map(LABEL_MAP).value_counts()
        total = len(df)
        for label, count in counts.items():
            print(f"  {label:<35} {count:>8,}  ({count/total*100:.1f}%)")

    if "source" in df.columns:
        print(f"\nSource distribution:")
        print(df["source"].value_counts().to_string())

    if "court_level" in df.columns:
        print(f"\nCourt level:")
        print(df["court_level"].value_counts().to_string())

    if "year" in df.columns:
        print(f"\nYear range: {df['year'].min()} – {df['year'].max()}")
        print(df["year"].value_counts().sort_index().to_string())

    if "act" in df.columns:
        print(f"\nTop 20 Acts:")
        print(df["act"].value_counts().head(20).to_string())

    # Sample rows per label
    if "final_label" in df.columns:
        for label_val in [0, 1, 2]:
            subset = df[df["final_label"] == label_val]
            if len(subset):
                print(f"\n── Sample rows: {LABEL_MAP[label_val]} ──────────────────")
                sample_cols = [c for c in ["title", "description", "act", "section",
                                            "case_type", "label_reason"] if c in df.columns]
                sample = subset.sample(min(3, len(subset)), random_state=1)
                for _, row in sample.iterrows():
                    for col in sample_cols:
                        val = str(row.get(col, ""))[:120]
                        if val and val != "nan":
                            print(f"  {col}: {val}")
                    print("  ---")


def main():
    print("=" * 60)
    print("Dataset Explorer & Validator")
    print("=" * 60)

    files = {
        "DDL District Court (raw)":    OUTPUT_DIR / "ddl_processed.parquet",
        "High Court Metadata":         OUTPUT_DIR / "hc_metadata.parquet",
        "Supreme Court Metadata":      OUTPUT_DIR / "sc_metadata.parquet",
        "DDL Labeled":                 OUTPUT_DIR / "ddl_labeled.parquet",
        "High Court Labeled":          OUTPUT_DIR / "hc_labeled.parquet",
        "Supreme Court Labeled":       OUTPUT_DIR / "sc_labeled.parquet",
        "TRAINING DATA (use this)":    OUTPUT_DIR / "training_data.parquet",
        "Needs LLM Labeling":          OUTPUT_DIR / "needs_llm_labeling.parquet",
    }

    for name, path in files.items():
        explore(path, name)

    # ── Training data quality summary ─────────────────────────────────────────
    training_path = OUTPUT_DIR / "training_data.parquet"
    if training_path.exists():
        df = pd.read_parquet(training_path)
        show_section("TRAINING DATA QUALITY REPORT")

        has_title = df["title"].notna().sum() if "title" in df.columns else 0
        has_desc  = df["description"].notna().sum() if "description" in df.columns else 0
        has_act   = df["act"].notna().sum() if "act" in df.columns else 0

        print(f"\n  Total training rows:       {len(df):,}")
        print(f"  Rows with title text:      {has_title:,}  ({has_title/len(df)*100:.1f}%)")
        print(f"  Rows with description:     {has_desc:,}  ({has_desc/len(df)*100:.1f}%)")
        print(f"  Rows with act name:        {has_act:,}  ({has_act/len(df)*100:.1f}%)")

        # Class balance
        counts = df["final_label"].value_counts()
        minority = counts.min()
        majority = counts.max()
        imbalance = majority / minority if minority > 0 else float("inf")
        print(f"\n  Class imbalance ratio:     {imbalance:.1f}x")
        if imbalance > 10:
            print("  ⚠ High imbalance — consider oversampling minority class (SMOTE)")
        else:
            print("  ✓ Imbalance is manageable")

        print("\n  Recommendation:")
        if has_desc > 10000:
            print("  → You have enough text data for a BERT/LegalBERT classifier")
        elif has_act > 50000:
            print("  → Use structured features (act/section/type) for a baseline RF/XGB model")
        print("  → Combine both for best performance")

    print("\n\nAll done ✓  → Next step: run 07_train_model.py")


if __name__ == "__main__":
    main()
