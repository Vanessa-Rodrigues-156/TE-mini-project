"""
STEP 2: Fetch High Court Judgment Metadata from Public S3
==========================================================
Source: s3://indian-high-court-judgments  (public, no credentials needed)
What we pull: metadata/parquet files — title, description, judge,
              date_of_registration, decision_date, disposal_nature, court_name
              (NOT the full PDFs — those are huge)

25 High Courts available. Court codes:
    9_13  → Allahabad HC
    27_1  → Bombay HC
    19_16 → Calcutta HC
    18_6  → Delhi HC
    36_29 → Telangana HC
    28_2  → Madras HC
    22_18 → Kerala HC
    7_26  → Gujarat HC
    24_17 → Himachal Pradesh HC
    2_5   → Chhattisgarh HC
    1_12  → Andhra Pradesh HC
    20_7  → Gauhati HC
    29_3  → Jharkhand HC
    32_4  → Punjab & Haryana HC
    23_23 → Patna HC
    14_25 → Manipur HC
    17_21 → Madhya Pradesh HC
    21_11 → Karnataka HC
    3_22  → Rajasthan HC
    8_9   → J&K HC
    11_24 → Sikkim HC
    16_20 → Orissa HC
    5_15  → Uttarakhand HC
    33_10 → Tripura HC
    10_8  → Meghalaya HC

Strategy:
- Use pandas to read parquet directly from the S3 URL (public, no auth needed)
- Focus on a manageable subset: Bombay HC + Delhi HC + Allahabad HC, 2015-2024
- Filter for cases with "description" text which we can use for NLP labeling

Output: hc_metadata.parquet
"""

import pandas as pd
from pathlib import Path
import time

OUTPUT_DIR = Path("compiled_dataset")
OUTPUT_DIR.mkdir(exist_ok=True)

S3_BASE = "https://indian-high-court-judgments.s3.ap-south-1.amazonaws.com"

# ── CONFIG: which courts and years to fetch ───────────────────────────────────
# Start focused — you can always expand later.
# Bombay, Delhi, Allahabad cover Maharashtra, Delhi, UP — large volumes.
TARGET_COURTS = {
    "27_1": "Bombay HC",
    "18_6": "Delhi HC",
    "9_13": "Allahabad HC",
    "21_11": "Karnataka HC",
    "28_2": "Madras HC",
}
TARGET_YEARS = list(range(2015, 2025))  # 2015–2024


def fetch_parquet(court_code: str, year: int) -> pd.DataFrame:
    """
    Download one parquet shard: metadata for court+year.
    URL pattern: {S3_BASE}/metadata/parquet/year={YEAR}/court={COURT}/metadata.parquet
    Returns empty DataFrame if not available.
    """
    url = f"{S3_BASE}/metadata/parquet/year={year}/court={court_code}/metadata.parquet"
    try:
        df = pd.read_parquet(url)
        df["source_court_code"] = court_code
        df["source_year"] = year
        return df
    except Exception as e:
        # Parquet not available for this court/year combination — skip silently
        return pd.DataFrame()


def main():
    print("=" * 60)
    print("High Court S3 Metadata Fetcher")
    print("=" * 60)
    print(f"\nTarget courts: {list(TARGET_COURTS.values())}")
    print(f"Target years:  {TARGET_YEARS}")
    print(f"Total requests to attempt: {len(TARGET_COURTS) * len(TARGET_YEARS)}\n")

    all_frames = []
    total_fetched = 0

    for court_code, court_name in TARGET_COURTS.items():
        for year in TARGET_YEARS:
            print(f"  Fetching {court_name} ({court_code}) — {year} ...", end=" ", flush=True)
            df = fetch_parquet(court_code, year)
            if df.empty:
                print("no data")
            else:
                all_frames.append(df)
                total_fetched += len(df)
                print(f"{len(df):,} rows")
            time.sleep(0.2)  # be polite to the server

    if not all_frames:
        print("\n[ERROR] No data fetched. Check your internet connection.")
        return

    print(f"\n[Combining] {total_fetched:,} total rows across all courts/years ...")
    combined = pd.concat(all_frames, ignore_index=True)

    # ── Rename/standardise columns ────────────────────────────────────────────
    # Actual columns from this dataset (from the Athena schema):
    # court_code, title, description, judge, pdf_link, cnr,
    # date_of_registration, decision_date, disposal_nature, court_name
    print(f"Columns found: {list(combined.columns)}")

    # Parse dates
    for datecol in ["decision_date", "date_of_registration"]:
        if datecol in combined.columns:
            combined[datecol] = pd.to_datetime(combined[datecol], errors="coerce")

    # Add a court-level label for clarity
    combined["court_name_clean"] = combined["source_court_code"].map(TARGET_COURTS)

    # Keep only rows where description is not null (needed for NLP)
    has_desc = combined["description"].notna() if "description" in combined.columns else pd.Series([True] * len(combined))
    combined_filtered = combined[has_desc].copy()
    print(f"Rows with description text: {len(combined_filtered):,} / {len(combined):,}")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = OUTPUT_DIR / "hc_metadata.parquet"
    combined_filtered.to_parquet(out_path, index=False)
    print(f"\nSaved → {out_path}")
    print(f"File size: {out_path.stat().st_size / 1e6:.1f} MB")

    # Quick summary
    print("\n── Summary ─────────────────────────────────────────")
    print("\nRows per court:")
    print(combined_filtered["court_name_clean"].value_counts())

    if "disposal_nature" in combined_filtered.columns:
        print("\nTop disposal natures:")
        print(combined_filtered["disposal_nature"].value_counts().head(15))

    print("\nDone ✓")


if __name__ == "__main__":
    main()
