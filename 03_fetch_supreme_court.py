"""
STEP 3: Fetch Supreme Court Judgment Metadata from Public S3
=============================================================
Source: s3://indian-supreme-court-judgments  (public, no credentials needed)
Bucket region: us-east-1

Structure:
    metadata/
        parquet/
            year=YYYY/
                metadata.parquet

Columns available:
    title, description, judge, pdf_link, date_of_registration,
    decision_date, disposal_nature, bench_strength, bench_type

Why include Supreme Court data?
- SC judgments often clarify whether a category of dispute is ADR-eligible
- SC orders under Arbitration & Conciliation Act give strong labeling signal
- Smaller volume (~5k-20k cases/year) so easy to download fully

Output: sc_metadata.parquet
"""

import pandas as pd
from pathlib import Path
import time

OUTPUT_DIR = Path("compiled_dataset")
OUTPUT_DIR.mkdir(exist_ok=True)

SC_S3_BASE = "https://indian-supreme-court-judgments.s3.amazonaws.com"

# SC data goes back to 1950, but for ADR-relevant cases focus on 2000+
# (Arbitration & Conciliation Act 1996 came into force, so 2000-2024 is ideal)
TARGET_YEARS = list(range(2000, 2025))


def fetch_sc_parquet(year: int) -> pd.DataFrame:
    url = f"{SC_S3_BASE}/metadata/parquet/year={year}/metadata.parquet"
    try:
        df = pd.read_parquet(url)
        df["source_year"] = year
        df["court_level"] = "Supreme Court"
        return df
    except Exception as e:
        return pd.DataFrame()


def main():
    print("=" * 60)
    print("Supreme Court S3 Metadata Fetcher")
    print("=" * 60)
    print(f"\nFetching years: {TARGET_YEARS[0]}–{TARGET_YEARS[-1]}\n")

    all_frames = []
    total = 0

    for year in TARGET_YEARS:
        print(f"  Year {year} ...", end=" ", flush=True)
        df = fetch_sc_parquet(year)
        if df.empty:
            print("no data")
        else:
            all_frames.append(df)
            total += len(df)
            print(f"{len(df):,} rows")
        time.sleep(0.2)

    if not all_frames:
        print("\n[ERROR] No SC data fetched.")
        return

    print(f"\n[Combining] {total:,} total SC judgment rows ...")
    combined = pd.concat(all_frames, ignore_index=True)
    print(f"Columns: {list(combined.columns)}")

    # Parse dates
    for datecol in ["decision_date", "date_of_registration"]:
        if datecol in combined.columns:
            combined[datecol] = pd.to_datetime(combined[datecol], errors="coerce")

    # ── ADR-signal filter ─────────────────────────────────────────────────────
    # Tag cases that mention ADR-related keywords in title/description
    # These are high-confidence POSITIVE labels for ADR-eligible cases
    ADR_KEYWORDS = [
        "arbitration", "mediation", "conciliation", "lok adalat",
        "settlement", "negotiation", "adr", "odr",
        "section 89", "arbitration and conciliation",
    ]

    if "title" in combined.columns:
        combined["title_lower"] = combined["title"].str.lower().fillna("")
        combined["adr_keyword_in_title"] = combined["title_lower"].apply(
            lambda t: any(kw in t for kw in ADR_KEYWORDS)
        )
        print(f"\nCases with ADR keyword in title: {combined['adr_keyword_in_title'].sum():,}")

    if "description" in combined.columns:
        combined["desc_lower"] = combined["description"].str.lower().fillna("")
        combined["adr_keyword_in_desc"] = combined["desc_lower"].apply(
            lambda d: any(kw in d for kw in ADR_KEYWORDS)
        )
        # Drop temp lower columns
        combined.drop(columns=["title_lower", "desc_lower"], errors="ignore", inplace=True)

    out_path = OUTPUT_DIR / "sc_metadata.parquet"
    combined.to_parquet(out_path, index=False)
    print(f"\nSaved → {out_path}")
    print(f"File size: {out_path.stat().st_size / 1e6:.1f} MB")

    print("\n── Summary ─────────────────────────────────────────")
    print(f"Total rows: {len(combined):,}")
    print(f"\nRows per decade:")
    combined["decade"] = (combined["source_year"] // 10 * 10).astype(str) + "s"
    print(combined["decade"].value_counts().sort_index())

    print("\nDone ✓")


if __name__ == "__main__":
    main()
