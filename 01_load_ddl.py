"""
STEP 1: Load & Process DDL District Court Dataset
=================================================
Source: Development Data Lab (devdatalab.org/judicial-data)
Format: .dta (Stata) files, one per year (2010-2018)
Coverage: ~81 million cases, India's District & Sessions Courts

Folder structure expected (from your Dropbox download):
    ddl_data/
        cases/
            cases_2010.dta
            cases_2011.dta
            ...
            cases_2018.dta
        acts_sections.dta
        keys/
            act_key.dta
            section_key.dta
            disp_name_key.dta
            type_name_key.dta
            purpose_name_key.dta
            cases_state_key.dta
            cases_district_key.dta
            cases_court_key.dta

Output: ddl_processed.parquet  (cleaned, merged, label-ready)
"""

import pandas as pd
import pyreadstat
import os
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────
DDL_ROOT = Path("ddl_data")          # ← change to your actual path
OUTPUT_DIR = Path("compiled_dataset")
OUTPUT_DIR.mkdir(exist_ok=True)

# Which years to load. Start with 2015-2018 to keep it manageable (~30M rows).
# Once pipeline works, expand to 2010-2018.
YEARS = [2015, 2016, 2017, 2018]

# ── HELPER: read a .dta file into a pandas DataFrame ─────────────────────────
def read_dta(path: Path) -> pd.DataFrame:
    """Read a Stata .dta file. Returns empty DataFrame if file missing."""
    if not path.exists():
        print(f"  [WARN] File not found: {path}")
        return pd.DataFrame()
    df, _ = pyreadstat.read_dta(str(path))
    return df


# ── STEP 1A: Load lookup/key tables ──────────────────────────────────────────
def load_keys():
    keys = {}

    # act_key: act (int) → act_s (string name of act)
    keys["act"] = read_dta(DDL_ROOT / "keys" / "act_key.dta")

    # section_key: section (int) → section_s (string name of section)
    keys["section"] = read_dta(DDL_ROOT / "keys" / "section_key.dta")

    # disp_name_key: disp_name + year → disp_name_s (disposition string)
    keys["disp_name"] = read_dta(DDL_ROOT / "keys" / "disp_name_key.dta")

    # type_name_key: type_name + year → type_name_s (case type string)
    keys["type_name"] = read_dta(DDL_ROOT / "keys" / "type_name_key.dta")

    # purpose_name_key: purpose_name + year → purpose_name_s
    keys["purpose_name"] = read_dta(DDL_ROOT / "keys" / "purpose_name_key.dta")

    # state / district / court name keys
    keys["state"] = read_dta(DDL_ROOT / "keys" / "cases_state_key.dta")
    keys["district"] = read_dta(DDL_ROOT / "keys" / "cases_district_key.dta")
    keys["court"] = read_dta(DDL_ROOT / "keys" / "cases_court_key.dta")

    return keys


# ── STEP 1B: Load acts_sections table (one row per act+section per case) ─────
def load_acts_sections():
    df = read_dta(DDL_ROOT / "acts_sections.dta")
    if df.empty:
        return df
    # Keep only the most relevant columns
    keep = [c for c in ["ddl_case_id", "act", "section", "bailable_ipc",
                         "criminal"] if c in df.columns]
    return df[keep]


# ── STEP 1C: Load and merge one year of case data ────────────────────────────
def load_year(year: int, acts_df: pd.DataFrame, keys: dict) -> pd.DataFrame:
    print(f"\n  Loading cases_{year}.dta ...")
    df = read_dta(DDL_ROOT / "cases" / f"cases_{year}.dta")
    if df.empty:
        return df

    df["year"] = year

    # 1. Merge act/section details
    if not acts_df.empty:
        df = df.merge(acts_df, on="ddl_case_id", how="left")

    # 2. Merge act string name
    if not keys["act"].empty and "act" in df.columns:
        df = df.merge(keys["act"], on="act", how="left", suffixes=("", "_key"))

    # 3. Merge section string name
    if not keys["section"].empty and "section" in df.columns:
        df = df.merge(keys["section"], on="section", how="left", suffixes=("", "_key"))

    # 4. Merge disposition name
    if not keys["disp_name"].empty and "disp_name" in df.columns:
        merge_cols = [c for c in ["disp_name", "year"] if c in keys["disp_name"].columns]
        df = df.merge(keys["disp_name"][merge_cols + ["disp_name_s"]],
                      on=merge_cols, how="left")

    # 5. Merge case type name
    if not keys["type_name"].empty and "type_name" in df.columns:
        merge_cols = [c for c in ["type_name", "year"] if c in keys["type_name"].columns]
        df = df.merge(keys["type_name"][merge_cols + ["type_name_s"]],
                      on=merge_cols, how="left")

    # 6. Merge state name
    if not keys["state"].empty and "state_code" in df.columns:
        state_cols = [c for c in ["state_code", "year"] if c in keys["state"].columns]
        name_col = [c for c in keys["state"].columns if "state_name" in c or "_s" in c]
        if name_col:
            df = df.merge(keys["state"][state_cols + name_col[:1]],
                          on=state_cols, how="left")

    # 7. Merge district name
    if not keys["district"].empty and "dist_code" in df.columns:
        dist_cols = [c for c in ["state_code", "dist_code", "year"]
                     if c in keys["district"].columns]
        name_col = [c for c in keys["district"].columns if "dist_name" in c or "_s" in c]
        if name_col:
            df = df.merge(keys["district"][dist_cols + name_col[:1]],
                          on=dist_cols, how="left")

    print(f"    → {len(df):,} rows loaded for {year}")
    return df


# ── STEP 1D: Select final columns relevant for ADR labeling ──────────────────
FINAL_COLS = [
    # Identifiers
    "ddl_case_id", "year", "state_code", "dist_code",
    # Case details (structured)
    "type_name_s",      # case type (civil, criminal, etc.)
    "act_s",            # act name (Indian Penal Code, Motor Vehicles Act, etc.)
    "section_s",        # section name
    "disp_name_s",      # how the case was disposed/decided
    # Flags
    "criminal",         # 1 = criminal case, 0 = civil
    "bailable_ipc",     # 1 = bailable offence (relevant for ADR)
    # Geography
    "state_name" if True else "state_code",   # resolved below
    # Dates
    "date_of_filing", "date_of_decision",
    # Petitioner/respondent (anonymized in public data)
    "pet_name", "res_name",
]


def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    available = [c for c in FINAL_COLS if c in df.columns]
    # Also keep any *_s (string) columns from key merges not listed above
    string_cols = [c for c in df.columns if c.endswith("_s") and c not in available]
    all_keep = list(dict.fromkeys(available + string_cols))  # dedup, preserve order
    return df[all_keep]


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("DDL District Court Data Loader")
    print("=" * 60)

    print("\n[1] Loading key/lookup tables ...")
    keys = load_keys()
    print("    Keys loaded:", {k: len(v) for k, v in keys.items() if not v.empty})

    print("\n[2] Loading acts_sections table ...")
    acts_df = load_acts_sections()
    print(f"    Acts/sections rows: {len(acts_df):,}")

    all_years = []
    print("\n[3] Loading case files by year ...")
    for year in YEARS:
        df = load_year(year, acts_df, keys)
        if not df.empty:
            df = select_columns(df)
            all_years.append(df)

    print("\n[4] Concatenating all years ...")
    combined = pd.concat(all_years, ignore_index=True)
    print(f"    Total rows: {len(combined):,}")
    print(f"    Columns: {list(combined.columns)}")

    print("\n[5] Basic cleanup ...")
    # Standardise string columns: strip whitespace, lowercase
    str_cols = combined.select_dtypes("object").columns
    for col in str_cols:
        combined[col] = combined[col].str.strip()

    # Parse dates
    for datecol in ["date_of_filing", "date_of_decision"]:
        if datecol in combined.columns:
            combined[datecol] = pd.to_datetime(combined[datecol], errors="coerce")

    print("\n[6] Saving to parquet ...")
    out_path = OUTPUT_DIR / "ddl_processed.parquet"
    combined.to_parquet(out_path, index=False)
    print(f"    Saved → {out_path}")
    print(f"    File size: {out_path.stat().st_size / 1e6:.1f} MB")

    # Quick summary
    print("\n── Summary ──────────────────────────────────────────")
    if "criminal" in combined.columns:
        print(combined["criminal"].value_counts().rename({0: "Civil", 1: "Criminal"}))
    if "act_s" in combined.columns:
        print("\nTop 20 Acts:")
        print(combined["act_s"].value_counts().head(20))

    print("\nDone ✓")


if __name__ == "__main__":
    main()
