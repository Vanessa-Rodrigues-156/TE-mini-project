"""
STEP 4: ADR/ODR Label Construction
====================================
This is the most critical step. None of the source datasets have ADR labels.
We build them using two approaches:

APPROACH A — Rule-based (from act/section codes in DDL data)
    Uses Indian law to determine ADR eligibility:
    - CPC Section 89: Civil disputes → ADR referral mandated
    - Arbitration & Conciliation Act 1996: arbitrable disputes
    - Motor Vehicles Act: Lok Adalat eligible
    - Consumer Protection Act: consumer forum / Lok Adalat eligible
    - Hindu Marriage Act / Family Courts: mediation eligible
    - Non-compoundable criminal offences → NOT eligible
    - Constitutional matters → NOT eligible

APPROACH B — Keyword-based (from title/description text in HC/SC data)
    Extracts ADR signal from free text using keyword rules.

Labels produced:
    0 = NOT ADR/ODR suitable
    1 = ADR suitable (any form: arbitration, mediation, Lok Adalat, etc.)
    2 = ODR suitable (subset of ADR — suitable for online/digital proceedings)

Output:
    compiled_dataset/ddl_labeled.parquet
    compiled_dataset/hc_labeled.parquet
    compiled_dataset/sc_labeled.parquet
    compiled_dataset/combined_labeled.parquet  ← main training dataset
"""

import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path("compiled_dataset")
OUTPUT_DIR.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# LABEL RULES — Indian Law Basis
# ═══════════════════════════════════════════════════════════════════════════════

# Acts where disputes are typically ADR-eligible
# Format: {act_name_fragment: (adr_label, odr_label, notes)}
# adr_label: 1=ADR suitable, 0=not suitable
# odr_label: 1=ODR suitable (subset of ADR), 0=not suitable for ODR

ADR_ELIGIBLE_ACTS = {
    # ── CLEARLY ADR/ODR ELIGIBLE ──────────────────────────────────────────
    "motor vehicles":       (1, 1, "Lok Adalat / MACT — high volume, routine, ODR-friendly"),
    "consumer protection":  (1, 1, "Consumer forum / Lok Adalat — often small claims, ODR-friendly"),
    "arbitration":          (1, 0, "Already in arbitration — ADR yes, ODR depends on complexity"),
    "negotiable instruments": (1, 1, "Cheque bounce — compoundable, Lok Adalat eligible, ODR-friendly"),
    "commercial courts":    (1, 0, "Pre-institution mediation mandatory under Commercial Courts Act"),
    "specific relief":      (1, 0, "Civil contract disputes — CPC S.89 ADR referral"),
    "transfer of property": (1, 0, "Property/rent disputes — mediation eligible"),
    "contract":             (1, 0, "Contract disputes — core arbitration territory"),
    "insurance":            (1, 1, "Insurance claims — ODR-friendly if quantum is main issue"),
    "recovery of debts":    (1, 0, "Debt recovery — mediation / Lok Adalat eligible"),
    "micro, small":         (1, 0, "MSME Act — mandatory conciliation"),
    "electricity":          (1, 0, "Electricity disputes — ombudsman / ADR"),
    "real estate":          (1, 0, "RERA — conciliation eligible"),
    "hindu marriage":       (1, 0, "Family matters — court-annexed mediation, not ODR"),
    "family courts":        (1, 0, "Family matters — mediation eligible"),
    "special marriage":     (1, 0, "Family matters — mediation eligible"),
    "maintenance":          (1, 0, "Family/maintenance — mediation eligible"),
    "partnership":          (1, 0, "Partnership disputes — arbitration eligible"),
    "companies act":        (1, 0, "Company disputes — NCLT conciliation"),
    "insolvency":           (1, 0, "IBC — settlement possible at pre-admission stage"),
    "labour":               (1, 0, "Labour disputes — conciliation under ID Act"),
    "industrial disputes":  (1, 0, "Conciliation mandatory under Industrial Disputes Act"),
    "workmen":              (1, 0, "Labour — conciliation eligible"),
    "employees":            (1, 0, "Labour — conciliation eligible"),
    "land acquisition":     (1, 0, "Compensation disputes — Lok Adalat eligible"),
    "public premises":      (1, 0, "Eviction — mediation eligible"),
    "rent control":         (1, 0, "Rent disputes — mediation eligible"),

    # ── NOT ADR ELIGIBLE ─────────────────────────────────────────────────
    "indian penal code":    (0, 0, "IPC — criminal, non-compoundable offences not ADR eligible"),
    "ipc":                  (0, 0, "IPC criminal"),
    "prevention of corruption": (0, 0, "Anti-corruption — public interest, not ADR"),
    "narcotic":             (0, 0, "NDPS — criminal, non-compoundable"),
    "pocso":                (0, 0, "Child protection — not ADR eligible"),
    "protection of children": (0, 0, "Child protection — not ADR eligible"),
    "terrorism":            (0, 0, "UAPA — not ADR eligible"),
    "unlawful activities":  (0, 0, "UAPA — not ADR eligible"),
    "arms act":             (0, 0, "Criminal — not ADR eligible"),
    "explosives":           (0, 0, "Criminal — not ADR eligible"),
    "scheduled castes":     (0, 0, "SC/ST Act — not ADR eligible (constitutional protection)"),
    "atrocities":           (0, 0, "SC/ST Act — not ADR eligible"),
    "constitution":         (0, 0, "Constitutional matters — not ADR eligible"),
    "fundamental rights":   (0, 0, "Writ jurisdiction — not ADR eligible"),
    "habeas corpus":        (0, 0, "Not ADR eligible"),
    "election":             (0, 0, "Election disputes — statutory process"),
    "contempt":             (0, 0, "Contempt — not ADR eligible"),
    "revenue":              (0, 0, "Revenue/tax — usually not ADR eligible"),
    "income tax":           (0, 0, "Tax — not ADR eligible"),
    "customs":              (0, 0, "Tax/customs — not ADR eligible"),
    "foreign exchange":     (0, 0, "FEMA — not ADR eligible"),
}

# Criminal cases: bailable offences are sometimes Lok Adalat eligible
# Non-bailable → not ADR eligible generally
BAILABLE_LABEL = (1, 0)    # ADR=yes (Lok Adalat), ODR=no
NON_BAILABLE_LABEL = (0, 0)

# Keyword triggers in text (for HC/SC data)
ADR_POSITIVE_KEYWORDS = [
    "arbitration", "mediation", "conciliation", "lok adalat",
    "settlement agreement", "section 89 cpc", "adr", "odr",
    "online dispute resolution", "negotiated settlement",
    "arbitral award", "arbitral tribunal", "mediator",
    "mutual consent", "amicable settlement",
    "pre-litigation", "pre institution mediation",
]
NON_ADR_KEYWORDS = [
    "murder", "rape", "dacoity", "terrorism", "pocso",
    "habeas corpus", "writ petition", "fundamental right",
    "election", "contempt of court", "ndps", "narcotic",
]
ODR_POSITIVE_KEYWORDS = [
    "online", "digital", "e-commerce", "electronic", "cyber",
    "internet", "odr", "online dispute", "cheque bounce",
    "motor accident", "consumer complaint", "insurance claim",
]


# ═══════════════════════════════════════════════════════════════════════════════
# LABELING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def label_from_act(act_s: str, section_s: str = None,
                   criminal: float = None, bailable: float = None) -> tuple:
    """
    Returns (adr_label, odr_label, reason) for a case based on act/section.
    """
    act = str(act_s).lower() if pd.notna(act_s) else ""
    section = str(section_s).lower() if pd.notna(section_s) else ""

    # Check act against rules
    for key, (adr, odr, reason) in ADR_ELIGIBLE_ACTS.items():
        if key in act or key in section:
            return (adr, odr, reason)

    # Fallback: use criminal/bailable flags
    if pd.notna(criminal):
        if int(criminal) == 0:
            # Civil case — CPC S.89 mandates ADR consideration
            return (1, 0, "Civil case — CPC S.89 ADR referral applicable")
        else:
            # Criminal case
            if pd.notna(bailable) and int(bailable) == 1:
                return BAILABLE_LABEL[0], BAILABLE_LABEL[1], "Bailable criminal — Lok Adalat eligible"
            else:
                return NON_BAILABLE_LABEL[0], NON_BAILABLE_LABEL[1], "Non-bailable criminal — not ADR eligible"

    # Unknown
    return (-1, -1, "Unknown — needs manual review")


def label_from_text(title: str = None, description: str = None) -> tuple:
    """
    Returns (adr_label, odr_label, reason) from free text.
    Uses keyword heuristics.
    """
    text = " ".join(filter(None, [
        str(title).lower() if pd.notna(title) else "",
        str(description).lower() if pd.notna(description) else "",
    ]))

    if not text.strip():
        return (-1, -1, "No text available")

    # Hard negative — non-ADR keywords trump everything
    for kw in NON_ADR_KEYWORDS:
        if kw in text:
            return (0, 0, f"Non-ADR keyword found: '{kw}'")

    # Positive ADR signal
    adr = 0
    odr = 0
    reasons = []

    for kw in ADR_POSITIVE_KEYWORDS:
        if kw in text:
            adr = 1
            reasons.append(f"ADR keyword: '{kw}'")
            break

    for kw in ODR_POSITIVE_KEYWORDS:
        if kw in text:
            odr = 1
            reasons.append(f"ODR keyword: '{kw}'")
            break

    if adr == 0 and odr == 0:
        return (-1, -1, "No ADR/ODR signal in text")

    return (adr, odr, "; ".join(reasons))


# ═══════════════════════════════════════════════════════════════════════════════
# LABEL EACH DATASET
# ═══════════════════════════════════════════════════════════════════════════════

def label_ddl(df: pd.DataFrame) -> pd.DataFrame:
    """Apply rule-based labels to DDL district court data."""
    print(f"  Labeling {len(df):,} DDL rows ...")

    results = df.apply(
        lambda r: label_from_act(
            r.get("act_s"), r.get("section_s"),
            r.get("criminal"), r.get("bailable_ipc")
        ),
        axis=1, result_type="expand"
    )
    results.columns = ["adr_label", "odr_label", "label_reason"]
    df = pd.concat([df, results], axis=1)

    # Final label: -1 = unknown, 0 = not eligible, 1 = ADR only, 2 = ADR+ODR
    df["final_label"] = df.apply(
        lambda r: -1 if r["adr_label"] == -1
                  else (2 if r["odr_label"] == 1 else r["adr_label"]),
        axis=1
    )
    df["source"] = "DDL_district_court"
    return df


def label_court_text(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Apply text-based labels to HC/SC data."""
    print(f"  Labeling {len(df):,} {source_name} rows ...")

    title_col = "title" if "title" in df.columns else None
    desc_col = "description" if "description" in df.columns else None

    results = df.apply(
        lambda r: label_from_text(
            r.get(title_col) if title_col else None,
            r.get(desc_col) if desc_col else None
        ),
        axis=1, result_type="expand"
    )
    results.columns = ["adr_label", "odr_label", "label_reason"]
    df = pd.concat([df, results], axis=1)

    df["final_label"] = df.apply(
        lambda r: -1 if r["adr_label"] == -1
                  else (2 if r["odr_label"] == 1 else r["adr_label"]),
        axis=1
    )
    df["source"] = source_name
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# STANDARDISE SCHEMA for combining all three sources
# ═══════════════════════════════════════════════════════════════════════════════

UNIFIED_COLS = [
    "source",
    "case_id",           # best available ID field
    "court_level",       # District / High Court / Supreme Court
    "court_name",        # human-readable court name
    "year",
    "case_type",         # type_name_s / case type
    "act",               # act_s / act name
    "section",           # section_s / section
    "title",             # title / pet_name vs res_name
    "description",       # description text
    "decision_date",
    "date_of_filing",
    "disposal_nature",   # how the case ended
    "is_criminal",       # 0=civil, 1=criminal
    "is_bailable",       # 1=bailable (criminal only)
    "state",
    "adr_label",
    "odr_label",
    "final_label",       # 0=no, 1=ADR only, 2=ADR+ODR, -1=unknown
    "label_reason",
]


def standardise(df: pd.DataFrame, court_level: str) -> pd.DataFrame:
    """Map source-specific columns to the unified schema."""
    out = pd.DataFrame()
    out["source"] = df.get("source", pd.Series([court_level] * len(df)))
    out["court_level"] = court_level

    # ID
    for id_col in ["ddl_case_id", "cnr", "case_id"]:
        if id_col in df.columns:
            out["case_id"] = df[id_col].astype(str)
            break
    else:
        out["case_id"] = range(len(df))

    # Court name
    for col in ["court_name_clean", "court_name", "state_name"]:
        if col in df.columns:
            out["court_name"] = df[col]
            break

    out["year"] = df.get("source_year", df.get("year", pd.Series([None] * len(df))))

    for src, tgt in [("type_name_s", "case_type"), ("act_s", "act"),
                     ("section_s", "section"), ("title", "title"),
                     ("description", "description"),
                     ("decision_date", "decision_date"),
                     ("date_of_filing", "date_of_filing"),
                     ("disp_name_s", "disposal_nature"),
                     ("disposal_nature", "disposal_nature"),
                     ("criminal", "is_criminal"),
                     ("bailable_ipc", "is_bailable"),
                     ("state_name", "state")]:
        if src in df.columns and tgt not in out.columns:
            out[tgt] = df[src]

    # Label columns
    for col in ["adr_label", "odr_label", "final_label", "label_reason"]:
        out[col] = df[col]

    return out


def main():
    print("=" * 60)
    print("ADR/ODR Label Constructor")
    print("=" * 60)

    all_labeled = []

    # ── DDL District Court ────────────────────────────────────────────────────
    ddl_path = OUTPUT_DIR / "ddl_processed.parquet"
    if ddl_path.exists():
        print("\n[1] Labeling DDL District Court data ...")
        ddl = pd.read_parquet(ddl_path)
        ddl_labeled = label_ddl(ddl)
        ddl_std = standardise(ddl_labeled, "District Court")
        ddl_std.to_parquet(OUTPUT_DIR / "ddl_labeled.parquet", index=False)
        print(f"    Saved ddl_labeled.parquet ({len(ddl_std):,} rows)")
        all_labeled.append(ddl_std)
    else:
        print(f"\n[1] SKIP — {ddl_path} not found. Run 01_load_ddl.py first.")

    # ── High Court ────────────────────────────────────────────────────────────
    hc_path = OUTPUT_DIR / "hc_metadata.parquet"
    if hc_path.exists():
        print("\n[2] Labeling High Court data ...")
        hc = pd.read_parquet(hc_path)
        hc_labeled = label_court_text(hc, "High Court")
        hc_std = standardise(hc_labeled, "High Court")
        hc_std.to_parquet(OUTPUT_DIR / "hc_labeled.parquet", index=False)
        print(f"    Saved hc_labeled.parquet ({len(hc_std):,} rows)")
        all_labeled.append(hc_std)
    else:
        print(f"\n[2] SKIP — {hc_path} not found. Run 02_fetch_high_court.py first.")

    # ── Supreme Court ─────────────────────────────────────────────────────────
    sc_path = OUTPUT_DIR / "sc_metadata.parquet"
    if sc_path.exists():
        print("\n[3] Labeling Supreme Court data ...")
        sc = pd.read_parquet(sc_path)
        sc_labeled = label_court_text(sc, "Supreme Court")
        sc_std = standardise(sc_labeled, "Supreme Court")
        sc_std.to_parquet(OUTPUT_DIR / "sc_labeled.parquet", index=False)
        print(f"    Saved sc_labeled.parquet ({len(sc_std):,} rows)")
        all_labeled.append(sc_std)
    else:
        print(f"\n[3] SKIP — {sc_path} not found. Run 03_fetch_supreme_court.py first.")

    # ── Combine ───────────────────────────────────────────────────────────────
    if all_labeled:
        print("\n[4] Combining all labeled data ...")
        combined = pd.concat(all_labeled, ignore_index=True)

        # Drop unknowns for training (or keep them for semi-supervised — your choice)
        known = combined[combined["final_label"] != -1].copy()
        unknown = combined[combined["final_label"] == -1].copy()

        print(f"\n    Total rows:   {len(combined):,}")
        print(f"    Labeled rows: {len(known):,}")
        print(f"    Unknown rows: {len(unknown):,} (need LLM labeling — see step 05)")
        print("\n    Label distribution (known rows):")
        label_map = {0: "NOT eligible", 1: "ADR eligible", 2: "ADR + ODR eligible"}
        print(known["final_label"].map(label_map).value_counts())

        combined.to_parquet(OUTPUT_DIR / "combined_labeled.parquet", index=False)
        known.to_parquet(OUTPUT_DIR / "training_data.parquet", index=False)
        unknown.to_parquet(OUTPUT_DIR / "needs_llm_labeling.parquet", index=False)

        print(f"\n    Saved combined_labeled.parquet  → {len(combined):,} rows")
        print(f"    Saved training_data.parquet     → {len(known):,} rows (use this for model)")
        print(f"    Saved needs_llm_labeling.parquet → {len(unknown):,} rows (optional LLM pass)")

    print("\nDone ✓")


if __name__ == "__main__":
    main()
