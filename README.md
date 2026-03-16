# ADR/ODR Suitability Classifier — Dataset Setup

## Project Goal
Build a classification model that takes a legal case description and determines
whether the case is suitable for ADR (Alternative Dispute Resolution) or ODR
(Online Dispute Resolution).

---

## Data Sources

| Source | Court Level | Format | Size | Access |
|--------|-------------|--------|------|--------|
| Development Data Lab | District & Sessions Courts | .dta (Stata) | ~81M cases, 2010–2018 | Manual download (Dropbox) |
| indian-high-court-judgments S3 | 25 High Courts | Parquet | ~10M+ judgments | Public S3, no auth needed |
| indian-supreme-court-judgments S3 | Supreme Court | Parquet | ~300K judgments (1950–2025) | Public S3, no auth needed |

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download DDL District Court Data
- Go to: https://www.devdatalab.org/judicial-data
- Click the Dropbox download link
- Download the full folder OR just the `cases/` subfolder + `acts_sections.dta` + `keys/`
- Place it at: `ddl_data/` (relative to this folder)

Your `ddl_data/` structure should look like:
```
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
```

### 3. Set your Anthropic API key (for step 05 only)
```bash
export ANTHROPIC_API_KEY=your_key_here
```

---

## Run Order

```bash
# Step 1: Process DDL district court .dta files → ddl_processed.parquet
python 01_load_ddl.py

# Step 2: Fetch High Court parquet metadata from public S3 → hc_metadata.parquet
python 02_fetch_high_court.py

# Step 3: Fetch Supreme Court parquet metadata from public S3 → sc_metadata.parquet
python 03_fetch_supreme_court.py

# Step 4: Apply rule-based ADR/ODR labels to all three datasets
python 04_label_adr.py
# Produces: training_data.parquet + needs_llm_labeling.parquet

# Step 5 (OPTIONAL): Use Claude API to label unlabeled/ambiguous cases
python 05_llm_label.py

# Step 6: Explore and validate the dataset
python 06_explore_dataset.py
```

---

## Output Files (in `compiled_dataset/`)

| File | Description |
|------|-------------|
| `ddl_processed.parquet` | Cleaned DDL district court data |
| `hc_metadata.parquet` | High court judgment metadata |
| `sc_metadata.parquet` | Supreme court judgment metadata |
| `training_data.parquet` | **Main training dataset** — labeled, all sources |
| `needs_llm_labeling.parquet` | Cases with ambiguous/unknown labels |
| `llm_labeled_sample.parquet` | LLM-labeled sample (if step 05 was run) |

---

## Label Schema

| Label | Meaning |
|-------|---------|
| 0 | NOT suitable for ADR or ODR |
| 1 | ADR suitable (arbitration / mediation / conciliation / Lok Adalat) |
| 2 | ADR + ODR suitable (can be resolved online) |
| -1 | Unknown / needs manual or LLM review |

---

## ADR Eligibility Rules (Indian Law Basis)

**Eligible (Label 1 or 2):**
- Civil disputes → CPC Section 89 mandates ADR referral
- Motor Vehicles Act cases → Lok Adalat / MACT
- Consumer disputes → Consumer forum, Lok Adalat, ODR
- Cheque bounce (NI Act 138) → Lok Adalat eligible, ODR-friendly
- Family disputes (HMA, maintenance) → Court-annexed mediation
- Commercial disputes → Commercial Courts Act: pre-institution mediation mandatory
- Arbitration & Conciliation Act cases → Already arbitrable
- Labour disputes → Conciliation under Industrial Disputes Act
- Insurance claims → Ombudsman, ODR-friendly

**Not Eligible (Label 0):**
- Non-compoundable criminal offences (IPC serious crimes)
- POCSO, NDPS, UAPA, Arms Act cases
- Constitutional matters / writ petitions
- Election disputes
- Contempt of court
- Revenue / tax matters

---

## Next Step
After exploring the dataset, proceed to model training:
- **Structured model**: Random Forest / XGBoost on act_name + case_type features
- **Text model**: Fine-tune LegalBERT on title + description text
- **Combined**: Ensemble of both for best accuracy
