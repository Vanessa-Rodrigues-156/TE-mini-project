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

### 3. Set your Gemini API key (for step 05 only)
```bash
export GEMINI_API_KEY=your_key_here
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

# Step 5 (OPTIONAL): Use GEMINI API to label unlabeled/ambiguous cases
python 05_llm_label.py

# Step 6: Explore and validate the dataset
python 06_explore_dataset.py

# Step 7: Train structured model (Random Forest / XGBoost)
# Run as notebook: jupyter notebook 07_train_structured_model.ipynb
# Or as script: python 07_train_structured_model.py
# Produces: best_model.pkl + metadata.pkl (in compiled_dataset/models/)
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
| `models/best_model.pkl` | **Trained classification model** (Step 7) |
| `models/metadata.pkl` | Model metadata: encoders, features, metrics (Step 7) |

---

## Label Schema

| Label | Meaning |
|-------|---------|
| 0 | NOT suitable for ADR or ODR |
| 1 | ADR suitable (arbitration / mediation / conciliation / Lok Adalat) |
| 2 | ADR + ODR suitable (can be resolved online) |
| -1 | Unknown / needs manual or LLM review |

Additional target columns kept in the dataset:
- `adr_target` (binary: -1/0/1)
- `odr_target` (binary: -1/0/1)

`final_label` is still provided for convenience as a multiclass target,
but `adr_target` and `odr_target` let you train independent models.

---

## HC/SC Text Labeling Notes

High Court and Supreme Court text labels are intentionally conservative:
- Keyword matches are boundary-aware (to reduce substring false positives)
- Negated contexts (for example, "arbitration rejected") are marked ambiguous
- Mixed ADR and non-ADR signals are marked ambiguous

Ambiguous cases are assigned `-1` and routed to `needs_llm_labeling.parquet`
for step 05.

---

## Data Validation Checks

After schema standardization, the pipeline validates:
- Required unified columns are present
- Label ranges are valid (`final_label` in `{-1,0,1,2}`; binary labels in `{-1,0,1}`)
- `case_id` is unique within each `source`
- Label consistency (`final_label=2` implies `odr_label=1`, etc.)

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
After training the structured model (Step 7), proceed to:
1. **Text-based model**: Fine-tune LegalBERT on case descriptions (Step 8)
2. **Ensemble**: Combine structured + text models for best accuracy (Step 9)
3. **Deployment**: Create API for real-time predictions (Step 10)

For detailed instructions on model training, see: `STEP_07_MODEL_TRAINING.md`

#For RAG_MODEL
Dataset — 750,442 court cases with ADR/ODR labels already tagged
Filtered — kept 63,084 cases (all ADR positives + 25K negatives)
Chunked — split texts into 100-word chunks for precise retrieval
Embedded — used all-mpnet-base-v2 to convert text to 768-dim vectors
FAISS index — stores all vectors for millisecond-speed search
Smart retrieval — keyword boosting to separate criminal vs civil cases
Generator — Flan-T5 generates answers from retrieved chunks
Accuracy — 85% on 20-query test, targeting 95%+
