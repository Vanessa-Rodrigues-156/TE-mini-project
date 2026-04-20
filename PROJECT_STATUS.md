# ADR/ODR Project Status & Next Steps

## Current Status: ✅ Steps 1-5 Complete

You have successfully:
- ✅ **Step 1**: Loaded and processed DDL district court data
- ✅ **Step 2**: Fetched High Court judgment metadata
- ✅ **Step 3**: Fetched Supreme Court judgment metadata
- ✅ **Step 4**: Applied rule-based ADR/ODR labels
- ✅ **Step 5**: LLM-labeled ambiguous cases

**Output**: `compiled_dataset/training_data.parquet` (on your Google Drive)

---

## 🚀 Next: Step 6-7 (Now Available)

### Step 6: Explore Dataset (Optional but Recommended)
**Status**: ✅ Complete (Python script available)

Run to understand your data:
```bash
python 06_explore_dataset.py
# OR
jupyter notebook 06_explore_dataset.ipynb
```

This will show:
- Label distributions
- Missing values
- Feature statistics
- Data quality metrics

---

### Step 7: Train Structured Model (NEW - Just Created)
**Status**: 🎉 Ready to use!

Two ways to run:

#### Option A: Jupyter Notebook (Interactive)
```bash
jupyter notebook 07_train_structured_model.ipynb
```
Perfect for exploration and learning what each step does.

#### Option B: Python Script (Automated)
```bash
python 07_train_structured_model.py \
  --data-path ./compiled_dataset \
  --target adr_target
```
Perfect for production and automation.

**What it does:**
1. Loads training data
2. Prepares features (act_name, case_type, source, year, court_level)
3. Trains 4 models:
   - Random Forest (baseline)
   - XGBoost (baseline)
   - Random Forest + SMOTE (balanced)
   - XGBoost + SMOTE (balanced) ← Usually best
4. Compares performance
5. Saves best model to `compiled_dataset/models/best_model.pkl`

**Expected output:**
- Model accuracy: 75-85%
- F1 score: 0.65-0.80
- Training time: 5-10 minutes

---

## Future Steps (Coming Soon)

### Step 8: Text Model Training
**Status**: 📋 Planned

Train a LegalBERT model on case descriptions:
- Extract text embeddings
- Fine-tune transformer model
- Better capture of legal language

### Step 9: Ensemble Model
**Status**: 📋 Planned

Combine structured + text models:
- Use voting classifier
- Stack predictions
- Achieve 85%+ accuracy

### Step 10: Model Deployment
**Status**: 📋 Planned

Create production API:
- Flask/FastAPI endpoint
- Docker containerization
- Cloud deployment (GCP/AWS)

---

## File Structure

```
TE-mini-project/
├── 01_load_ddl.ipynb              # ✅ Done
├── 01_load_ddl.py
├── 02_fetch_high_court.ipynb       # ✅ Done
├── 02_fetch_high_court.py
├── 03_fetch_supreme_court.ipynb    # ✅ Done
├── 03_fetch_supreme_court.py
├── 04_label_adr.ipynb              # ✅ Done
├── 04_label_adr.py
├── 05_llm_label.ipynb              # ✅ Done
├── 05_llm_label.py
├── 06_explore_dataset.ipynb        # ✅ Available
├── 06_explore_dataset.py
├── 07_train_structured_model.ipynb # 🎉 NEW
├── 07_train_structured_model.py    # 🎉 NEW
├── README.md
├── QUICKSTART_MODEL_TRAINING.md    # 🎉 NEW (for Colab)
├── STEP_07_MODEL_TRAINING.md       # 🎉 NEW (detailed guide)
├── requirements.txt
└── compiled_dataset/               # On Google Drive
    ├── ddl_processed.parquet
    ├── hc_metadata.parquet
    ├── sc_metadata.parquet
    ├── training_data.parquet       # ← You have this
    ├── needs_llm_labeling.parquet
    └── models/                     # Will be created by Step 7
        ├── best_model.pkl
        └── metadata.pkl
```

---

## Quick Reference: What Each Model Does

### Random Forest
```
Pros: Faster, handles missing data, feature importance clear
Cons: Less powerful than boosting
Best for: Quick baseline, understanding feature importance
```

### XGBoost
```
Pros: High accuracy, handles both numerical and categorical data
Cons: Slower, more hyperparameters to tune
Best for: Production, when accuracy matters most
```

### SMOTE (Synthetic Minority Oversampling)
```
Pros: Handles class imbalance, better F1 scores
Cons: Slightly slower training
Best for: Imbalanced datasets (like legal data)
```

### Best Practice
**Start with**: XGBoost + SMOTE ← What we're using
**Then try**: Fine-tune hyperparameters if needed
**Then add**: Text features for better accuracy

---

## How to Run on Colab

### 1. Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 2. Install Requirements
```bash
!pip install -r requirements.txt
```

### 3. Navigate to Notebook
```python
%cd /content/drive/MyDrive/TE-mini-project
```

### 4. Run Training
Option A (Interactive):
```python
# Open and run: 07_train_structured_model.ipynb
# Make sure data_path = '/content/drive/MyDrive/compiled_dataset'
```

Option B (Script):
```bash
!python 07_train_structured_model.py \
  --data-path '/content/drive/MyDrive/compiled_dataset'
```

### 5. Download Results
```python
# After training completes
# Download from: /content/drive/MyDrive/compiled_dataset/models/
```

---

## Key Decisions to Make

### 1. Which Target to Predict?
- **adr_target**: ADR (Alternative Dispute Resolution) suitability
- **odr_target**: ODR (Online Dispute Resolution) suitability

Current setup uses `adr_target`. To switch:
```bash
python 07_train_structured_model.py --target odr_target
```

### 2. Hyperparameter Tuning?
Current values are reasonable defaults. For optimization:
- Use GridSearchCV (see Step 7 notebook)
- Try: max_depth ∈ [5, 10, 15, 20]
- Try: n_estimators ∈ [50, 100, 200, 500]

### 3. Text Features or Not?
- **Now**: Structured model only (good 75-85% baseline)
- **Later**: Add text features (Step 8) for 85%+ accuracy

---

## Monitoring Training

### Warnings to Ignore
```
DeprecationWarning: ...
UserWarning: ...
```
These are normal and safe.

### Errors to Fix
```
FileNotFoundError: training_data.parquet not found
→ Fix: Check data_path is correct
```

### Signs of Good Training
- Accuracy increasing over iterations
- F1 score > 0.60
- Training time reasonable (5-10 min)

---

## Model Performance Expectations

| Model | Accuracy | F1 Score | Speed |
|-------|----------|----------|-------|
| RF Baseline | 75% | 0.65 | ⚡⚡⚡ |
| XGB Baseline | 78% | 0.68 | ⚡⚡ |
| RF + SMOTE | 76% | 0.72 | ⚡⚡⚡ |
| XGB + SMOTE | **80%** | **0.75** | ⚡⚡ |

Note: Actual performance varies based on data quality and class balance.

---

## Common Issues & Solutions

### Issue: Very Low Accuracy (<50%)
```
Solution 1: Check if features have many missing values
Solution 2: Verify target labels are correct
Solution 3: Try adding text features (Step 8)
```

### Issue: All Predictions are Class 0
```
Solution: SMOTE should fix this (already in Step 7)
If not: Try class_weight='balanced' in model parameters
```

### Issue: Memory Error During Training
```
Solution 1: Sample data: df.sample(frac=0.5)
Solution 2: Use smaller model: max_depth=5, n_estimators=50
Solution 3: Reduce batch processing
```

---

## Next Actions

1. **Read**: `QUICKSTART_MODEL_TRAINING.md` (for Colab setup)
2. **Run**: `07_train_structured_model.ipynb` or `.py`
3. **Evaluate**: Check confusion matrices and feature importance
4. **Save**: Model will auto-save to `compiled_dataset/models/`
5. **Plan**: Decide if you want Step 8 (text model) or deploy now

---

## Support Files

| File | Purpose |
|------|---------|
| `QUICKSTART_MODEL_TRAINING.md` | Colab setup guide |
| `STEP_07_MODEL_TRAINING.md` | Detailed documentation |
| `README.md` | Project overview |
| `requirements.txt` | All dependencies |

---

**Ready to train?** Start with `QUICKSTART_MODEL_TRAINING.md` or run:
```bash
python 07_train_structured_model.py
```

**Questions?** Check the relevant `.md` file or examine the notebook cells.
