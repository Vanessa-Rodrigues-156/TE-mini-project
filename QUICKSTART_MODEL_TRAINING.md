# Quick Start Guide — Model Training on Colab

## Setup (First Time Only)

### 1. Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 2. Install Dependencies
```bash
!pip install -r requirements.txt
```

Ensure these key packages are installed:
- `pandas`, `numpy` — Data manipulation
- `scikit-learn` — ML models
- `xgboost` — Gradient boosting
- `imbalanced-learn` — SMOTE for balancing
- `matplotlib`, `seaborn` — Visualization

## Training Flow

### Option 1: Run as Jupyter Notebook (Recommended for Exploration)
```python
# In Colab cell 1
%cd /path/to/TE-mini-project

# Then open and run 07_train_structured_model.ipynb
# Modify the data_path variable to point to your Google Drive location:
data_path = '/content/drive/My Drive/compiled_dataset'
```

**Advantages:**
- Interactive exploration
- Visualizations inline
- Easy to modify and iterate

### Option 2: Run as Python Script (Recommended for Production)
```bash
!python 07_train_structured_model.py \
  --data-path '/content/drive/My Drive/compiled_dataset' \
  --target adr_target \
  --output-dir '/content/drive/My Drive/compiled_dataset/models'
```

## What the Training Does

### Phase 1: Data Preparation
- ✅ Loads `training_data.parquet`
- ✅ Removes unknown labels (-1)
- ✅ Selects features: `act_name`, `case_type`, `source`, `year`, `court_level`
- ✅ Handles missing values
- ✅ Encodes categorical features

### Phase 2: Baseline Training
- ✅ **Random Forest**: 100 trees, max_depth=15
- ✅ **XGBoost**: 100 trees, max_depth=5
- ✅ Evaluates on 20% test set

### Phase 3: Class Balancing
- ✅ Applies SMOTE (Synthetic Minority Oversampling)
- ✅ Retrains RF and XGBoost with balanced data
- ✅ Compares performance before/after balancing

### Phase 4: Model Selection & Saving
- ✅ Selects best model (usually XGBoost + SMOTE)
- ✅ Saves as `best_model.pkl`
- ✅ Saves metadata (encoders, feature names)

## Expected Outputs

After training completes, you'll have:

```
compiled_dataset/
└── models/
    ├── best_model.pkl          # Ready-to-use model
    └── metadata.pkl            # Feature encoders & info
```

**Training time**: ~5-10 minutes on Colab (depending on dataset size)

## Making Predictions

```python
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load model and metadata
with open('models/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

# Example prediction
test_case = pd.DataFrame({
    'act_name': ['Code of Civil Procedure'],
    'case_type': ['Civil'],
    'source': ['DDL'],
    'year': [2023],
    'court_level': [0]
})

# Encode categorical features
for col in metadata['label_encoders']:
    encoder = metadata['label_encoders'][col]
    test_case[col] = encoder.transform(test_case[col])

# Predict
prediction = model.predict(test_case)
probability = model.predict_proba(test_case)

print(f"Prediction: {prediction[0]}")  # 0 or 1
print(f"Probability (Not Suitable): {probability[0][0]:.2%}")
print(f"Probability (ADR Suitable): {probability[0][1]:.2%}")
```

## Key Metrics to Monitor

### Accuracy
- Good baseline metric
- Can be misleading with imbalanced data

### F1 Score (More Important)
- Harmonic mean of precision and recall
- Better for imbalanced datasets

### Confusion Matrix
- Understand False Positives vs False Negatives
- High FP → Over-predicting ADR suitability
- High FN → Missing ADR-suitable cases

## Troubleshooting

### Issue: File Not Found
```
FileNotFoundError: training_data.parquet not found
```
**Fix**: Verify the data_path is correct and your Google Drive is mounted

### Issue: Out of Memory
```
MemoryError: Unable to allocate X GB
```
**Fix**: Sample the data before training:
```python
df = pd.read_parquet(...)
df = df.sample(frac=0.5)  # Use 50% of data
```

### Issue: Poor Model Performance
If F1 Score < 0.50:
- Check target label distribution
- Verify features aren't all missing
- Try increasing model complexity (max_depth, n_estimators)
- Move to Step 8: Add text features for better performance

## Next: Text Model Training (Step 8)

After structured model works well, you can:
1. Extract text features from case descriptions
2. Fine-tune LegalBERT
3. Combine with structured model for ensemble

For now, focus on getting the structured model working!

## Files Reference

| File | Purpose |
|------|---------|
| `07_train_structured_model.ipynb` | Interactive Jupyter notebook |
| `07_train_structured_model.py` | Command-line script |
| `STEP_07_MODEL_TRAINING.md` | Detailed documentation |
| `requirements.txt` | All required packages |

---

**Questions?** Check the error messages or see `STEP_07_MODEL_TRAINING.md` for detailed troubleshooting.
