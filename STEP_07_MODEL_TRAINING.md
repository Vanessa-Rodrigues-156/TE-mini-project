# Step 7: Train Structured Model — Random Forest & XGBoost

## Overview
Train a **structured classification model** using categorical features (`act_name`, `case_type`, etc.) to predict ADR/ODR suitability.

## Model Approaches

### 1. **Baseline Models** (no balancing)
- **Random Forest**: Ensemble of decision trees
- **XGBoost**: Gradient boosting classifier

### 2. **Balanced Models** (with SMOTE)
- **RF + SMOTE**: Random Forest with Synthetic Minority Oversampling
- **XGBoost + SMOTE**: XGBoost with class balancing

## Quick Start

### Option A: Jupyter Notebook (Interactive)
```bash
jupyter notebook 07_train_structured_model.ipynb
```

**Steps:**
1. Mount Google Drive (if on Colab)
2. Load `training_data.parquet`
3. Explore features and target distribution
4. Train baseline models
5. Apply SMOTE for class balancing
6. Compare model performance
7. Save best model

### Option B: Python Script (Automated)
```bash
python 07_train_structured_model.py \
  --data-path ./compiled_dataset \
  --target adr_target \
  --output-dir ./compiled_dataset/models
```

## Features Used

| Feature | Type | Description |
|---------|------|-------------|
| `act_name` | Categorical | Legal act/statute name |
| `case_type` | Categorical | Type of case (Civil, Criminal, etc.) |
| `source` | Categorical | Data source (DDL, HC, SC) |
| `year` | Numerical | Year of case filing |
| `court_level` | Numerical | Court hierarchy level |

## Target Variables

### Primary: `adr_target` (Binary)
- **0**: Not suitable for ADR
- **1**: Suitable for ADR
- **-1**: Unknown (filtered before training)

### Alternative: `odr_target` (Binary)
- **0**: Not suitable for ODR
- **1**: Suitable for ODR (can be resolved online)

## Output Files

After training, models are saved in `compiled_dataset/models/`:

| File | Description |
|------|-------------|
| `best_model.pkl` | Best trained model (XGBoost or RF) |
| `metadata.pkl` | Feature encoders, column names, model info |
| Training logs | Metrics and feature importance |

## Notebook Walkthrough

### Step 1: Setup
- Mount Google Drive
- Import dependencies
- Load `training_data.parquet`

### Step 2: Target Selection
- Explore target distributions
- Remove unknown labels (-1)
- Check class imbalance

### Step 3: Feature Engineering
- Select key features
- Handle missing values
- Encode categorical variables using `LabelEncoder`

### Step 4: Data Splitting
- 80% train, 20% test (stratified)
- Analyze class distribution in train/test

### Step 5: Baseline Models
- Train Random Forest (100 trees, max_depth=15)
- Train XGBoost (100 trees, max_depth=5)
- Evaluate on test set

### Step 6: Class Balancing
- Apply SMOTE to training data
- Retrain RF and XGBoost
- Compare performance

### Step 7: Model Comparison
- Accuracy, F1 Score metrics
- Confusion matrices
- Feature importance rankings

### Step 8: Save & Export
- Save best model
- Export label encoders
- Create prediction function

## Key Metrics

### Accuracy
% of correct predictions

### F1 Score
Harmonic mean of precision and recall (better for imbalanced data)

### ROC-AUC
Area Under the Receiver Operating Characteristic curve

### Confusion Matrix
- True Positives (TP): Correctly predicted ADR-suitable
- False Positives (FP): Incorrectly predicted ADR-suitable
- True Negatives (TN): Correctly predicted not suitable
- False Negatives (FN): Incorrectly predicted not suitable

## Expected Performance

Based on the dataset quality and feature richness, expect:
- **Accuracy**: 75-85% (depending on class imbalance)
- **F1 Score**: 0.65-0.80 (after SMOTE)

Performance varies based on:
- Data quality and completeness
- Feature representation
- Class balance
- Test set characteristics

## Next Steps

### Option 1: Fine-tune Hyperparameters
```python
# In notebook or script
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 15, 20],
    'learning_rate': [0.01, 0.1, 0.5]
}

grid_search = GridSearchCV(xgb.XGBClassifier(), param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train_smote, y_train_smote)
```

### Option 2: Add Text Features (Step 8)
- Fine-tune LegalBERT on case titles/descriptions
- Extract embeddings as features
- Combine with structured model

### Option 3: Ensemble Methods
- Combine RF + XGBoost predictions
- Use Voting Classifier or Stacking

### Option 4: Production Deployment
- Create Flask/FastAPI service
- Containerize with Docker
- Deploy to cloud (AWS, GCP, Azure)

## Troubleshooting

### Issue: Memory Error
```
MemoryError: Unable to allocate X GB
```
**Solution**: 
- Reduce data sample: `df = df.sample(frac=0.5)`
- Use smaller models: fewer trees, lower max_depth

### Issue: Class Imbalance
```
Model predicts everything as class 0
```
**Solution**:
- Already handled with SMOTE in the notebook
- Adjust class_weight in model: `class_weight='balanced'`

### Issue: Poor Performance
```
F1 Score < 0.50
```
**Solution**:
- Add more features from text (Step 8)
- Check feature quality and missing values
- Try different hyperparameters (GridSearchCV)
- Verify target labels are correct

## Files Reference

- **Notebook**: `07_train_structured_model.ipynb`
- **Script**: `07_train_structured_model.py`
- **Input**: `compiled_dataset/training_data.parquet`
- **Output**: `compiled_dataset/models/`

---

**Next**: Step 08 - Text Model Training with LegalBERT
