#!/usr/bin/env python3
"""
ADR/ODR Suitability Classifier — Structured Model Training Script
Trains Random Forest and XGBoost models on categorical features.
Run: python 07_train_structured_model.py --data-path /path/to/compiled_dataset
"""

import argparse
import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import xgboost as xgb

import matplotlib.pyplot as plt
import seaborn as sns


def load_data(data_path):
    """Load training data from parquet."""
    training_file = os.path.join(data_path, "training_data.parquet")
    if not os.path.exists(training_file):
        raise FileNotFoundError(f"training_data.parquet not found at {training_file}")

    df = pd.read_parquet(training_file)
    print(f"Loaded data: {df.shape}")
    return df


def prepare_features(df, target="adr_target"):
    """Prepare features and target."""
    # Remove unknown labels
    df_clean = df[df[target] != -1].copy()
    print(f"After removing unknown labels: {df_clean.shape}")

    # Select key features
    feature_cols = []
    for col in ["act_name", "case_type", "source", "year", "court_level"]:
        if col in df_clean.columns:
            feature_cols.append(col)

    print(f"Selected features: {feature_cols}")

    # Prepare X and y
    X = df_clean[feature_cols].copy()
    y = df_clean[target].copy()

    # Handle missing values
    for col in X.select_dtypes(include="object").columns:
        X[col] = X[col].fillna("Unknown")
    for col in X.select_dtypes(include=["int64", "float64"]).columns:
        X[col] = X[col].fillna(X[col].median())

    # Encode categorical features
    label_encoders = {}
    X_encoded = X.copy()
    for col in X.select_dtypes(include="object").columns:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
        print(f"{col}: {len(le.classes_)} unique values")

    return X_encoded, y, feature_cols, label_encoders


def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into train and test."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Train set: {X_train.shape[0]}, Test set: {X_test.shape[0]}")
    print(f"Train class distribution:\n{y_train.value_counts().sort_index()}")
    return X_train, X_test, y_train, y_test


def balance_with_smote(X_train, y_train):
    """Apply SMOTE for class balancing."""
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE: {y_train_smote.value_counts().sort_index()}")
    return X_train_smote, y_train_smote


def train_random_forest(X_train, y_train, **kwargs):
    """Train Random Forest classifier."""
    print("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
        **kwargs,
    )
    rf.fit(X_train, y_train)
    return rf


def train_xgboost(X_train, y_train, **kwargs):
    """Train XGBoost classifier."""
    print("Training XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        **kwargs,
    )
    xgb_model.fit(X_train, y_train)
    return xgb_model


def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model performance."""
    y_pred = model.predict(X_test)
    accuracy = (y_pred == y_test).mean()
    f1 = f1_score(y_test, y_pred)

    print(f"\n{model_name}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(classification_report(y_test, y_pred))

    return {
        "name": model_name,
        "model": model,
        "preds": y_pred,
        "accuracy": accuracy,
        "f1": f1,
    }


def save_model(model, path):
    """Save model to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="Train ADR/ODR classification model")
    parser.add_argument(
        "--data-path", default="./compiled_dataset", help="Path to compiled_dataset"
    )
    parser.add_argument("--target", default="adr_target", help="Target column")
    parser.add_argument(
        "--output-dir",
        default="./compiled_dataset/models",
        help="Output directory for models",
    )
    args = parser.parse_args()

    # Load and prepare data
    df = load_data(args.data_path)
    X, y, feature_cols, label_encoders = prepare_features(df, args.target)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train baseline models
    print("\n" + "=" * 80)
    print("BASELINE MODELS (without SMOTE)")
    print("=" * 80)

    rf_baseline = train_random_forest(X_train, y_train)
    results_rf_baseline = evaluate_model(
        rf_baseline, X_test, y_test, "Random Forest Baseline"
    )

    xgb_baseline = train_xgboost(X_train, y_train)
    results_xgb_baseline = evaluate_model(
        xgb_baseline, X_test, y_test, "XGBoost Baseline"
    )

    # Apply SMOTE and retrain
    print("\n" + "=" * 80)
    print("MODELS WITH SMOTE")
    print("=" * 80)

    X_train_smote, y_train_smote = balance_with_smote(X_train, y_train)

    rf_smote = train_random_forest(X_train_smote, y_train_smote)
    results_rf_smote = evaluate_model(rf_smote, X_test, y_test, "Random Forest + SMOTE")

    xgb_smote = train_xgboost(X_train_smote, y_train_smote)
    results_xgb_smote = evaluate_model(xgb_smote, X_test, y_test, "XGBoost + SMOTE")

    # Summary
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)

    all_results = [
        results_rf_baseline,
        results_xgb_baseline,
        results_rf_smote,
        results_xgb_smote,
    ]
    comparison_df = pd.DataFrame(
        [
            {"Model": r["name"], "Accuracy": r["accuracy"], "F1 Score": r["f1"]}
            for r in all_results
        ]
    ).sort_values("Accuracy", ascending=False)

    print(comparison_df.to_string(index=False))

    # Save best model
    best_result = all_results[np.argmax([r["accuracy"] for r in all_results])]
    best_model = best_result["model"]
    best_name = best_result["name"]

    print(f"\nBest model: {best_name}")

    os.makedirs(args.output_dir, exist_ok=True)
    save_model(best_model, os.path.join(args.output_dir, "best_model.pkl"))

    # Save metadata
    metadata = {
        "feature_columns": feature_cols,
        "label_encoders": label_encoders,
        "target_column": args.target,
        "model_name": best_name,
        "accuracy": best_result["accuracy"],
        "f1_score": best_result["f1"],
    }

    with open(os.path.join(args.output_dir, "metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)
    print(f"Metadata saved to {os.path.join(args.output_dir, 'metadata.pkl')}")

    # Feature importance
    print("\n" + "=" * 80)
    print("FEATURE IMPORTANCE")
    print("=" * 80)

    feature_importance = pd.DataFrame(
        {"feature": feature_cols, "importance": best_model.feature_importances_}
    ).sort_values("importance", ascending=False)

    print(feature_importance.to_string(index=False))


if __name__ == "__main__":
    main()
