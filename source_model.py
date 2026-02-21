"""
source_model.py — Delhi Air Pollution Source Fingerprint Classifier
Trains XGBoost multi-class model to identify pollution sources.

Usage:  python source_model.py
Output: models/source_classifier.json   (trained model)
        models/classification_report.txt
        models/feature_importance.png
        models/shap_summary.png
        models/confusion_matrix.png
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score
)
from sklearn.utils.class_weight import compute_sample_weight

import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ═════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════

DATA_PATH = Path("data/model_ready_delhi.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# Columns to EXCLUDE from features
DROP_COLS = [
    "location_name", "datetime_hour", "latitude", "longitude",
    "source_label",  # target
]

# XGBoost hyperparameters (tuned for this dataset)
XGB_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "eval_metric": "mlogloss",
    "early_stopping_rounds": 30,
}

CV_FOLDS = 5

# ═════════════════════════════════════════════════════════════════
# STEP 1: LOAD & PREPARE DATA
# ═════════════════════════════════════════════════════════════════

def load_data():
    print("=" * 60)
    print("STEP 1: Loading data")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH)
    print(f"   Loaded: {df.shape}")

    # Target
    y_raw = df["source_label"]
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    class_names = le.classes_
    print(f"   Classes: {list(class_names)}")
    print(f"   Distribution:")
    for cls, cnt in zip(*np.unique(y, return_counts=True)):
        print(f"     {class_names[cls]:20s}: {cnt:6d} ({100*cnt/len(y):.1f}%)")

    # Features
    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    X = df[feature_cols].copy()

    # Ensure all numeric
    for col in X.columns:
        if X[col].dtype == "object":
            print(f"   ⚠ Dropping non-numeric column: {col}")
            X.drop(columns=[col], inplace=True)

    # Keep NaN as true missing values so XGBoost can route them explicitly.
    nan_cols = X.columns[X.isnull().any()].tolist()
    if nan_cols:
        print(f"   NaN present in: {nan_cols}")

    print(f"   Features: {X.shape[1]}")
    print(f"   Samples:  {X.shape[0]}")

    return X, y, le, class_names, feature_cols

# ═════════════════════════════════════════════════════════════════
# STEP 2: TRAIN WITH STRATIFIED K-FOLD CV
# ═════════════════════════════════════════════════════════════════

def train_model(X, y, class_names):
    print("\n" + "=" * 60)
    print(f"STEP 2: Training XGBoost ({CV_FOLDS}-fold Stratified CV)")
    print("=" * 60)

    # Compute sample weights for class imbalance
    sample_weights = compute_sample_weight("balanced", y)

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)

    # Cross-val predictions for evaluation
    y_pred_cv = np.zeros(len(y), dtype=int)
    y_proba_cv = np.zeros((len(y), len(class_names)))
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n   Fold {fold}/{CV_FOLDS}")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        w_train = sample_weights[train_idx]

        model = xgb.XGBClassifier(
            **{k: v for k, v in XGB_PARAMS.items() if k != "early_stopping_rounds"},
            use_label_encoder=False,
        )
        model.fit(
            X_train, y_train,
            sample_weight=w_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        y_val_pred = model.predict(X_val)
        y_val_proba = model.predict_proba(X_val)
        y_pred_cv[val_idx] = y_val_pred
        y_proba_cv[val_idx] = y_val_proba

        acc = accuracy_score(y_val, y_val_pred)
        f1 = f1_score(y_val, y_val_pred, average="weighted")
        fold_scores.append({"fold": fold, "accuracy": acc, "f1_weighted": f1})
        print(f"     Accuracy: {acc:.4f}  |  F1 (weighted): {f1:.4f}")

    print("\n   ── CV Summary ──")
    mean_acc = np.mean([s["accuracy"] for s in fold_scores])
    mean_f1  = np.mean([s["f1_weighted"] for s in fold_scores])
    print(f"   Mean Accuracy:     {mean_acc:.4f}")
    print(f"   Mean F1 (weighted): {mean_f1:.4f}")

    return y_pred_cv, y_proba_cv, fold_scores

# ═════════════════════════════════════════════════════════════════
# STEP 3: TRAIN FINAL MODEL ON ALL DATA
# ═════════════════════════════════════════════════════════════════

def train_final_model(X, y):
    print("\n" + "=" * 60)
    print("STEP 3: Training final model on all data")
    print("=" * 60)

    sample_weights = compute_sample_weight("balanced", y)

    # Remove early_stopping since no val set
    params = {k: v for k, v in XGB_PARAMS.items() if k != "early_stopping_rounds"}

    model = xgb.XGBClassifier(**params, use_label_encoder=False)
    model.fit(X, y, sample_weight=sample_weights, verbose=True)

    # Save model
    model_path = MODEL_DIR / "source_classifier.json"
    model.save_model(str(model_path))
    print(f"   ✓ Model saved: {model_path}")

    # Also save as joblib for sklearn compatibility
    joblib_path = MODEL_DIR / "source_classifier.joblib"
    joblib.dump(model, str(joblib_path))
    print(f"   ✓ Joblib saved: {joblib_path}")

    return model

# ═════════════════════════════════════════════════════════════════
# STEP 4: EVALUATION & VISUALIZATIONS
# ═════════════════════════════════════════════════════════════════

def evaluate_and_plot(y, y_pred_cv, y_proba_cv, class_names, model, X):
    print("\n" + "=" * 60)
    print("STEP 4: Evaluation & Visualizations")
    print("=" * 60)

    # ── Classification Report ──
    report = classification_report(y, y_pred_cv, target_names=class_names)
    print("\n" + report)

    report_path = MODEL_DIR / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write("Delhi Air Pollution Source Classifier — CV Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(report)
        f.write(f"\nOverall Accuracy: {accuracy_score(y, y_pred_cv):.4f}\n")
        f.write(f"Weighted F1: {f1_score(y, y_pred_cv, average='weighted'):.4f}\n")
    print(f"   ✓ Report saved: {report_path}")

    # ── Confusion Matrix ──
    cm = confusion_matrix(y, y_pred_cv)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("Source Classification — Confusion Matrix (5-Fold CV)", fontsize=14)
    plt.tight_layout()
    cm_path = MODEL_DIR / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"   ✓ Confusion matrix: {cm_path}")

    # ── Feature Importance ──
    importance = model.feature_importances_
    feat_names = X.columns
    sorted_idx = np.argsort(importance)[-25:]  # top 25

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(sorted_idx)), importance[sorted_idx], color="#4C72B0")
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels(feat_names[sorted_idx], fontsize=10)
    ax.set_xlabel("Feature Importance (Gain)", fontsize=12)
    ax.set_title("Top 25 Features — Source Classifier", fontsize=14)
    plt.tight_layout()
    fi_path = MODEL_DIR / "feature_importance.png"
    fig.savefig(fi_path, dpi=150)
    plt.close(fig)
    print(f"   ✓ Feature importance: {fi_path}")

    # ── SHAP Values ──
    print("\n   Computing SHAP values (this may take a minute)...")
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        # Use a sample for speed
        sample_idx = np.random.RandomState(42).choice(len(X), min(2000, len(X)), replace=False)
        X_sample = X.iloc[sample_idx]
        shap_values = explainer.shap_values(X_sample)

        fig, ax = plt.subplots(figsize=(12, 8))
        # For multi-class, shap_values is a list of arrays
        if isinstance(shap_values, list):
            # Use the class with most variance in SHAP
            shap_abs = [np.abs(sv).mean(0) for sv in shap_values]
            best_class = np.argmax([s.sum() for s in shap_abs])
            shap.summary_plot(shap_values[best_class], X_sample,
                              max_display=20, show=False)
            plt.title(f"SHAP Values — Class: {class_names[best_class]}", fontsize=14)
        else:
            shap.summary_plot(shap_values, X_sample,
                              max_display=20, show=False)
            plt.title("SHAP Summary", fontsize=14)
        plt.tight_layout()
        shap_path = MODEL_DIR / "shap_summary.png"
        plt.savefig(shap_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"   ✓ SHAP plot: {shap_path}")
    except Exception as e:
        print(f"   ⚠ SHAP failed: {e}")

    return report

# ═════════════════════════════════════════════════════════════════
# STEP 5: SAVE METADATA
# ═════════════════════════════════════════════════════════════════

def save_metadata(le, class_names, feature_cols, fold_scores, X):
    """Save everything needed to reload and use the model."""
    meta = {
        "class_names": list(class_names),
        "label_encoding": {name: int(idx) for idx, name in enumerate(class_names)},
        "feature_columns": list(X.columns),
        "n_features": len(X.columns),
        "n_samples": len(X),
        "cv_folds": CV_FOLDS,
        "cv_scores": fold_scores,
        "xgb_params": {k: v for k, v in XGB_PARAMS.items()},
        "drop_columns": DROP_COLS,
    }

    meta_path = MODEL_DIR / "model_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"   ✓ Metadata: {meta_path}")

    # Save label encoder
    le_path = MODEL_DIR / "label_encoder.joblib"
    joblib.dump(le, str(le_path))
    print(f"   ✓ Label encoder: {le_path}")

# ═════════════════════════════════════════════════════════════════
# STEP 6: PREDICTION HELPER (for later use)
# ═════════════════════════════════════════════════════════════════

def predict_source(model_path="models/source_classifier.json",
                   meta_path="models/model_metadata.json",
                   input_df=None):
    """
    Load saved model and predict pollution source for new data.

    Usage:
        from source_model import predict_source
        results = predict_source(input_df=new_data)
    """
    with open(meta_path) as f:
        meta = json.load(f)

    model = xgb.XGBClassifier()
    model.load_model(model_path)

    class_names = meta["class_names"]
    feature_cols = meta["feature_columns"]

    X = input_df.reindex(columns=feature_cols)
    X = X.apply(pd.to_numeric, errors="coerce")
    proba = model.predict_proba(X)
    pred_idx = proba.argmax(axis=1)

    results = pd.DataFrame({
        "predicted_source": [class_names[i] for i in pred_idx],
        "confidence": proba.max(axis=1),
    })
    for i, cls in enumerate(class_names):
        results[f"prob_{cls}"] = proba[:, i]

    return results

# ═════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════

def main():
    import time
    t0 = time.time()

    print("╔══════════════════════════════════════════════════════════╗")
    print("║  Pollution Source Fingerprint Classifier                 ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    # Step 1: Load
    X, y, le, class_names, feature_cols = load_data()

    # Step 2: Cross-validated evaluation
    y_pred_cv, y_proba_cv, fold_scores = train_model(X, y, class_names)

    # Step 3: Train final model on all data
    model = train_final_model(X, y)

    # Step 4: Evaluate & visualize
    evaluate_and_plot(y, y_pred_cv, y_proba_cv, class_names, model, X)

    # Step 5: Save metadata
    save_metadata(le, class_names, feature_cols, fold_scores, X)

    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print(f"✅ DONE in {elapsed:.1f}s")
    print(f"   Model:   models/source_classifier.json")
    print(f"   Report:  models/classification_report.txt")
    print(f"   Plots:   models/confusion_matrix.png")
    print(f"            models/feature_importance.png")
    print(f"            models/shap_summary.png")
    print("=" * 60)

if __name__ == "__main__":
    main()
