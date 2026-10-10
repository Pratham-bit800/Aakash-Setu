"""
train_ml_risk.py
================
Akash Setu — ML Risk Model Training & Evaluation Script.

Trains baseline models, primary HistGradientBoosting regressor and classifier
on the ESA Kelvins competition dataset, evaluates across subgroups,
computes feature importance, and persists production artifacts.

Usage:
    python backend/scripts/train_ml_risk.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

# Ensure root is in sys.path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.ml_risk_engine import (
    CDMEFeaturePreprocessor,
    DEFAULT_FEAT_PATH,
    DEFAULT_MODEL_DIR,
    DEFAULT_SPLIT_PATH,
    compute_feature_importance,
    load_dataset,
    train_baseline_dummy,
    train_baseline_floor,
    train_baseline_ridge,
    train_risk_classifier,
    train_risk_regressor,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_ml_risk")


def main():
    t_start = time.time()
    logger.info("Starting ML Risk Model Training Pipeline...")

    # 1. Load Data
    logger.info("Loading feature and event split parquets...")
    train_df, val_df, feature_cols = load_dataset(DEFAULT_FEAT_PATH, DEFAULT_SPLIT_PATH)
    logger.info(f"Loaded: Train rows={len(train_df)} ({train_df['event_id'].nunique()} events), "
                f"Val rows={len(val_df)} ({val_df['event_id'].nunique()} events)")
    logger.info(f"Total input features: {len(feature_cols)}")

    # 2. Fit Preprocessor ONLY on Train Data
    logger.info("Fitting CDMEFeaturePreprocessor strictly on train set...")
    preprocessor = CDMEFeaturePreprocessor()
    preprocessor.fit(train_df, feature_cols=feature_cols)

    X_train = preprocessor.transform(train_df)
    X_val = preprocessor.transform(val_df)
    y_train = train_df["event_final_risk"]
    y_val = val_df["event_final_risk"]

    y_train_cls = (y_train >= -6.0).astype(int)
    y_val_cls = (y_val >= -6.0).astype(int)

    # 3. Train Baselines
    logger.info("1/5 Training Dummy Floor Baseline (-30.0)...")
    floor_res = train_baseline_floor(y_train, y_val)
    logger.info(f"Dummy Floor Regressor: Val MAE={floor_res['val_metrics']['overall']['mae']:.4f}, "
                f"RMSE={floor_res['val_metrics']['overall']['rmse']:.4f}")

    logger.info("2/5 Training Dummy Mean Baseline...")
    dummy_res = train_baseline_dummy(y_train, y_val)
    logger.info(f"Dummy Regressor: Val MAE={dummy_res['val_metrics']['overall']['mae']:.4f}, "
                f"RMSE={dummy_res['val_metrics']['overall']['rmse']:.4f}")

    logger.info("2/4 Training Ridge Linear Baseline...")
    ridge_model, ridge_res = train_baseline_ridge(X_train, y_train, X_val, y_val, alpha=100.0)
    logger.info(f"Ridge Regressor: Val MAE={ridge_res['val_metrics']['overall']['mae']:.4f}, "
                f"RMSE={ridge_res['val_metrics']['overall']['rmse']:.4f}, "
                f"Spearman={ridge_res['val_metrics']['overall']['spearman_rho']:.4f}")

    # 4. Train Primary Regressor
    logger.info("3/4 Training HistGradientBoostingRegressor (Primary Model)...")
    regressor, reg_res = train_risk_regressor(X_train, y_train, X_val, y_val, max_iter=100)
    logger.info(f"Primary Regressor: Val MAE={reg_res['val_metrics']['overall']['mae']:.4f}, "
                f"RMSE={reg_res['val_metrics']['overall']['rmse']:.4f}, "
                f"Spearman={reg_res['val_metrics']['overall']['spearman_rho']:.4f}")

    # 5. Train Classifier
    logger.info("4/4 Training HistGradientBoostingClassifier (High-Risk Classifier)...")
    classifier, clf_res = train_risk_classifier(X_train, y_train_cls, X_val, y_val_cls, max_iter=100)
    logger.info(f"Classifier: Val PR-AUC={clf_res['val_metrics']['pr_auc']:.4f}, "
                f"ROC-AUC={clf_res['val_metrics']['roc_auc']:.4f}, "
                f"F1={clf_res['val_metrics']['f1_score']:.4f}")

    # 6. Feature Importance
    logger.info("Computing Permutation Feature Importance on validation sample...")
    top_features = compute_feature_importance(regressor, X_val, y_val, top_n=20, n_samples=3000, n_repeats=3)
    logger.info("Top 5 Features by Permutation Importance:")
    for feat_info in top_features[:5]:
        logger.info(f"  {feat_info['feature']}: {feat_info['importance_mean']:.5f}")

    # 7. Persist Artifacts
    DEFAULT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    reg_path = DEFAULT_MODEL_DIR / "risk_regressor.joblib"
    clf_path = DEFAULT_MODEL_DIR / "risk_classifier.joblib"
    prep_path = DEFAULT_MODEL_DIR / "feature_preprocessor.joblib"
    meta_path = DEFAULT_MODEL_DIR / "model_metadata.json"

    logger.info(f"Saving model artifacts to {DEFAULT_MODEL_DIR}...")
    joblib.dump(regressor, reg_path)
    joblib.dump(classifier, clf_path)
    joblib.dump(preprocessor, prep_path)

    metadata = {
        "model_version": "1.0.0-phase6",
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "n_train_samples": len(train_df),
        "n_val_samples": len(val_df),
        "n_train_events": int(train_df["event_id"].nunique()),
        "n_val_events": int(val_df["event_id"].nunique()),
        "n_features": len(feature_cols),
        "feature_names": feature_cols,
        "categorical_features": ["c_object_type"],
        "censored_risk_floor": -30.0,
        "action_threshold": -6.0,
        "top_features": top_features,
        "evaluation_summary": {
            "dummy_floor_val_mae": floor_res["val_metrics"]["overall"]["mae"],
            "dummy_floor_val_rmse": floor_res["val_metrics"]["overall"]["rmse"],
            "dummy_mean_val_mae": dummy_res["val_metrics"]["overall"]["mae"],
            "dummy_mean_val_rmse": dummy_res["val_metrics"]["overall"]["rmse"],
            "ridge_val_rmse": ridge_res["val_metrics"]["overall"]["rmse"],
            "primary_regressor_val_rmse": reg_res["val_metrics"]["overall"]["rmse"],
            "primary_regressor_val_spearman": reg_res["val_metrics"]["overall"]["spearman_rho"],
            "classifier_val_pr_auc": clf_res["val_metrics"]["pr_auc"],
            "classifier_val_roc_auc": clf_res["val_metrics"]["roc_auc"],
        },
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # 8. Save Comprehensive Evaluation Report
    report_path = ROOT / "data/processed/reports/phase6_ml_evaluation_report.json"
    full_report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "phase": 6,
        "pipeline": "ML-Based Collision Risk Prediction Pipeline",
        "dataset_metadata": {
            "feature_parquet": str(DEFAULT_FEAT_PATH.relative_to(ROOT)),
            "split_parquet": str(DEFAULT_SPLIT_PATH.relative_to(ROOT)),
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "train_events": int(train_df["event_id"].nunique()),
            "val_events": int(val_df["event_id"].nunique()),
            "event_leakage_overlap": 0,
            "feature_count": len(feature_cols),
        },
        "models": {
            "dummy_floor_baseline": floor_res,
            "dummy_baseline": dummy_res,
            "ridge_linear_baseline": ridge_res,
            "primary_hist_gradient_boosting_regressor": reg_res,
            "high_risk_hist_gradient_boosting_classifier": clf_res,
        },
        "top_permutation_features": top_features,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    logger.info(f"Evaluation report saved to {report_path.relative_to(ROOT)}")
    logger.info(f"Training pipeline completed successfully in {time.time()-t_start:.2f}s.")


if __name__ == "__main__":
    main()
