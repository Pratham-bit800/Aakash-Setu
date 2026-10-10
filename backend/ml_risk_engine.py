"""
ml_risk_engine.py
=================
Akash Setu — Machine Learning-Based Collision Risk Prediction Engine.

Provides end-to-end ML training, evaluation, and inference for conjunction risk
assessment based on the ESA Kelvins Collision Avoidance Challenge dataset.

Key Astrodynamic & ML Principles:
1. Strict Event-Level Partitioning:
   Data is split exclusively by event_id to prevent temporal and state leakage across CDMs.
2. Leakage Prevention:
   Columns risk, true_risk, max_risk_estimate, max_risk_scaling, event_final_risk,
   event_id, mission_id, and is_last_cdm are strictly excluded from feature matrix X.
3. Proper Handling of Censored Risk Floor:
   In ESA CDMs, log10 collision probability is clamped at -30.0 for negligible risk.
   Evaluation separates clamped (-30) vs unclamped (> -30) subgroups.
4. Scientific Disclaimer:
   Trained on anonymized historical ESA Kelvins challenge CDMs for research purposes.
   Does not represent calibrated operational collision probabilities for active satellites.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_recall_fscore_support,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

DEFAULT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FEAT_PATH = DEFAULT_ROOT / "data/processed/esa_train_features.parquet"
DEFAULT_SPLIT_PATH = DEFAULT_ROOT / "data/processed/esa_event_split.parquet"
DEFAULT_MODEL_DIR = DEFAULT_ROOT / "backend/models/phase6"

RISK_ACTION_THRESHOLD: float = -6.0  # log10(P) >= -6.0 (P >= 1e-6, ESA maneuver threshold)
CENSORED_RISK_FLOOR: float = -30.0    # Lower bound in ESA dataset

NEVER_IN_X: List[str] = [
    "risk",
    "true_risk",
    "max_risk_estimate",
    "max_risk_scaling",
    "event_final_risk",
    "event_id",
    "mission_id",
    "is_last_cdm",
    "split",
]

CATEGORICAL_COLS: List[str] = ["c_object_type"]
ALLOWED_OBJECT_TYPES: List[str] = ["DEBRIS", "UNKNOWN", "PAYLOAD", "ROCKET BODY", "TBA"]


class CDMEFeaturePreprocessor:
    """Fits preprocessing transformations strictly on training data."""

    def __init__(self):
        self.feature_cols: List[str] = []
        self.numeric_cols: List[str] = []
        self.categorical_cols: List[str] = list(CATEGORICAL_COLS)
        self.numeric_medians: Dict[str, float] = {}
        self.object_type_categories: List[str] = list(ALLOWED_OBJECT_TYPES)
        self.is_fitted: bool = False

    def fit(self, df_train: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> "CDMEFeaturePreprocessor":
        if feature_cols is None:
            feature_cols = [c for c in df_train.columns if c not in NEVER_IN_X]
        self.feature_cols = list(feature_cols)
        self.numeric_cols = [c for c in self.feature_cols if c not in self.categorical_cols]

        medians = df_train[self.numeric_cols].median(numeric_only=True).to_dict()
        self.numeric_medians = {k: float(v) if pd.notnull(v) else 0.0 for k, v in medians.items()}
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before calling transform.")
        res = df.copy()

        for col in self.numeric_cols:
            if col not in res.columns:
                res[col] = self.numeric_medians.get(col, 0.0)

        if "c_object_type" not in res.columns:
            res["c_object_type"] = "UNKNOWN"

        res["c_object_type"] = pd.Categorical(
            res["c_object_type"].fillna("UNKNOWN").astype(str),
            categories=self.object_type_categories,
        )

        return res[self.feature_cols]

    def transform_single(self, record: Dict[str, Any]) -> pd.DataFrame:
        """Prepares a single CDM dictionary for inference, applying defaults for missing values."""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before calling transform_single.")
        row_dict = {}
        for col in self.numeric_cols:
            val = record.get(col)
            if val is None or (isinstance(val, (int, float)) and np.isnan(val)):
                row_dict[col] = self.numeric_medians.get(col, 0.0)
            else:
                try:
                    row_dict[col] = float(val)
                except (ValueError, TypeError):
                    row_dict[col] = self.numeric_medians.get(col, 0.0)

        obj_type = record.get("c_object_type", "UNKNOWN")
        if obj_type not in self.object_type_categories:
            obj_type = "UNKNOWN"
        row_dict["c_object_type"] = obj_type

        df_single = pd.DataFrame([row_dict])
        df_single["c_object_type"] = pd.Categorical(
            df_single["c_object_type"],
            categories=self.object_type_categories,
        )
        return df_single[self.feature_cols]


def load_dataset(
    feat_path: Union[str, Path] = DEFAULT_FEAT_PATH,
    split_path: Union[str, Path] = DEFAULT_SPLIT_PATH,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Loads feature matrix and merges with event split."""
    feat = pd.read_parquet(feat_path)
    split = pd.read_parquet(split_path)

    merged = feat.merge(split, on="event_id", how="left")
    feature_cols = [c for c in feat.columns if c not in NEVER_IN_X]

    train_df = merged[merged["split"] == "train"].copy()
    val_df = merged[merged["split"] == "val"].copy()

    # Validate zero event leakage
    train_events = set(train_df["event_id"].unique())
    val_events = set(val_df["event_id"].unique())
    overlap = train_events.intersection(val_events)
    if overlap:
        raise ValueError(f"Event leakage detected! {len(overlap)} events in both train and val.")

    return train_df, val_df, feature_cols


def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    time_to_tca: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Calculates comprehensive regression metrics overall and across subgroups."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(root_mean_squared_error(y_true, y_pred))
    rho, _ = spearmanr(y_true, y_pred)
    pearson, _ = pearsonr(y_true, y_pred) if np.std(y_pred) > 1e-8 else (0.0, 1.0)

    results: Dict[str, Any] = {
        "overall": {
            "n_samples": int(len(y_true)),
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "spearman_rho": round(float(rho), 4) if not np.isnan(rho) else 0.0,
            "pearson_r": round(float(pearson), 4) if not np.isnan(pearson) else 0.0,
        }
    }

    # Subgroup: Clamped at floor (-30) vs Unclamped (> -30)
    clamped_mask = (y_true <= CENSORED_RISK_FLOOR + 1e-5)
    unclamped_mask = ~clamped_mask

    if clamped_mask.any():
        c_mae = float(mean_absolute_error(y_true[clamped_mask], y_pred[clamped_mask]))
        c_rmse = float(root_mean_squared_error(y_true[clamped_mask], y_pred[clamped_mask]))
        results["clamped_floor_subgroup"] = {
            "n_samples": int(clamped_mask.sum()),
            "pct_of_total": round(100.0 * clamped_mask.sum() / len(y_true), 2),
            "mae": round(c_mae, 4),
            "rmse": round(c_rmse, 4),
        }

    if unclamped_mask.any():
        u_true = y_true[unclamped_mask]
        u_pred = y_pred[unclamped_mask]
        u_mae = float(mean_absolute_error(u_true, u_pred))
        u_rmse = float(root_mean_squared_error(u_true, u_pred))
        u_rho, _ = spearmanr(u_true, u_pred)
        results["unclamped_subgroup"] = {
            "n_samples": int(unclamped_mask.sum()),
            "pct_of_total": round(100.0 * unclamped_mask.sum() / len(y_true), 2),
            "mae": round(u_mae, 4),
            "rmse": round(u_rmse, 4),
            "spearman_rho": round(float(u_rho), 4) if not np.isnan(u_rho) else 0.0,
        }

    # Subgroups by time_to_tca bins
    if time_to_tca is not None:
        t_tca = np.asarray(time_to_tca, dtype=float)
        bins = {
            "near_encounter_le_2d": t_tca <= 2.0,
            "mid_range_2d_to_5d": (t_tca > 2.0) & (t_tca <= 5.0),
            "early_warning_gt_5d": t_tca > 5.0,
        }
        time_breakdown = {}
        for b_name, b_mask in bins.items():
            if b_mask.any():
                b_mae = float(mean_absolute_error(y_true[b_mask], y_pred[b_mask]))
                b_rmse = float(root_mean_squared_error(y_true[b_mask], y_pred[b_mask]))
                b_rho, _ = spearmanr(y_true[b_mask], y_pred[b_mask])
                time_breakdown[b_name] = {
                    "n_samples": int(b_mask.sum()),
                    "mae": round(b_mae, 4),
                    "rmse": round(b_rmse, 4),
                    "spearman_rho": round(float(b_rho), 4) if not np.isnan(b_rho) else 0.0,
                }
        results["time_to_tca_subgroups"] = time_breakdown

    return results


def evaluate_classification(
    y_true_cls: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Evaluates high-risk event binary classification."""
    y_true_cls = np.asarray(y_true_cls, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    pr_auc = float(average_precision_score(y_true_cls, y_prob))
    roc_auc = float(roc_auc_score(y_true_cls, y_prob))
    p, r, f1, _ = precision_recall_fscore_support(y_true_cls, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_true_cls, y_pred)

    return {
        "threshold": threshold,
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "precision": round(float(p), 4),
        "recall": round(float(r), 4),
        "f1_score": round(float(f1), 4),
        "confusion_matrix": {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        },
        "positive_prevalence": round(float(np.mean(y_true_cls)), 4),
    }


def train_baseline_floor(y_train: pd.Series, y_val: pd.Series, floor_val: float = CENSORED_RISK_FLOOR) -> Dict[str, Any]:
    """Evaluates dummy floor baseline constantly predicting CENSORED_RISK_FLOOR (-30.0)."""
    y_pred_val = np.full(len(y_val), floor_val)
    return {
        "model_name": "DummyFloorRegressor",
        "floor_value": floor_val,
        "val_metrics": evaluate_regression(y_val.to_numpy(), y_pred_val),
    }


def train_baseline_dummy(y_train: pd.Series, y_val: pd.Series) -> Dict[str, Any]:
    """Trains dummy baseline predicting training mean."""
    mean_val = float(y_train.mean())
    y_pred_val = np.full(len(y_val), mean_val)
    return {
        "model_name": "DummyMeanRegressor",
        "train_mean": round(mean_val, 4),
        "val_metrics": evaluate_regression(y_val.to_numpy(), y_pred_val),
    }


def train_baseline_ridge(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    alpha: float = 100.0,
) -> Tuple[Pipeline, Dict[str, Any]]:
    """Trains regularized Ridge linear regression baseline with imputing and scaling."""
    num_cols = [c for c in X_train.columns if c != "c_object_type"]
    cat_cols = ["c_object_type"]

    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sca", StandardScaler())]), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])

    pipe = Pipeline([
        ("preprocessor", pre),
        ("regressor", Ridge(alpha=alpha, random_state=42)),
    ])

    pipe.fit(X_train, y_train)
    y_pred_val = pipe.predict(X_val)
    metrics = evaluate_regression(y_val.to_numpy(), y_pred_val)

    return pipe, {
        "model_name": "RidgeLinearBaseline",
        "alpha": alpha,
        "val_metrics": metrics,
    }


def train_risk_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    max_iter: int = 100,
    random_state: int = 42,
) -> Tuple[HistGradientBoostingRegressor, Dict[str, Any]]:
    """Trains primary tree-based gradient boosted regressor."""
    cat_cols = ["c_object_type"]
    reg = HistGradientBoostingRegressor(
        max_iter=max_iter,
        categorical_features=cat_cols,
        random_state=random_state,
        l2_regularization=0.1,
        min_samples_leaf=20,
    )
    reg.fit(X_train, y_train)
    y_pred_val = reg.predict(X_val)
    t_val = X_val["time_to_tca"].to_numpy() if "time_to_tca" in X_val.columns else None
    metrics = evaluate_regression(y_val.to_numpy(), y_pred_val, time_to_tca=t_val)

    return reg, {
        "model_name": "HistGradientBoostingRegressor",
        "max_iter": max_iter,
        "val_metrics": metrics,
    }


def train_risk_classifier(
    X_train: pd.DataFrame,
    y_train_cls: pd.Series,
    X_val: pd.DataFrame,
    y_val_cls: pd.Series,
    max_iter: int = 100,
    random_state: int = 42,
) -> Tuple[HistGradientBoostingClassifier, Dict[str, Any]]:
    """Trains high-risk encounter binary classifier."""
    cat_cols = ["c_object_type"]
    clf = HistGradientBoostingClassifier(
        max_iter=max_iter,
        class_weight="balanced",
        categorical_features=cat_cols,
        random_state=random_state,
        min_samples_leaf=20,
    )
    clf.fit(X_train, y_train_cls)
    y_prob_val = clf.predict_proba(X_val)[:, 1]
    metrics = evaluate_classification(y_val_cls.to_numpy(), y_prob_val, threshold=0.5)

    return clf, {
        "model_name": "HistGradientBoostingClassifier",
        "class_weight": "balanced",
        "val_metrics": metrics,
    }


def compute_feature_importance(
    model: Any,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    top_n: int = 20,
    n_samples: int = 3000,
    n_repeats: int = 3,
    random_state: int = 42,
) -> List[Dict[str, Any]]:
    """Computes permutation feature importance on validation sample."""
    sample_df = X_val.sample(n=min(n_samples, len(X_val)), random_state=random_state)
    sample_y = y_val.loc[sample_df.index]

    res = permutation_importance(
        model,
        sample_df,
        sample_y,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=2,
    )
    sorted_idx = res.importances_mean.argsort()[::-1][:top_n]
    feature_names = sample_df.columns.tolist()

    top_features = []
    for i in sorted_idx:
        top_features.append({
            "feature": feature_names[i],
            "importance_mean": round(float(res.importances_mean[i]), 6),
            "importance_std": round(float(res.importances_std[i]), 6),
        })
    return top_features


class MLRiskPredictor:
    """Production predictor providing inference, schema validation, and risk categorization."""

    def __init__(
        self,
        regressor: Optional[Any] = None,
        classifier: Optional[Any] = None,
        preprocessor: Optional[CDMEFeaturePreprocessor] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.regressor = regressor
        self.classifier = classifier
        self.preprocessor = preprocessor
        self.metadata = metadata or {}

    @classmethod
    def load(cls, model_dir: Union[str, Path] = DEFAULT_MODEL_DIR) -> "MLRiskPredictor":
        p = Path(model_dir)
        reg_file = p / "risk_regressor.joblib"
        clf_file = p / "risk_classifier.joblib"
        prep_file = p / "feature_preprocessor.joblib"
        meta_file = p / "model_metadata.json"

        if not reg_file.exists():
            raise FileNotFoundError(f"Regressor artifact not found at {reg_file}")

        regressor = joblib.load(reg_file)
        classifier = joblib.load(clf_file) if clf_file.exists() else None
        preprocessor = joblib.load(prep_file) if prep_file.exists() else None
        metadata = {}
        if meta_file.exists():
            with open(meta_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)

        return cls(regressor, classifier, preprocessor, metadata)

    def predict_cdm(self, cdm_record: Dict[str, Any]) -> Dict[str, Any]:
        """Validates input CDM record and computes predicted risk metrics."""
        if self.preprocessor is None or self.regressor is None:
            raise RuntimeError("MLRiskPredictor is not initialized with trained models.")

        df_row = self.preprocessor.transform_single(cdm_record)
        pred_log10_risk = float(self.regressor.predict(df_row)[0])

        clamped_log10 = max(CENSORED_RISK_FLOOR, min(0.0, pred_log10_risk))
        risk_prob = float(np.power(10.0, clamped_log10))

        high_risk_prob: Optional[float] = None
        if self.classifier is not None:
            high_risk_prob = float(self.classifier.predict_proba(df_row)[0, 1])
        # is_high_risk is always derived from the regressor threshold for consistency
        # with risk_category and action_recommendation. The classifier probability
        # is reported separately as high_risk_probability for ensemble context.
        is_high_risk: bool = bool(clamped_log10 >= RISK_ACTION_THRESHOLD)

        if clamped_log10 >= RISK_ACTION_THRESHOLD:
            risk_category = "CRITICAL"
            action_recommendation = "ESA Action Threshold exceeded (log10 >= -6.0). Collision avoidance maneuver recommended."
        elif clamped_log10 >= -10.0:
            risk_category = "ELEVATED"
            action_recommendation = "Elevated conjunction risk (-10.0 <= log10 < -6.0). Increase tracking frequency."
        else:
            risk_category = "LOW"
            action_recommendation = "Low conjunction risk (log10 < -10.0). Routine monitoring."

        return {
            "predicted_log10_risk": round(pred_log10_risk, 4),
            "clamped_log10_risk": round(clamped_log10, 4),
            "risk_probability": float(f"{risk_prob:.4e}"),
            "risk_category": risk_category,
            "is_high_risk": is_high_risk,
            "high_risk_probability": round(high_risk_prob, 4) if high_risk_prob is not None else None,
            "action_recommendation": action_recommendation,
            "model_version": self.metadata.get("model_version", "1.0.0-phase6"),
            "features_evaluated": len(self.preprocessor.feature_cols),
            "scientific_caveat": (
                "Prototype ML model trained on historical ESA Kelvins competition CDMs. "
                "Outputs provide relative risk estimates and do not represent certified operational "
                "collision probabilities for active on-orbit assets."
            ),
        }

    def predict_batch(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Scores a batch of CDM records efficiently."""
        return [self.predict_cdm(rec) for rec in records]
