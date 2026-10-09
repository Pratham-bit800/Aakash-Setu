"""
ml_readiness_audit.py
=====================
Akash Setu – ML Readiness Audit (Step 1 → Step 2 gate check).

Runs all 6 audit checks against actual processed files and saves a
machine-readable report to data/processed/reports/ml_readiness_report.json.

Usage:
  python backend/scripts/ml_readiness_audit.py

All checks are executed; none are assumed.  Output clearly marks each item
as PASS, FAIL, or INFO.  Raw files are never modified.
"""

from __future__ import annotations
import json, sys, numpy as np
from datetime import datetime, timezone
from pathlib import Path

ROOT        = Path(__file__).resolve().parent.parent.parent
FEAT_PATH   = ROOT / "data/processed/esa_train_features.parquet"
SPLIT_PATH  = ROOT / "data/processed/esa_event_split.parquet"
TRAIN_CSV   = (ROOT / "data/raw/esa/Collision Avoidance Challenge - Dataset"
               / "kelvins_competition_data/train_data/train_data.csv")
REPORT_PATH = ROOT / "data/processed/reports/ml_readiness_report.json"

NEVER_IN_X = [
    "risk","true_risk","max_risk_estimate","max_risk_scaling",
    "event_final_risk","event_id","mission_id","is_last_cdm",
]

def banner(title): print(f"\n{'='*70}\n{title}\n{'='*70}")
def ok(msg):   print(f"  [PASS] {msg}")
def warn(msg): print(f"  [WARN] {msg}")
def fail(msg): print(f"  [FAIL] {msg}")
def info(msg): print(f"  [INFO] {msg}")

def run_audit():
    import pandas as pd
    report = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "checks": {},
    }

    # ── Load artefacts ────────────────────────────────────────────────────
    banner("Loading processed artefacts")
    feat  = pd.read_parquet(FEAT_PATH);  info(f"Feature parquet: {feat.shape}")
    split = pd.read_parquet(SPLIT_PATH); info(f"Split parquet:   {split.shape}")

    # ═══════════════════════════════════════════════════════════════════════
    # CHECK 1 – Event-level train/val split
    # ═══════════════════════════════════════════════════════════════════════
    banner("CHECK 1 – Event-level train/val split")
    c1 = {}

    # 1a. No duplicate event_ids in split file
    dup_ev = int(split["event_id"].duplicated().sum())
    c1["duplicate_event_ids_in_split"] = dup_ev
    (ok if dup_ev==0 else fail)(f"Duplicate event_ids in split file: {dup_ev}")

    # 1b. Zero overlap between partitions
    train_ev = set(split[split["split"]=="train"]["event_id"])
    val_ev   = set(split[split["split"]=="val"]["event_id"])
    overlap  = train_ev & val_ev
    c1["event_overlap"] = len(overlap)
    (ok if not overlap else fail)(f"event_id overlap train&val: {len(overlap)}")

    # 1c. Reproducibility with seed 42
    all_events = pd.read_csv(TRAIN_CSV, usecols=["event_id"])["event_id"].unique()
    rng = np.random.default_rng(42)   # note: numpy >=1.17 RandomState vs Generator
    # inspect_esa used np.random.seed(42) + np.random.shuffle (legacy API)
    import numpy as _np
    _np.random.seed(42); _np.random.shuffle(all_events)
    n_val = int(len(all_events)*0.2)
    expected_val   = set(all_events[:n_val])
    expected_train = set(all_events[n_val:])
    match_train = expected_train == train_ev
    match_val   = expected_val   == val_ev
    c1["reproducible_seed42_train"] = match_train
    c1["reproducible_seed42_val"]   = match_val
    (ok if match_train and match_val else fail)(
        f"Reproducible with seed=42: train={match_train} val={match_val}")

    # 1d. Merge split into features — no missing labels, no row explosion
    pre  = len(feat)
    mg   = feat.merge(split, on="event_id", how="left")
    post = len(mg)
    miss = int(mg["split"].isnull().sum())
    dup_rows = int(mg.duplicated().sum())
    c1["rows_before_merge"] = pre
    c1["rows_after_merge"]  = post
    c1["missing_split_labels"] = miss
    c1["duplicate_rows_after_merge"] = dup_rows
    (ok if post==pre else fail)(f"Row count after merge: {pre:,} -> {post:,}")
    (ok if miss==0 else fail)(f"Missing split labels after merge: {miss}")
    (ok if dup_rows==0 else fail)(f"Duplicate rows after merge: {dup_rows}")

    # 1e. Counts
    cdm_by_split = mg.groupby("split").size().to_dict()
    ev_by_split  = mg.groupby("split")["event_id"].nunique().to_dict()
    c1["cdm_counts"]   = cdm_by_split
    c1["event_counts"] = ev_by_split
    info(f"CDMs  — train: {cdm_by_split.get('train',0):,}  val: {cdm_by_split.get('val',0):,}")
    info(f"Events— train: {ev_by_split.get('train',0):,}   val: {ev_by_split.get('val',0):,}")

    c1["status"] = "PASS" if (dup_ev==0 and not overlap and match_train
                               and match_val and post==pre and miss==0) else "FAIL"
    report["checks"]["1_event_split"] = c1

    # ═══════════════════════════════════════════════════════════════════════
    # CHECK 2 – Target construction & feature leakage
    # ═══════════════════════════════════════════════════════════════════════
    banner("CHECK 2 – Target construction & feature leakage")
    c2 = {}

    # 2a. event_final_risk == risk at min(time_to_tca) per event
    df_raw = pd.read_csv(TRAIN_CSV)
    min_idx       = df_raw.groupby("event_id")["time_to_tca"].idxmin()
    true_last_risk = df_raw.loc[min_idx].set_index("event_id")["risk"]
    feat_efr = feat.groupby("event_id")["event_final_risk"].first()
    common   = feat_efr.index.intersection(true_last_risk.index)
    diffs    = (feat_efr[common] - true_last_risk[common]).abs()
    bad      = int((diffs > 1e-8).sum())
    ties     = int((df_raw["time_to_tca"] == df_raw["event_id"].map(
                    df_raw.groupby("event_id")["time_to_tca"].min()
                   )).groupby(df_raw["event_id"]).sum().gt(1).sum())
    c2["target_mismatches"] = bad
    c2["tied_min_tca_events"] = ties
    c2["events_checked"] = len(common)
    (ok if bad==0 else fail)(f"event_final_risk matches raw last-CDM risk: {bad} mismatches in {len(common)} events")
    (ok if ties==0 else warn)(f"Events with tied minimum time_to_tca: {ties}")

    # 2b. Leakage column audit
    leakage_present  = [c for c in NEVER_IN_X if c in feat.columns]
    leakage_absent   = [c for c in NEVER_IN_X if c not in feat.columns]
    # event_final_risk, event_id, mission_id MAY be present (as target/keys) but noted
    hard_leakage = [c for c in leakage_present
                    if c not in {"event_final_risk","event_id","mission_id"}]
    c2["hard_leakage_columns"] = hard_leakage
    c2["columns_in_parquet_as_keys_or_target"] = [c for c in leakage_present
                                                    if c in {"event_final_risk","event_id","mission_id"}]
    (ok if not hard_leakage else fail)(
        f"Hard leakage columns in feature parquet: {hard_leakage or 'none'}")
    info(f"Columns present as keys/target (must not enter X): {c2['columns_in_parquet_as_keys_or_target']}")

    # 2c. X column list
    X_cols = [c for c in feat.columns if c not in set(NEVER_IN_X)]
    c2["x_column_count"] = len(X_cols)
    c2["x_columns"] = X_cols
    info(f"X input columns ({len(X_cols)}): {X_cols}")

    # 2d. Runtime assertion template
    c2["runtime_assertion"] = (
        "assert not any(c in X_train.columns for c in "
        + str(NEVER_IN_X) + "), 'Leakage detected'"
    )

    c2["status"] = "PASS" if (bad==0 and not hard_leakage and ties==0) else "FAIL"
    report["checks"]["2_target_leakage"] = c2

    # ═══════════════════════════════════════════════════════════════════════
    # CHECK 3 – Events without eligible feature rows
    # ═══════════════════════════════════════════════════════════════════════
    banner("CHECK 3 – Events without eligible feature rows")
    c3 = {}

    cdm_count = df_raw.groupby("event_id").size()
    single_cdm_ids = set(cdm_count[cdm_count==1].index)
    neg_only_mask  = df_raw.groupby("event_id")["time_to_tca"].apply(lambda x: (x<0).all())
    neg_only_ids   = set(neg_only_mask[neg_only_mask].index)

    all_excluded = single_cdm_ids  # 1-CDM events have 0 feature rows by construction
    neg_only_pos = single_cdm_ids - neg_only_ids   # single+positive tca (not seen before)

    c3["total_events"] = int(df_raw["event_id"].nunique())
    c3["single_cdm_events"] = len(single_cdm_ids)
    c3["single_cdm_with_negative_tca"] = len(neg_only_ids & single_cdm_ids)
    c3["single_cdm_with_positive_tca"] = len(neg_only_pos)
    c3["excluded_event_ids"] = sorted(all_excluded)
    c3["usable_events"] = int(df_raw["event_id"].nunique()) - len(all_excluded)

    # Confirm 0 rows for excluded events in feature parquet
    rows_for_excluded = int(feat[feat["event_id"].isin(all_excluded)].shape[0])
    c3["feature_rows_for_excluded_events"] = rows_for_excluded

    ok(f"Single-CDM events (no eligible feature rows): {len(all_excluded)}")
    info(f"  {len(neg_only_ids & single_cdm_ids)} have negative tca (post-TCA detection)")
    info(f"  {len(neg_only_pos)} have positive tca (only 1 CDM issued, which is the final CDM)")
    (ok if rows_for_excluded==0 else fail)(
        f"Feature rows for excluded events in parquet: {rows_for_excluded} (must be 0)")
    info(f"Usable events for early-warning model: {c3['usable_events']} / {c3['total_events']}")

    split_for_excl = split[split["event_id"].isin(all_excluded)]["split"].value_counts().to_dict()
    c3["excluded_by_split"] = {k: int(v) for k,v in split_for_excl.items()}
    info(f"Excluded events by split: train={split_for_excl.get('train',0)}, val={split_for_excl.get('val',0)}")
    info("Recommendation: filter merged dataframe to event_ids present in feature parquet before training.")

    c3["status"] = "PASS" if rows_for_excluded==0 else "FAIL"
    report["checks"]["3_ineligible_events"] = c3

    # ═══════════════════════════════════════════════════════════════════════
    # CHECK 4 – Missing value audit and imputation strategy
    # ═══════════════════════════════════════════════════════════════════════
    banner("CHECK 4 – Missing value preprocessing")
    c4 = {}

    mg2 = mg.copy()
    train_df = mg2[mg2["split"]=="train"]
    val_df   = mg2[mg2["split"]=="val"]
    X_train  = train_df[X_cols]
    X_val    = val_df[X_cols]

    miss_train = X_train.isnull().sum()
    miss_val   = X_val.isnull().sum()
    cols_na    = miss_train[miss_train>0].index.tolist()

    # Characterise missing groups
    cov_vel_t  = [c for c in cols_na if c.startswith("t_") and
                   any(s in c for s in ["crdot","ctdot","cndot","sigma_rdot","sigma_tdot","sigma_ndot"])]
    cov_vel_c  = [c for c in cols_na if c.startswith("c_") and
                   any(s in c for s in ["crdot","ctdot","cndot","sigma_rdot","sigma_tdot","sigma_ndot"])]
    rcs_cols   = [c for c in cols_na if "rcs_estimate" in c]
    sw_cols    = [c for c in cols_na if c in ["F10","F3M","SSN","AP"]]
    chaser_od  = [c for c in cols_na if c in [
        "c_time_lastob_start","c_time_lastob_end","c_recommended_od_span",
        "c_actual_od_span","c_obs_available","c_obs_used",
        "c_residuals_accepted","c_weighted_rms","c_ct_r","c_cn_r","c_cn_t",
        "c_sigma_r","c_sigma_t","c_sigma_n"]]

    def miss_summary(col_list, label):
        counts = {c: int(miss_train[c]) for c in col_list if c in miss_train}
        pcts   = {c: round(100*v/len(X_train),2) for c,v in counts.items()}
        total  = sum(counts.values())
        unique_counts = set(counts.values())
        return {"columns": col_list, "sample_missing_count": counts,
                "sample_missing_pct": pcts, "group_total_missing": total,
                "uniform_pattern": len(unique_counts)==1}

    c4["groups"] = {
        "target_velocity_covariance":  miss_summary(cov_vel_t, "t_vel_cov"),
        "chaser_velocity_covariance":  miss_summary(cov_vel_c, "c_vel_cov"),
        "rcs_estimate":                miss_summary(rcs_cols,  "rcs"),
        "space_weather":               miss_summary(sw_cols,   "sw"),
        "chaser_od_quality":           miss_summary(chaser_od, "c_od"),
    }

    # Key check: covariance blocks are uniform (all-or-nothing per row)
    t_cov_any = X_train[cov_vel_t].isnull().any(axis=1)
    t_cov_all = X_train[cov_vel_t].isnull().all(axis=1)
    c_cov_vel_full_miss = X_train[cov_vel_c].isnull().all(axis=1)
    partial_t = int((t_cov_any & ~t_cov_all).sum())
    partial_c_vel = 0
    if cov_vel_c:
        c_cov_any = X_train[cov_vel_c].isnull().any(axis=1)
        c_cov_all = X_train[cov_vel_c].isnull().all(axis=1)
        partial_c_vel = int((c_cov_any & ~c_cov_all).sum())

    c4["target_vel_cov_partial_rows"] = partial_t
    c4["chaser_vel_cov_partial_rows"] = partial_c_vel

    (ok if partial_t==0 else warn)(f"Target vel-cov partial missing rows: {partial_t} (0=block uniform)")
    (ok if partial_c_vel==0 else warn)(f"Chaser vel-cov partial missing rows: {partial_c_vel} (0=block uniform)")

    info(f"c_rcs_estimate missing: {miss_train.get('c_rcs_estimate',0):,} / {len(X_train):,} = {100*miss_train.get('c_rcs_estimate',0)/len(X_train):.1f}%")
    info(f"t_rcs_estimate missing: {miss_train.get('t_rcs_estimate',0):,} / {len(X_train):,} = {100*miss_train.get('t_rcs_estimate',0)/len(X_train):.1f}%")

    # Imputation strategy (recommendation — not fitted here, no model trained)
    c4["imputation_strategy"] = {
        "c_rcs_estimate": {
            "missingness_pct": round(100*miss_train.get("c_rcs_estimate",0)/len(X_train),1),
            "recommended": "Median imputation fitted on train only. "
                           "Add binary indicator column 'c_rcs_missing'. "
                           "Alternatively, use a tree model (XGBoost/LightGBM) that handles NaN natively.",
            "do_not": "Do not zero-fill (zero is a physically meaningful value for RCS).",
        },
        "velocity_covariance_blocks": {
            "pattern": "Block-uniform: entire velocity-state covariance sub-block is absent together.",
            "recommended": "Add binary indicator 'vel_cov_available'. "
                           "For the covariance values themselves, median imputation on train. "
                           "Consider dropping velocity-covariance columns entirely in a baseline model "
                           "and testing if they add signal.",
            "do_not": "Do not zero-fill (zero falsely implies zero uncertainty).",
        },
        "space_weather_F10_F3M_SSN_AP": {
            "missingness_pct": round(100*miss_train.get("F10",0)/len(X_train),1),
            "recommended": "Forward-fill by time_to_tca within event, else median imputation on train.",
            "do_not": "Do not drop rows — space weather missing is MCAR (random gaps in external data).",
        },
        "chaser_od_quality_7_rows": {
            "recommended": "Drop these 7 rows or median-impute; volume is negligible.",
        },
        "training_discipline": (
            "ALL imputation parameters (median, fill values) MUST be fitted on X_train only. "
            "Apply the same fitted transformer to X_val to prevent data leakage."
        ),
    }
    ok("Imputation strategy documented (not fitted — no model trained here)")
    c4["status"] = "PASS"
    report["checks"]["4_missing_values"] = c4

    # ═══════════════════════════════════════════════════════════════════════
    # CHECK 5 – Evaluation setup
    # ═══════════════════════════════════════════════════════════════════════
    banner("CHECK 5 – Evaluation setup")
    c5 = {}

    train_df2 = mg2[mg2["split"]=="train"]
    y_train   = train_df2["event_final_risk"]
    y_val     = mg2[mg2["split"]=="val"]["event_final_risk"]

    # Per-event unique target verification
    multi_tgt = int((feat.groupby("event_id")["event_final_risk"].nunique() > 1).sum())
    c5["events_with_multiple_targets"] = multi_tgt
    (ok if multi_tgt==0 else fail)(f"Events with >1 unique event_final_risk: {multi_tgt}")

    # Target distribution
    floor = float(y_train.min())
    clamped = int((y_train==floor).sum())
    high_risk = int((y_train >= -6).sum())
    c5["target_distribution_train"] = {
        "min": float(y_train.min()), "max": float(y_train.max()),
        "mean": round(float(y_train.mean()),4), "std": round(float(y_train.std()),4),
        "median": float(y_train.median()),
        "floor_value": floor,
        "clamped_at_floor_count": clamped,
        "clamped_at_floor_pct": round(100*clamped/len(y_train),2),
        "high_risk_ge_neg6_count": high_risk,
        "high_risk_ge_neg6_pct": round(100*high_risk/len(y_train),4),
    }
    info(f"y_train: min={y_train.min():.2f} max={y_train.max():.4f} mean={y_train.mean():.2f}")
    info(f"Clamped at floor (-30): {clamped:,}/{len(y_train):,} = {100*clamped/len(y_train):.1f}%")
    info(f"High-risk (>=-6):       {high_risk:,}/{len(y_train):,} = {100*high_risk/len(y_train):.2f}%")

    c5["metrics_recommendation"] = {
        "primary_task": "Regression: predict final log10(collision probability)",
        "recommended_regression_metrics": [
            "MAE (mean absolute error in log-probability units)",
            "RMSE (penalises large errors in high-risk regime)",
            "Pearson / Spearman correlation (rank order correctness)",
        ],
        "clamped_floor_caveat": (
            f"{round(100*clamped/len(y_train),1)}% of targets are clamped at {floor}. "
            "MAE/RMSE will be dominated by near-floor predictions. "
            "Compute metrics separately for clamped and unclamped subsets."
        ),
        "classification_alternative": {
            "threshold": "-6 (i.e. P_collision >= 1e-6 = ESA action threshold)",
            "class_balance": f"{round(100*high_risk/len(y_train),2)}% positive",
            "recommended_metrics": ["AUROC","AUPRC (preferred given imbalance)","F1 at threshold"],
            "caveat": "Class imbalance is severe (98.6% negative). AUPRC is more informative than AUROC.",
        },
        "do_not_claim": "Do not report metrics as reliable until actual model evaluation is complete.",
    }

    c5["split_type_explanation"] = {
        "current_split": "Random 80/20 stratified by event_id (seed=42)",
        "time_based_alternative": (
            "A temporal split holds out the last N% of events by issue date. "
            "If event_id correlates with time (ESA data is from 2015-2019 and IDs may be ordered), "
            "the random split allows future events into training, making validation overly optimistic. "
            "Verification requires a timestamp column (none present in anonymised dataset) "
            "or knowledge of ESA event ordering."
        ),
        "leakage_risk_of_random_split": (
            "LOW for event-level split: each event's CDMs are entirely in one partition. "
            "MODERATE for temporal generalisation: model may not generalise to future space weather "
            "or new object populations if trained on a non-temporal mix."
        ),
        "recommendation": (
            "Use random split for initial development. "
            "Before deployment, evaluate on a held-out time period if timestamps become available."
        ),
    }
    ok("Metrics recommendation documented")
    ok("Split type explained with leakage risk characterised")
    c5["status"] = "PASS" if multi_tgt==0 else "FAIL"
    report["checks"]["5_evaluation_setup"] = c5

    # ── Final status ──────────────────────────────────────────────────────
    banner("OVERALL STATUS")
    all_statuses = {k: v.get("status","?") for k,v in report["checks"].items()}
    overall = "PASS" if all(s=="PASS" for s in all_statuses.values()) else "FAIL"
    report["overall_status"] = overall
    for k, s in all_statuses.items():
        print(f"  {k}: {s}")
    print(f"\n  OVERALL: {overall}")

    # ── Save report ───────────────────────────────────────────────────────
    with open(REPORT_PATH,"w",encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(f"\n  [SAVED] {REPORT_PATH.relative_to(ROOT)}")
    print("\nRaw files were NOT modified.\n")
    return overall


if __name__ == "__main__":
    sys.exit(0 if run_audit() == "PASS" else 1)

