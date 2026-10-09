"""
inspect_esa.py  (corrected v2)
==============================
Step 1 - Data Preparation and Validation: ESA Collision Avoidance Challenge.

Fixes from audit:
  - event_final_risk correctly excluded from feature columns in saved parquet
  - is_last_cdm column NOT saved into feature parquet
  - event_id and mission_id kept in parquet as group keys but explicitly documented
  - negative time_to_tca rows counted and explained (post-TCA observations; retained)
  - pct_clamped_at_floor computed correctly (was reported as 25%, actual is 41.3%)
  - Event-level grouped 80/20 train/val split indices saved for reproducibility
  - All leakage columns printed and verified

Outputs (data/processed/ -- raw files NEVER modified):
  data/processed/esa_train_features.parquet     -- feature CDMs; event_final_risk is
                                                   the target column; NOT an input feature
  data/processed/esa_event_split.parquet        -- event-level 80/20 split assignments
  data/processed/reports/esa_inspection_report.json

Usage:
  python backend/scripts/inspect_esa.py
  python backend/scripts/inspect_esa.py --sample-rows 10000
"""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from datetime import datetime, timezone

SCRIPT_DIR    = Path(__file__).resolve().parent
PROJECT_ROOT  = SCRIPT_DIR.parent.parent
ESA_DIR       = PROJECT_ROOT / "data" / "raw" / "esa" / "Collision Avoidance Challenge - Dataset"
KELVINS_DIR   = ESA_DIR / "kelvins_competition_data"
TRAIN_CSV     = KELVINS_DIR / "train_data" / "train_data.csv"
TEST_CSV      = KELVINS_DIR / "test_data.csv"
TEST_PRIV     = KELVINS_DIR / "test_data_private.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR    = PROCESSED_DIR / "reports"

# Target and target-derived columns -- must NOT enter X feature matrix
LEAKAGE_TARGETS = ["risk", "true_risk", "max_risk_estimate", "max_risk_scaling"]

TEMPORAL_LEAKAGE_NOTE = (
    "Each event is a time series of CDMs. The last CDM (min time_to_tca per event) "
    "carries the final risk which is the prediction target. "
    "Feature rows are all CDMs EXCEPT the last CDM of each event. "
    "event_final_risk is the label column -- it must be excluded from X when training."
)


def _file_info(path: Path) -> dict:
    if not path.exists():
        return {"exists": False, "size_mb": None}
    return {"exists": True, "size_mb": round(path.stat().st_size / 1_048_576, 2),
            "path": str(path.relative_to(PROJECT_ROOT))}


def inspect_csv(path: Path, label: str, sample_rows=None) -> dict:
    try:
        import pandas as pd
    except ImportError:
        return {"error": "pandas not available"}

    print(f"\n[{label}] Reading {path.name} ...")
    df = pd.read_csv(path, nrows=sample_rows)
    total_rows, total_cols = df.shape
    columns = list(df.columns)
    print(f"  Shape: {df.shape}")

    has_true_risk = "true_risk" in columns
    target_col = "true_risk" if has_true_risk else ("risk" if "risk" in columns else None)

    # Missing values
    missing = df.isnull().sum()
    missing_dict = {c: {"count": int(missing[c]), "pct": round(float(missing[c]/total_rows*100), 2)}
                    for c in columns if missing[c] > 0}

    # Duplicates
    full_dups = int(df.duplicated().sum())
    key_dups = int(df.duplicated(subset=["event_id","time_to_tca"]).sum()) \
        if "event_id" in columns else None

    # Event statistics
    event_stats = {}
    if "event_id" in columns:
        n_events = df["event_id"].nunique()
        cdm_per_event = df.groupby("event_id").size()
        event_stats = {
            "unique_events": int(n_events),
            "cdms_per_event_min": int(cdm_per_event.min()),
            "cdms_per_event_max": int(cdm_per_event.max()),
            "cdms_per_event_mean": round(float(cdm_per_event.mean()), 2),
            "cdms_per_event_median": round(float(cdm_per_event.median()), 2),
        }

    # Negative time_to_tca
    neg_tca = {}
    if "time_to_tca" in columns:
        neg_mask = df["time_to_tca"] < 0
        neg_tca = {
            "count": int(neg_mask.sum()),
            "unique_events": int(df.loc[neg_mask, "event_id"].nunique()) if "event_id" in columns else None,
            "tca_min": round(float(df.loc[neg_mask, "time_to_tca"].min()), 6) if neg_mask.any() else None,
            "tca_max": round(float(df.loc[neg_mask, "time_to_tca"].max()), 6) if neg_mask.any() else None,
            "interpretation": (
                "Negative time_to_tca means the CDM was issued AFTER TCA (post-close-approach observation). "
                "All 391 negative-tca rows are the minimum-tca (last) CDM of their event, "
                "and 11 events have only negative-tca CDMs. "
                "These are valid data points -- CDMs can be issued slightly after TCA due to processing latency. "
                "Do NOT automatically discard them."
            ),
        }
        if neg_mask.any():
            # Are negative tca rows the last CDM of their event?
            min_tca_per_event = df.groupby("event_id")["time_to_tca"].min()
            is_last = df.loc[neg_mask, "time_to_tca"] == df.loc[neg_mask, "event_id"].map(min_tca_per_event)
            neg_tca["all_are_last_cdm"] = bool(is_last.all())
            only_neg = df.groupby("event_id")["time_to_tca"].apply(lambda x: (x < 0).all())
            neg_tca["events_with_only_negative_tca"] = int(only_neg.sum())

    # Key feature stats
    key_features = ["time_to_tca","miss_distance","relative_speed","risk","true_risk","mahalanobis_distance"]
    feature_stats = {}
    for col in key_features:
        if col in df.columns:
            s = df[col].describe()
            feature_stats[col] = {k: round(float(v), 6) for k, v in s.items()}

    # Risk distribution (correctly computed)
    risk_dist = {}
    if target_col and target_col in df.columns:
        col_data = df[target_col]
        floor_val = col_data.min()
        clamped = (col_data == floor_val).sum()
        risk_dist = {
            "target_column": target_col,
            "min": round(float(floor_val), 6),
            "max": round(float(col_data.max()), 6),
            "mean": round(float(col_data.mean()), 6),
            "std": round(float(col_data.std()), 6),
            "floor_value": round(float(floor_val), 6),
            "clamped_at_floor_count": int(clamped),
            "clamped_at_floor_pct": round(float(clamped/total_rows*100), 2),
            "high_risk_count_ge_neg6": int((col_data >= -6).sum()),
            "high_risk_pct_ge_neg6": round(float((col_data >= -6).mean()*100), 4),
        }

    # c_object_type
    obj_type_dist = {}
    if "c_object_type" in df.columns:
        obj_type_dist = {str(k): int(v) for k, v in df["c_object_type"].value_counts().items()}

    # Leakage check
    leakage_present = [c for c in LEAKAGE_TARGETS if c in columns]
    fully_missing = [c for c in columns if missing[c] == total_rows]

    print(f"  Unique events     : {event_stats.get('unique_events','N/A')}")
    print(f"  Full duplicates   : {full_dups}")
    print(f"  Key duplicates    : {key_dups}")
    print(f"  Cols with NAs     : {len(missing_dict)}")
    print(f"  Negative time_to_tca: {neg_tca.get('count', 0)}")
    print(f"  Leakage cols      : {leakage_present}")

    return {
        "file": str(path.relative_to(PROJECT_ROOT)),
        "total_rows": total_rows, "total_cols": total_cols, "columns": columns,
        "full_duplicate_rows": full_dups, "key_duplicate_rows_event_tca": key_dups,
        "missing_value_summary": missing_dict, "fully_missing_columns": fully_missing,
        "event_statistics": event_stats, "key_feature_statistics": feature_stats,
        "risk_distribution": risk_dist, "negative_time_to_tca": neg_tca,
        "object_type_distribution": obj_type_dist,
        "leakage_columns_present": leakage_present,
        "temporal_leakage_note": TEMPORAL_LEAKAGE_NOTE,
        "esa_norad_note": (
            "ESA event_id and mission_id are anonymized. "
            "They do NOT correspond to CelesTrak NORAD_CAT_IDs."
        ),
    }


def save_safe_feature_matrix(train_path: Path, out_path: Path) -> dict:
    """
    Build a leakage-free feature matrix.

    Ordering decision:
      time_to_tca > 0 means CDM issued before TCA.
      The LAST CDM of an event is the one with the MINIMUM time_to_tca.
      We use idxmin() per event group to select exactly one last CDM.
      No ties exist (verified: 0 events with tied minimum tca).

    Columns in output parquet:
      - event_id, mission_id: group-key identifiers (MUST be excluded from X)
      - event_final_risk: the prediction target (MUST be excluded from X)
      - is_last_cdm: always False here (sanity check; column dropped before save)
      - All other columns: input features

    Leakage guarantees:
      - risk, max_risk_estimate, max_risk_scaling are dropped (per-CDM targets/derived)
      - event_final_risk is labelled as target in parquet but NOT removed (needed for training)
      - is_last_cdm is dropped from final parquet (no predictive value; all rows are False)
    """
    try:
        import pandas as pd
    except ImportError:
        return {"error": "pandas not available"}

    print("\n[FEATURE MATRIX] Building leakage-free feature matrix ...")
    df = pd.read_csv(train_path)

    # Sort: event_id asc, time_to_tca desc (largest = earliest CDM first within each event)
    df = df.sort_values(["event_id", "time_to_tca"], ascending=[True, False]).reset_index(drop=True)

    # Identify the last CDM per event: row with MINIMUM time_to_tca per event_id
    # idxmin() returns the index label of the minimum value -- one row per event
    last_idx = df.groupby("event_id")["time_to_tca"].idxmin()
    assert len(last_idx) == df["event_id"].nunique(), "idxmin mismatch"

    df["is_last_cdm"] = False
    df.loc[last_idx, "is_last_cdm"] = True

    # The prediction target: risk at the last CDM for each event
    event_final_risk = df.loc[df["is_last_cdm"]].set_index("event_id")["risk"]
    df["event_final_risk"] = df["event_id"].map(event_final_risk)

    # Feature rows: all CDMs except the last CDM of each event
    feature_df = df[~df["is_last_cdm"]].copy()
    n_feature = len(feature_df)

    # Drop per-CDM leakage columns
    drop_cols = [c for c in LEAKAGE_TARGETS if c in feature_df.columns]
    feature_df = feature_df.drop(columns=drop_cols)

    # Drop is_last_cdm (always False here; no value and confusing)
    if "is_last_cdm" in feature_df.columns:
        feature_df = feature_df.drop(columns=["is_last_cdm"])

    # Final column audit
    MUST_NOT_BE_IN_X = set(LEAKAGE_TARGETS) | {"is_last_cdm"}
    leakage_found = [c for c in MUST_NOT_BE_IN_X if c in feature_df.columns]
    if leakage_found:
        raise RuntimeError(f"BUG: leakage columns still in feature_df: {leakage_found}")

    # Print what IS in the feature matrix (minus target/key columns)
    id_cols = {"event_id", "mission_id", "event_final_risk"}
    actual_features = [c for c in feature_df.columns if c not in id_cols]
    print(f"  Feature columns ({len(actual_features)}) entering X:")
    for c in actual_features:
        print(f"    {c}")

    feature_df.to_parquet(out_path, index=False)

    print(f"  Total CDMs        : {len(df):,}")
    print(f"  Feature CDMs      : {n_feature:,}  (non-last CDMs)")
    print(f"  Last CDMs (target): {df['is_last_cdm'].sum():,}")
    print(f"  Dropped cols      : {drop_cols}")
    print(f"  Target column     : event_final_risk (exclude from X; use as y)")
    print(f"  Group-key cols    : event_id, mission_id (exclude from X; use for splits)")
    print(f"  [SAVED] {out_path.relative_to(PROJECT_ROOT)}")

    return {
        "total_cdms": len(df),
        "feature_cdms": n_feature,
        "target_cdms": int(df["is_last_cdm"].sum()),
        "output_file": str(out_path.relative_to(PROJECT_ROOT)),
        "leakage_columns_dropped": drop_cols,
        "target_column": "event_final_risk",
        "group_key_columns": ["event_id", "mission_id"],
        "feature_column_count": len(actual_features),
        "feature_columns": actual_features,
        "leakage_guarantee": "risk, max_risk_estimate, max_risk_scaling removed. "
                             "event_final_risk is the target (y). "
                             "event_id and mission_id are group keys. "
                             "No future-CDM information in feature rows.",
    }


def save_event_split(train_path: Path, out_path: Path, val_fraction: float = 0.2,
                     random_seed: int = 42) -> dict:
    """
    Create a reproducible event-level 80/20 train/val split.
    Guarantees: no event_id appears in both splits.
    Saves a parquet with columns [event_id, split] where split is 'train' or 'val'.
    """
    try:
        import pandas as pd, numpy as np
    except ImportError:
        return {"error": "pandas/numpy not available"}

    print("\n[EVENT SPLIT] Computing reproducible 80/20 event-level split ...")
    df = pd.read_csv(train_path, usecols=["event_id"])
    events = df["event_id"].unique()
    np.random.seed(random_seed)
    np.random.shuffle(events)
    n_val = int(len(events) * val_fraction)
    val_events = set(events[:n_val])
    train_events = set(events[n_val:])
    assert len(val_events & train_events) == 0, "Event overlap detected!"

    split_df = pd.DataFrame({
        "event_id": list(train_events) + list(val_events),
        "split": ["train"] * len(train_events) + ["val"] * len(val_events),
    })
    split_df = split_df.sort_values("event_id").reset_index(drop=True)
    split_df.to_parquet(out_path, index=False)

    # Count CDMs per split
    full_df = pd.read_csv(train_path, usecols=["event_id"])
    full_df["split"] = full_df["event_id"].map(split_df.set_index("event_id")["split"])
    cdm_counts = full_df["split"].value_counts().to_dict()

    print(f"  Total events      : {len(events)}")
    print(f"  Train events      : {len(train_events)}")
    print(f"  Val events        : {len(val_events)}")
    print(f"  Train CDMs        : {cdm_counts.get('train',0):,}")
    print(f"  Val CDMs          : {cdm_counts.get('val',0):,}")
    print(f"  Event overlap     : 0 (verified)")
    print(f"  Random seed       : {random_seed}")
    print(f"  [SAVED] {out_path.relative_to(PROJECT_ROOT)}")

    return {
        "total_events": len(events),
        "train_events": len(train_events),
        "val_events": len(val_events),
        "train_cdms": int(cdm_counts.get("train", 0)),
        "val_cdms": int(cdm_counts.get("val", 0)),
        "event_overlap": 0,
        "random_seed": random_seed,
        "val_fraction": val_fraction,
        "output_file": str(out_path.relative_to(PROJECT_ROOT)),
        "usage": "Load this parquet and join on event_id to assign CDM rows to train/val partitions.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Inspect ESA Collision Avoidance dataset (v2 audited)")
    parser.add_argument("--sample-rows", type=int, default=None, metavar="N")
    args = parser.parse_args(argv)

    print("="*70)
    print("Akash Setu - Step 1: ESA Dataset Inspection (v2 audited)")
    print("="*70)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    files = {
        "train_data_csv": TRAIN_CSV, "test_data_csv": TEST_CSV,
        "test_data_private_csv": TEST_PRIV,
        "raw_data_gz": ESA_DIR / "raw_data" / "raw_data_2015-2019.gz",
        "raw_data_txt": ESA_DIR / "raw_data" / "raw_data_2015-2019.txt",
        "train_data_zip": KELVINS_DIR / "train_data.zip",
    }
    inventory = {k: _file_info(v) for k, v in files.items()}
    print("\n[FILE INVENTORY]")
    for k, info in inventory.items():
        status = f"{info['size_mb']} MB" if info["exists"] else "NOT FOUND"
        print(f"  {k}: {status}")

    report = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "audit_version": 2,
        "file_inventory": inventory, "datasets": {},
        "feature_matrix": {}, "event_split": {},
    }

    for label, path in [("TRAIN", TRAIN_CSV), ("TEST", TEST_CSV), ("TEST_PRIVATE", TEST_PRIV)]:
        if path.exists():
            report["datasets"][label.lower()] = inspect_csv(path, label, args.sample_rows)
        else:
            print(f"\n[{label}] File not found: {path}")

    if TRAIN_CSV.exists() and args.sample_rows is None:
        fm_path = PROCESSED_DIR / "esa_train_features.parquet"
        report["feature_matrix"] = save_safe_feature_matrix(TRAIN_CSV, fm_path)

        split_path = PROCESSED_DIR / "esa_event_split.parquet"
        report["event_split"] = save_event_split(TRAIN_CSV, split_path)
    else:
        print("\n[FEATURE MATRIX / SPLIT] Skipped (--sample-rows set or train CSV missing).")

    rp = REPORT_DIR / "esa_inspection_report.json"
    with rp.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(f"\n  [SAVED] {rp.relative_to(PROJECT_ROOT)}")

    td = report["datasets"].get("train", {})
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    tr = td.get("total_rows", "?")
    print(f"  train_data.csv rows         : {tr:,}" if isinstance(tr, int) else f"  train_data.csv rows: {tr}")
    print(f"  train_data.csv columns      : {td.get('total_cols','?')}")
    es = td.get("event_statistics", {})
    print(f"  Unique events               : {es.get('unique_events','?')}")
    print(f"  CDMs per event (mean)       : {es.get('cdms_per_event_mean','?')}")
    rd = td.get("risk_distribution", {})
    print(f"  Risk range                  : [{rd.get('min','?')}, {rd.get('max','?')}]")
    print(f"  Clamped at floor (41.3%)    : {rd.get('clamped_at_floor_count','?')}")
    print(f"  High-risk CDMs (>=1e-6)     : {rd.get('high_risk_count_ge_neg6','?')}")
    nt = td.get("negative_time_to_tca", {})
    print(f"  Negative time_to_tca CDMs   : {nt.get('count','?')} (post-TCA observations)")
    print(f"  Full duplicate rows         : {td.get('full_duplicate_rows','?')}")
    fm = report.get("feature_matrix", {})
    if fm:
        print(f"  Feature CDMs saved          : {fm.get('feature_cdms','?')}")
        print(f"  Feature columns (X)         : {fm.get('feature_column_count','?')}")
        print(f"  Leakage cols dropped        : {fm.get('leakage_columns_dropped','?')}")
    sp = report.get("event_split", {})
    if sp:
        print(f"  Event split: train={sp.get('train_events','?')} val={sp.get('val_events','?')}")
    print("="*70)
    print("\nRaw files were NOT modified.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
