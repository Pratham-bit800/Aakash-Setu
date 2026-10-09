"""
validate_celestrak.py  (v2 audited)
=====================================
Step 1 - Data Preparation and Validation: CelesTrak satellite datasets.

Changes from audit:
  - SGP4 now tests ALL valid records by default (not just a sample).
    Use --sgp4-sample N to limit to N records.
  - Added explicit disclaimer: propagation success at epoch ONLY proves the
    TLE is numerically processable by SGP4. It does NOT validate physical
    accuracy, applicability to highly elliptical orbits, or long-term prediction.
  - _sgp4_ok and _sgp4_error columns added to the output parquet per-record.

Outputs (data/processed/ -- raw files NEVER modified):
  data/processed/celestrak_validated.parquet
  data/processed/celestrak_validated.csv
  data/processed/reports/celestrak_validation_report.json

Usage:
  python backend/scripts/validate_celestrak.py              # tests all records
  python backend/scripts/validate_celestrak.py --sgp4-sample 150
  python backend/scripts/validate_celestrak.py --stale-days 30
"""

from __future__ import annotations
import argparse, json, math as _math, sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
RAW_DIR      = PROJECT_ROOT / "data" / "raw" / "celestrak"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR   = PROCESSED_DIR / "reports"

REQUIRED_FIELDS = [
    "NORAD_CAT_ID","OBJECT_NAME","EPOCH","MEAN_MOTION","ECCENTRICITY",
    "INCLINATION","RA_OF_ASC_NODE","ARG_OF_PERICENTER","MEAN_ANOMALY",
    "BSTAR","MEAN_MOTION_DOT","MEAN_MOTION_DDOT",
]
NUMERIC_FIELDS = [
    "MEAN_MOTION","ECCENTRICITY","INCLINATION","RA_OF_ASC_NODE",
    "ARG_OF_PERICENTER","MEAN_ANOMALY","BSTAR","MEAN_MOTION_DOT",
    "MEAN_MOTION_DDOT","NORAD_CAT_ID",
]
RANGE_CHECKS = {
    "MEAN_MOTION":       (0.0, 17.0),
    "ECCENTRICITY":      (0.0,  1.0),
    "INCLINATION":       (0.0, 180.0),
    "RA_OF_ASC_NODE":    (0.0, 360.0),
    "ARG_OF_PERICENTER": (0.0, 360.0),
    "MEAN_ANOMALY":      (0.0, 360.0),
}
STALE_DAYS_DEFAULT = 14
_DEG2RAD = _math.pi / 180.0
_REVDAY2RADMIN = 2 * _math.pi / 1440.0

SGP4_ACCURACY_DISCLAIMER = (
    "Propagation success at epoch (tsince=0) means the TLE is numerically processable. "
    "It does NOT validate: (1) long-term accuracy, (2) suitability for highly elliptic "
    "or deep-space orbits (use SDP4 for those), or (3) physical correctness of the elements. "
    "Failed records indicate an SGP4 internal error code != 0 or exception during init/propagation."
)


def _load_json_file(path):
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, list):
        return data
    raise ValueError(f"Unexpected JSON type {type(data)} in {path.name}")


def _parse_epoch(epoch_str):
    if not isinstance(epoch_str, str): return None
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f","%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S.%f","%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(epoch_str, fmt).replace(tzinfo=timezone.utc)
        except ValueError: pass
    return None


def _epoch_to_sgp4_days(epoch_str):
    dt = _parse_epoch(epoch_str)
    if dt is None: raise ValueError(f"Cannot parse epoch: {epoch_str!r}")
    y, m, d = dt.year, dt.month, dt.day
    h = dt.hour + dt.minute/60.0 + dt.second/3600.0 + dt.microsecond/3600e6
    if m <= 2: y -= 1; m += 12
    A = int(y/100); B = 2 - A + int(A/4)
    jd = int(365.25*(y+4716)) + int(30.6001*(m+1)) + d + h/24 + B - 1524.5
    return jd - 2433281.5


def _check_sgp4(records, n):
    """
    Test SGP4 propagation for up to n records.
    Returns per-record results and aggregate counts.
    """
    aggregate = {
        "library": "sgp4", "sample_size": 0,
        "success": 0, "failed": 0,
        "failed_ids": [], "error": None,
        "accuracy_disclaimer": SGP4_ACCURACY_DISCLAIMER,
    }
    per_record = {}  # norad -> {"ok": bool, "error_code": int|None, "error_msg": str|None}

    try:
        from sgp4.api import Satrec, WGS72
    except ImportError as exc:
        aggregate["error"] = f"sgp4 not installed: {exc}"; return aggregate, per_record

    sample = records[:n]; aggregate["sample_size"] = len(sample)
    for rec in sample:
        norad = rec.get("NORAD_CAT_ID", "?")
        norad_key = int(norad) if str(norad).isdigit() else str(norad)
        try:
            sat = Satrec()
            sat.sgp4init(
                WGS72, "i", int(norad),
                _epoch_to_sgp4_days(str(rec["EPOCH"])),
                float(rec["BSTAR"]), float(rec["MEAN_MOTION_DOT"]), float(rec["MEAN_MOTION_DDOT"]),
                float(rec["ECCENTRICITY"]),
                float(rec["ARG_OF_PERICENTER"]) * _DEG2RAD,
                float(rec["INCLINATION"]) * _DEG2RAD,
                float(rec["MEAN_ANOMALY"]) * _DEG2RAD,
                float(rec["MEAN_MOTION"]) * _REVDAY2RADMIN,
                float(rec["RA_OF_ASC_NODE"]) * _DEG2RAD,
            )
            e, r, v = sat.sgp4(sat.jdsatepoch, sat.jdsatepochF)
            if e != 0:
                aggregate["failed"] += 1
                aggregate["failed_ids"].append(norad_key)
                per_record[norad_key] = {"ok": False, "error_code": e, "error_msg": None, "name": rec.get("OBJECT_NAME","?")}
            else:
                aggregate["success"] += 1
                per_record[norad_key] = {"ok": True, "error_code": 0, "error_msg": None, "name": rec.get("OBJECT_NAME","?")}
        except Exception as exc:
            aggregate["failed"] += 1
            aggregate["failed_ids"].append(norad_key)
            per_record[norad_key] = {"ok": False, "error_code": None, "error_msg": str(exc), "name": rec.get("OBJECT_NAME","?")}

    return aggregate, per_record


def validate_dataset(records, source_file, stale_days):
    now_utc = datetime.now(tz=timezone.utc)
    stale_threshold = stale_days * 86400
    stats = {
        "source": source_file, "total": len(records),
        "missing_required_field": [], "bad_epoch": [], "stale_epoch": [],
        "non_numeric": [], "out_of_range": [], "duplicate_norad": [],
        "valid_count": 0, "invalid_count": 0, "invalid_indices": [],
    }
    seen_norads = {}; flagged = set()
    for idx, rec in enumerate(records):
        issues = []; norad_raw = rec.get("NORAD_CAT_ID")
        for field in REQUIRED_FIELDS:
            if field not in rec or rec[field] is None or rec[field] == "":
                issues.append(f"missing_required:{field}")
                stats["missing_required_field"].append({"index":idx,"field":field,"norad":norad_raw})
        epoch_str = rec.get("EPOCH","")
        epoch_dt = _parse_epoch(epoch_str)
        if epoch_dt is None:
            issues.append("bad_epoch")
            stats["bad_epoch"].append({"index":idx,"epoch":epoch_str,"norad":norad_raw})
        else:
            age_s = (now_utc - epoch_dt).total_seconds()
            if age_s > stale_threshold:
                age_d = age_s/86400
                issues.append(f"stale_epoch:{age_d:.1f}d")
                stats["stale_epoch"].append({"index":idx,"age_days":round(age_d,2),"norad":norad_raw})
        for field in NUMERIC_FIELDS:
            val = rec.get(field)
            if val is not None and val != "":
                try: float(val)
                except (TypeError, ValueError):
                    issues.append(f"non_numeric:{field}={val!r}")
                    stats["non_numeric"].append({"index":idx,"field":field,"value":val,"norad":norad_raw})
        for field,(lo,hi) in RANGE_CHECKS.items():
            val = rec.get(field)
            if val is not None and val != "":
                try:
                    fval = float(val)
                    if not (lo <= fval <= hi):
                        issues.append(f"out_of_range:{field}={fval}")
                        stats["out_of_range"].append({"index":idx,"field":field,"value":fval,"allowed":[lo,hi],"norad":norad_raw})
                except (TypeError, ValueError): pass
        if norad_raw is not None:
            try:
                nid = int(norad_raw)
                if nid in seen_norads:
                    issues.append(f"duplicate_norad:first_at={seen_norads[nid]}")
                    stats["duplicate_norad"].append({"norad":nid,"indices":[seen_norads[nid],idx]})
                else:
                    seen_norads[nid] = idx
            except (TypeError, ValueError): pass
        rec["_issues"] = issues; rec["_source"] = source_file
        if any(not i.startswith("stale_epoch") for i in issues):
            flagged.add(idx)
    stats["valid_count"] = sum(1 for i in range(len(records)) if i not in flagged)
    stats["invalid_count"] = len(flagged); stats["invalid_indices"] = sorted(flagged)
    return {
        "stats": stats,
        "valid_records": [r for i,r in enumerate(records) if i not in flagged],
        "invalid_records": [r for i,r in enumerate(records) if i in flagged],
        "all_records": records,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Validate CelesTrak orbital data (v2 audited)")
    parser.add_argument("--sgp4-sample", type=int, default=None, metavar="N",
                        help="Limit SGP4 test to N records. Default: ALL valid records.")
    parser.add_argument("--stale-days", type=int, default=STALE_DAYS_DEFAULT, metavar="DAYS")
    args = parser.parse_args(argv)

    print("="*70)
    print("Akash Setu - Step 1: CelesTrak Data Validation (v2 audited)")
    print("="*70)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    files = {
        "active_satellites": RAW_DIR / "active_satellites.json",
        "stations": RAW_DIR / "stations.json",
    }
    all_results = {}
    for name, path in files.items():
        if not path.exists():
            print(f"  [SKIP] {path.name} not found."); continue
        print(f"\n[{name}] Loading {path.name} ...")
        records = _load_json_file(path)
        print(f"  Loaded {len(records):,} records.")
        result = validate_dataset(records, path.name, args.stale_days)
        s = result["stats"]
        print(f"  Valid             : {s['valid_count']:,}")
        print(f"  Invalid (flagged) : {s['invalid_count']:,}  (retained with _issues flags)")
        print(f"  Stale epoch warns : {len(s['stale_epoch']):,}  (threshold={args.stale_days}d)")
        print(f"  Missing fields    : {len(s['missing_required_field'])}")
        print(f"  Bad epochs        : {len(s['bad_epoch'])}")
        print(f"  Non-numeric       : {len(s['non_numeric'])}")
        print(f"  Out-of-range      : {len(s['out_of_range'])}")
        print(f"  Duplicate NORAD   : {len(s['duplicate_norad'])}")
        all_results[name] = result

    # Cross-file NORAD duplicates
    norads_by_file = {}
    for name, result in all_results.items():
        norads_by_file[name] = set()
        for rec in result["all_records"]:
            try: norads_by_file[name].add(int(rec["NORAD_CAT_ID"]))
            except (KeyError, TypeError, ValueError): pass
    cross_dups = []
    fnames = list(norads_by_file.keys())
    for i in range(len(fnames)):
        for j in range(i+1, len(fnames)):
            overlap = norads_by_file[fnames[i]] & norads_by_file[fnames[j]]
            cross_dups.extend(sorted(overlap))
            if overlap:
                print(f"\n[CROSS-FILE] {len(overlap)} NORAD IDs in BOTH '{fnames[i]}' AND '{fnames[j]}': {sorted(overlap)[:5]} ...")
                print("  (Expected: ISS, Tiangong etc. are stations AND active satellites.)")

    # SGP4 test -- all valid records unless --sgp4-sample overrides
    sgp4_cands = []
    for name in ("active_satellites","stations"):
        if name in all_results:
            sgp4_cands.extend(all_results[name]["valid_records"])
    n_to_test = args.sgp4_sample if args.sgp4_sample is not None else len(sgp4_cands)
    print(f"\n[SGP4] Testing propagation on {n_to_test} / {len(sgp4_cands)} valid records ...")
    print(f"  NOTE: {SGP4_ACCURACY_DISCLAIMER}")
    sgp4_result, sgp4_per_rec = _check_sgp4(sgp4_cands, n_to_test)
    print(f"  Tested    : {sgp4_result['sample_size']}")
    print(f"  Succeeded : {sgp4_result['success']}")
    print(f"  Failed    : {sgp4_result['failed']}")
    if sgp4_result["error"]: print(f"  ERROR     : {sgp4_result['error']}")
    if sgp4_result["failed_ids"]: print(f"  Failed IDs: {sgp4_result['failed_ids'][:20]}")

    # Attach per-record SGP4 result to records for output parquet
    norads_tested = {k for k in sgp4_per_rec}
    for result in all_results.values():
        for rec in result["all_records"]:
            nk = int(rec["NORAD_CAT_ID"]) if str(rec["NORAD_CAT_ID"]).isdigit() else str(rec["NORAD_CAT_ID"])
            if nk in sgp4_per_rec:
                rec["_sgp4_ok"] = sgp4_per_rec[nk]["ok"]
                rec["_sgp4_error_code"] = sgp4_per_rec[nk]["error_code"]
                rec["_sgp4_error_msg"] = sgp4_per_rec[nk]["error_msg"]
            else:
                rec["_sgp4_ok"] = None  # not tested
                rec["_sgp4_error_code"] = None
                rec["_sgp4_error_msg"] = None

    # Save processed data
    try:
        import pandas as pd
        all_recs = []
        for result in all_results.values(): all_recs.extend(result["all_records"])
        df = pd.DataFrame(all_recs)
        df["_issues"] = df["_issues"].apply(json.dumps)
        df["_valid"] = df["_issues"].apply(
            lambda x: json.loads(x)==[] or all(i.startswith("stale_epoch") for i in json.loads(x))
        )
        out_parquet = PROCESSED_DIR / "celestrak_validated.parquet"
        df.to_parquet(out_parquet, index=False)
        print(f"\n  [SAVED] {out_parquet.relative_to(PROJECT_ROOT)}")
        out_csv = PROCESSED_DIR / "celestrak_validated.csv"
        df.to_csv(out_csv, index=False)
        print(f"  [SAVED] {out_csv.relative_to(PROJECT_ROOT)}")
    except ImportError:
        print("  [WARN] pandas/pyarrow not available; skipping Parquet/CSV output.")

    # JSON report
    report = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "audit_version": 2,
        "stale_days_threshold": args.stale_days,
        "sgp4_test": sgp4_result,
        "cross_file_duplicate_norads": cross_dups,
        "cross_file_duplicate_note": "These are expected (ISS etc. appear in both files). Not an error.",
        "datasets": {},
    }
    for name, result in all_results.items():
        s = result["stats"]
        report["datasets"][name] = {
            "source_file": s["source"], "total_records": s["total"],
            "valid_records": s["valid_count"], "invalid_records": s["invalid_count"],
            "stale_epoch_warnings": len(s["stale_epoch"]),
            "missing_required_field_count": len(s["missing_required_field"]),
            "bad_epoch_count": len(s["bad_epoch"]),
            "non_numeric_count": len(s["non_numeric"]),
            "out_of_range_count": len(s["out_of_range"]),
            "duplicate_norad_count": len(s["duplicate_norad"]),
            "invalid_indices": s["invalid_indices"],
            "missing_required_fields_detail": s["missing_required_field"],
            "bad_epoch_detail": s["bad_epoch"],
            "non_numeric_detail": s["non_numeric"],
            "out_of_range_detail": s["out_of_range"][:50],
            "duplicate_norad_detail": s["duplicate_norad"],
        }
    rp = REPORT_DIR / "celestrak_validation_report.json"
    with rp.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(f"  [SAVED] {rp.relative_to(PROJECT_ROOT)}")

    tv = sum(r["stats"]["valid_count"] for r in all_results.values())
    ti = sum(r["stats"]["invalid_count"] for r in all_results.values())
    ta = sum(r["stats"]["total"] for r in all_results.values())
    sgp4_ok = sgp4_result["error"] is None and sgp4_result["failed"] == 0
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"  Total records loaded   : {ta:,}")
    print(f"  Records valid (clean)  : {tv:,}")
    print(f"  Records flagged invalid: {ti:,}  (retained with _issues field)")
    print(f"  Cross-file dupe NORADs : {len(cross_dups)}  (expected, see note)")
    print(f"  SGP4 tested            : {sgp4_result['sample_size']} records")
    print(f"  SGP4 result            : {'ALL PASS' if sgp4_ok else 'ISSUES (see report)'}")
    print(f"  SGP4 disclaimer        : success = numerically processable (not physically accurate)")
    print("="*70)
    print("\nRaw files were NOT modified.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
