"""
app.py ??? Akash Setu Flask API backend
======================================
Serves satellite data and SGP4-propagated positions for the 3D visualization.

Endpoints:
  GET /api/satellites          ??? List all satellites (name, norad_id, epoch, metadata)
  GET /api/propagate?ts=<iso>  ??? Propagate all loaded satellites to a UTC timestamp
  GET /api/propagate_one?norad_id=<id>&ts=<iso> ??? Single satellite with full orbit path
  GET /api/health              ??? Health check

Architecture notes:
  - Loads data from raw JSON files (data/raw/celestrak/*.json)
  - Deduplicates by NORAD_CAT_ID (stations take priority over active_satellites)
  - SGP4 propagation converts TEME ??? approximate ECEF via Greenwich sidereal angle
  - Positions are in km from Earth center
  - ECEF???TEME approximation ignores polar motion and nutation (< 1 km error for visualization)

Usage:
  python backend/app.py
"""

from __future__ import annotations
import json, math, os, sys, time
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BACKEND_DIR  = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
RAW_DIR      = PROJECT_ROOT / "data" / "raw" / "celestrak"
FRONTEND_DIR = PROJECT_ROOT / "frontend"

# ---------------------------------------------------------------------------
# SGP4 setup
# ---------------------------------------------------------------------------
from sgp4.api import Satrec, WGS72

DEG2RAD       = math.pi / 180.0
REVDAY2RADMIN = 2.0 * math.pi / 1440.0
EARTH_RADIUS  = 6371.0  # km

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _parse_epoch(s):
    if not isinstance(s, str):
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return None


def _epoch_to_jd(epoch_str):
    """Convert ISO epoch string to Julian Date pair (jd, jd_fraction)."""
    dt = _parse_epoch(epoch_str)
    if dt is None:
        raise ValueError(f"Bad epoch: {epoch_str!r}")
    y, m, d = dt.year, dt.month, dt.day
    h = dt.hour + dt.minute / 60.0 + dt.second / 3600.0 + dt.microsecond / 3.6e9
    if m <= 2:
        y -= 1; m += 12
    A = int(y / 100); B = 2 - A + int(A / 4)
    jd = int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + d + h / 24.0 + B - 1524.5
    jd_day = int(jd)
    jd_frac = jd - jd_day
    return float(jd_day), float(jd_frac)


def _sgp4_epoch_days(epoch_str):
    """SGP4 epoch: days since 1949-12-31 00:00 UT = JD - 2433281.5"""
    jd, jdf = _epoch_to_jd(epoch_str)
    return (jd + jdf) - 2433281.5


def _init_satrec(rec):
    """Build a Satrec from a CelesTrak OMM JSON record. Returns (satrec, error_msg)."""
    try:
        sat = Satrec()
        sat.sgp4init(
            WGS72, "i",
            int(rec["NORAD_CAT_ID"]),
            _sgp4_epoch_days(rec["EPOCH"]),
            float(rec["BSTAR"]),
            float(rec["MEAN_MOTION_DOT"]),
            float(rec["MEAN_MOTION_DDOT"]),
            float(rec["ECCENTRICITY"]),
            float(rec["ARG_OF_PERICENTER"]) * DEG2RAD,
            float(rec["INCLINATION"])       * DEG2RAD,
            float(rec["MEAN_ANOMALY"])      * DEG2RAD,
            float(rec["MEAN_MOTION"])       * REVDAY2RADMIN,
            float(rec["RA_OF_ASC_NODE"])    * DEG2RAD,
        )
        return sat, None
    except Exception as exc:
        return None, str(exc)


def _load_satellites():
    """Load and deduplicate satellite records from raw JSON files."""
    sats = {}  # norad_id -> {meta, satrec}

    for fname, source, priority in [
        ("stations.json",          "stations",          10),
        ("active_satellites.json", "active_satellites",  1),
    ]:
        path = RAW_DIR / fname
        if not path.exists():
            print(f"  [WARN] {path} not found, skipping")
            continue
        with open(path, encoding="utf-8") as f:
            records = json.load(f)
        print(f"  Loaded {len(records)} records from {fname}")

        for rec in records:
            norad = int(rec["NORAD_CAT_ID"])
            # Higher priority wins (stations > active_satellites for dedup)
            if norad in sats and sats[norad]["priority"] >= priority:
                continue  # keep existing higher-priority entry
            sat, err = _init_satrec(rec)
            epoch_dt = _parse_epoch(rec["EPOCH"])
            epoch_age_days = None
            if epoch_dt:
                epoch_age_days = (datetime.now(tz=timezone.utc) - epoch_dt).total_seconds() / 86400.0
            sats[norad] = {
                "norad_id":     norad,
                "name":         rec.get("OBJECT_NAME", "UNKNOWN"),
                "intl_desig":   rec.get("OBJECT_ID", ""),
                "epoch":        rec.get("EPOCH", ""),
                "epoch_age_days": round(epoch_age_days, 2) if epoch_age_days else None,
                "inclination":  float(rec.get("INCLINATION", 0)),
                "eccentricity": float(rec.get("ECCENTRICITY", 0)),
                "mean_motion":  float(rec.get("MEAN_MOTION", 0)),
                "period_min":   round(1440.0 / float(rec["MEAN_MOTION"]), 2) if float(rec.get("MEAN_MOTION",0)) > 0 else None,
                "source":       source,
                "priority":     priority,
                "satrec":       sat,
                "init_error":   err,
                "stale":        epoch_age_days is not None and epoch_age_days > 14,
            }

    print(f"  Total unique satellites loaded: {len(sats)}")
    return sats


def _greenwich_sidereal_angle(jd, jd_frac):
    """
    Approximate Greenwich Mean Sidereal Time in radians.
    Uses IAU 1982 formula. Accuracy: ~1 arcsec (fine for visualization).
    """
    T = ((jd - 2451545.0) + jd_frac) / 36525.0
    # GMST in seconds of time
    gmst_sec = 67310.54841 + (876600.0 * 3600.0 + 8640184.812866) * T + 0.093104 * T * T - 6.2e-6 * T * T * T
    gmst_rad = (gmst_sec % 86400.0) / 86400.0 * 2.0 * math.pi
    return gmst_rad


def _teme_to_ecef(x, y, z, gmst):
    """Rotate TEME position to ECEF (ignoring polar motion ??? <1 km error)."""
    cos_g = math.cos(gmst)
    sin_g = math.sin(gmst)
    xe =  cos_g * x + sin_g * y
    ye = -sin_g * x + cos_g * y
    ze = z
    return xe, ye, ze


def _propagate_sat(sat_data, jd, jd_frac, gmst):
    """Propagate a single satellite and return ECEF position or error."""
    satrec = sat_data["satrec"]
    if satrec is None:
        return None, sat_data["init_error"]
    e, r_teme, v_teme = satrec.sgp4(jd, jd_frac)
    if e != 0:
        return None, f"SGP4 error code {e}"
    x, y, z = _teme_to_ecef(r_teme[0], r_teme[1], r_teme[2], gmst)
    alt = math.sqrt(x*x + y*y + z*z) - EARTH_RADIUS
    return {
        "x": round(x, 3),
        "y": round(y, 3),
        "z": round(z, 3),
        "alt_km": round(alt, 2),
    }, None


def _compute_orbit_path(sat_data, jd, jd_frac, steps=90):
    """Compute one full orbit path (or 24h for high orbits) as a list of ECEF points."""
    satrec = sat_data["satrec"]
    if satrec is None:
        return []
    period_min = sat_data.get("period_min") or 1440.0
    # Limit to 1 orbit or 24h max
    duration_min = min(period_min, 1440.0)
    dt_min = duration_min / steps
    points = []
    for i in range(steps + 1):
        t = i * dt_min
        # Add t minutes to the base JD
        frac = jd_frac + t / 1440.0
        jd_i = jd + int(frac)
        frac_i = frac - int(frac)
        gmst = _greenwich_sidereal_angle(jd_i, frac_i)
        e, r, v = satrec.sgp4(jd_i, frac_i)
        if e != 0:
            continue
        x, y, z = _teme_to_ecef(r[0], r[1], r[2], gmst)
        points.append({"x": round(x, 2), "y": round(y, 2), "z": round(z, 2)})
    return points


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__, static_folder=None)
CORS(app)

SATELLITES = _load_satellites()  # populated on load

# Serve frontend files
@app.route("/")
def serve_index():
    return send_from_directory(str(FRONTEND_DIR), "index.html")

@app.route("/css/<path:filename>")
def serve_css(filename):
    return send_from_directory(str(FRONTEND_DIR / "css"), filename)

@app.route("/js/<path:filename>")
def serve_js(filename):
    return send_from_directory(str(FRONTEND_DIR / "js"), filename)

@app.route("/assets/<path:filename>")
def serve_assets(filename):
    return send_from_directory(str(FRONTEND_DIR / "assets"), filename)


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "satellites_loaded": len(SATELLITES)})


@app.route("/api/satellites")
def list_satellites():
    """Return satellite metadata list (no positions ??? lightweight)."""
    source_filter = request.args.get("source")  # "stations" or "active_satellites"
    search = request.args.get("search", "").upper()
    limit = int(request.args.get("limit", 200))
    offset = int(request.args.get("offset", 0))

    results = []
    for norad_id, s in SATELLITES.items():
        if source_filter and s["source"] != source_filter:
            continue
        if search and search not in s["name"].upper() and search not in str(norad_id):
            continue
        results.append({
            "norad_id":      s["norad_id"],
            "name":          s["name"],
            "intl_desig":    s["intl_desig"],
            "epoch":         s["epoch"],
            "epoch_age_days": s["epoch_age_days"],
            "inclination":   s["inclination"],
            "eccentricity":  s["eccentricity"],
            "mean_motion":   s["mean_motion"],
            "period_min":    s["period_min"],
            "source":        s["source"],
            "stale":         s["stale"],
            "has_error":     s["init_error"] is not None,
        })

    # Sort: stations first, then by name
    results.sort(key=lambda r: (0 if r["source"]=="stations" else 1, r["name"]))
    total = len(results)
    page = results[offset:offset+limit]
    return jsonify({"total": total, "offset": offset, "limit": limit, "satellites": page})


@app.route("/api/propagate")
def propagate_all():
    """Propagate a batch of satellites to a given UTC timestamp."""
    ts = request.args.get("ts")
    norad_ids = request.args.get("norad_ids")  # comma-separated, or empty = default set
    limit = int(request.args.get("limit", 300))

    if ts:
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except Exception:
            return jsonify({"error": f"Invalid timestamp: {ts}"}), 400
    else:
        dt = datetime.now(tz=timezone.utc)

    jd, jdf = _epoch_to_jd(dt.strftime("%Y-%m-%dT%H:%M:%S.%f"))
    gmst = _greenwich_sidereal_angle(jd, jdf)

    if norad_ids:
        ids = [int(x.strip()) for x in norad_ids.split(",") if x.strip().isdigit()]
    else:
        # Default: all stations + first N active satellites
        station_ids = [nid for nid, s in SATELLITES.items() if s["source"] == "stations"]
        other_ids = [nid for nid, s in SATELLITES.items() if s["source"] != "stations"]
        ids = station_ids + other_ids[:limit - len(station_ids)]

    positions = []
    errors = []
    for nid in ids:
        if nid not in SATELLITES:
            continue
        s = SATELLITES[nid]
        pos, err = _propagate_sat(s, jd, jdf, gmst)
        if pos:
            pos["norad_id"] = nid
            pos["name"] = s["name"]
            pos["source"] = s["source"]
            pos["stale"] = s["stale"]
            positions.append(pos)
        else:
            errors.append({"norad_id": nid, "name": s["name"], "error": err})

    return jsonify({
        "timestamp": dt.isoformat(),
        "count": len(positions),
        "error_count": len(errors),
        "positions": positions,
        "errors": errors[:20],
        "note": "Positions are SGP4-propagated estimates in ECEF (km). TEME->ECEF uses approximate GMST (IAU 1982). Not real-time tracking.",
    })


@app.route("/api/propagate_one")
def propagate_one():
    """Propagate a single satellite with full orbit path."""
    norad_id = request.args.get("norad_id")
    ts = request.args.get("ts")

    if not norad_id:
        return jsonify({"error": "norad_id required"}), 400

    nid = int(norad_id)
    if nid not in SATELLITES:
        return jsonify({"error": f"NORAD ID {nid} not found"}), 404

    s = SATELLITES[nid]

    if ts:
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except Exception:
            return jsonify({"error": f"Invalid timestamp: {ts}"}), 400
    else:
        dt = datetime.now(tz=timezone.utc)

    jd, jdf = _epoch_to_jd(dt.strftime("%Y-%m-%dT%H:%M:%S.%f"))
    gmst = _greenwich_sidereal_angle(jd, jdf)

    pos, err = _propagate_sat(s, jd, jdf, gmst)
    orbit = _compute_orbit_path(s, jd, jdf, steps=120)

    return jsonify({
        "timestamp": dt.isoformat(),
        "norad_id": nid,
        "name": s["name"],
        "intl_desig": s["intl_desig"],
        "epoch": s["epoch"],
        "epoch_age_days": s["epoch_age_days"],
        "inclination": s["inclination"],
        "eccentricity": s["eccentricity"],
        "mean_motion": s["mean_motion"],
        "period_min": s["period_min"],
        "source": s["source"],
        "stale": s["stale"],
        "position": pos,
        "position_error": err,
        "orbit_path": orbit,
        "note": "SGP4 propagated estimate. Positions in ECEF (km). Orbital path covers 1 period or 24h (whichever is shorter).",
    })




@app.route("/api/orbit_batch")
def orbit_batch():
    """
    Compute orbit paths for a list of NORAD IDs in one HTTP call.
    Parameters:
      norad_ids -- comma-separated NORAD IDs (required)
      ts        -- UTC ISO timestamp (default: now)
      steps     -- orbit resolution (default 90, max 180)
    Errors per satellite are reported inline, not raised as HTTP errors.
    """
    norad_ids_raw = request.args.get("norad_ids", "")
    ts = request.args.get("ts")
    steps = min(int(request.args.get("steps", 90)), 180)

    if not norad_ids_raw.strip():
        limit = min(int(request.args.get("limit", 200)), 300)
        station_ids = [nid for nid, s in SATELLITES.items() if s.get("source") == "stations"]
        other_ids = [nid for nid, s in SATELLITES.items() if s.get("source") != "stations"]
        nids = (station_ids + other_ids)[:limit]
        if not nids:
            return jsonify({"error": "norad_ids required (comma-separated)"}), 400
    else:
        nids = [int(x.strip()) for x in norad_ids_raw.split(",") if x.strip().isdigit()]

    if ts:
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except Exception:
            return jsonify({"error": f"Invalid timestamp: {ts}"}), 400
    else:
        dt = datetime.now(tz=timezone.utc)

    jd, jdf = _epoch_to_jd(dt.strftime("%Y-%m-%dT%H:%M:%S.%f"))

    results = []
    for nid in nids:
        if nid not in SATELLITES:
            results.append({"norad_id": nid, "error": "not found", "orbit_path": []})
            continue
        s = SATELLITES[nid]
        orbit = _compute_orbit_path(s, jd, jdf, steps=steps)
        entry = {
            "norad_id": nid,
            "name": s["name"],
            "source": s["source"],
            "stale": s["stale"],
            "orbit_path": orbit,
        }
        if not orbit:
            entry["error"] = "SGP4 propagation failed (0 points)"
        results.append(entry)

    return jsonify({
        "timestamp": dt.isoformat(),
        "count": len(results),
        "orbits": results,
    })



# ---------------------------------------------------------------------------
# Collision Screening Endpoints
# ---------------------------------------------------------------------------
from collision_screening import ScreeningConfig, screen_satellites
import dataclasses

# In-memory store for the most recent screening run result
_last_screening_run = None


@app.route("/api/screen")
def screen():
    """
    Run collision screening against a subset of loaded satellites.

    Query parameters (all optional):
      horizon_minutes       : float, default 180
      coarse_step_minutes   : float, default 1.0
      threshold_km          : float, default 5.0
      broad_phase_margin_km : float, default 50.0
      refinement_steps      : int,   default 20
      stale_epoch_days      : float, default 14.0
      threshold_comparison  : "lt" or "lte", default "lt"
      norad_ids             : comma-separated NORAD IDs to screen
                              (default: all stations + first 200 active sats)
      limit                 : int, max satellites to include (default 222)

    Response schema:
      {
        "run_id": str,
        "screening_epoch": str (UTC ISO),
        "horizon_minutes": float,
        "coarse_step_minutes": float,
        "screening_threshold_km": float,
        "total_objects_input": int,
        "objects_screened": int,
        "objects_skipped": int,
        "total_pairs_possible": int,
        "pairs_after_broad_phase": int,
        "pairs_evaluated": int,
        "alert_count": int,
        "alerts": [ ScreeningResult... ],
        "skipped": [ {norad_id, name, reason}... ],
        "runtime_seconds": float,
        "config": { ... },
        "disclaimer": str
      }

    IMPORTANT: A result with screening_status="ALERT" means the propagated
    trajectories come within the threshold.  This is NOT a confirmed collision
    or a calibrated probability.  SGP4 accuracy degrades with TLE age.
    """
    global _last_screening_run

    # Parse config
    import math as _math
    try:
        _h  = float(request.args.get("horizon_minutes", 180))
        _cs = float(request.args.get("coarse_step_minutes", 1.0))
        _th = float(request.args.get("threshold_km", 5.0))
        _bm = float(request.args.get("broad_phase_margin_km", 50.0))
        _rs = int(request.args.get("refinement_steps", 20))
        _sd = float(request.args.get("stale_epoch_days", 14.0))
        _tc = request.args.get("threshold_comparison", "lt")
        # Reject NaN / Inf which would crash downstream
        for _name, _val in [("horizon_minutes",_h),("coarse_step_minutes",_cs),
                             ("threshold_km",_th),("broad_phase_margin_km",_bm),
                             ("stale_epoch_days",_sd)]:
            if not _math.isfinite(_val):
                raise ValueError(f"{_name}={_val} is not a finite number")
        if _cs <= 0:
            raise ValueError("coarse_step_minutes must be > 0")
        if _tc not in ("lt", "lte"):
            raise ValueError(f"threshold_comparison must be 'lt' or 'lte', got {_tc!r}")
        cfg = ScreeningConfig(
            horizon_minutes        = _h,
            coarse_step_minutes    = _cs,
            screening_threshold_km = _th,
            broad_phase_margin_km  = _bm,
            refinement_steps       = _rs,
            stale_epoch_days       = _sd,
            threshold_comparison   = _tc,
        )
    except Exception as exc:
        return jsonify({"error": f"Bad config parameter: {exc}"}), 400

    # Select satellites
    norad_ids_raw = request.args.get("norad_ids")
    limit = int(request.args.get("limit", 222))

    if norad_ids_raw:
        ids = [int(x.strip()) for x in norad_ids_raw.split(",") if x.strip().isdigit()]
        recs = [SATELLITES[nid] for nid in ids if nid in SATELLITES]
    else:
        station_recs = [s for s in SATELLITES.values() if s["source"] == "stations"]
        active_recs  = [s for s in SATELLITES.values() if s["source"] != "stations"]
        recs = station_recs + active_recs[:max(0, limit - len(station_recs))]

    dt = datetime.now(tz=timezone.utc)
    run = screen_satellites(recs, dt, cfg)
    _last_screening_run = run

    # Serialise dataclasses to dict
    def _serialise_result(r):
        d = dataclasses.asdict(r)
        # Convert nan to None for JSON compatibility
        for k, v in d.items():
            if isinstance(v, float) and (v != v):  # nan check
                d[k] = None
        return d

    return jsonify({
        "run_id":                   run.run_id,
        "screening_epoch":          run.screening_epoch,
        "horizon_minutes":          run.horizon_minutes,
        "coarse_step_minutes":      run.coarse_step_minutes,
        "screening_threshold_km":   run.screening_threshold_km,
        "total_objects_input":      run.total_objects_input,
        "objects_screened":         run.objects_screened,
        "objects_skipped":          run.objects_skipped,
        "total_pairs_possible":     run.total_pairs_possible,
        "pairs_after_broad_phase":  run.pairs_after_broad_phase,
        "pairs_evaluated":          run.pairs_evaluated,
        "alert_count":              sum(1 for a in run.alerts if a.screening_status == "ALERT"),
        "alerts":                   [_serialise_result(a) for a in run.alerts],
        "skipped":                  run.skipped,
        "runtime_seconds":          run.runtime_seconds,
        "config":                   run.config,
        "disclaimer":               run.disclaimer,
    })


@app.route("/api/screen/results")
def screen_results():
    """
    Return the most recent screening run result (if available).

    This endpoint is idempotent -- it does not trigger a new screening run.
    Returns 404 if no screening has been run since server start.
    """
    global _last_screening_run
    if _last_screening_run is None:
        return jsonify({
            "error": "No screening run available. Call /api/screen first.",
            "hint": "GET /api/screen to run screening with default parameters."
        }), 404

    run = _last_screening_run

    def _serialise_result(r):
        d = dataclasses.asdict(r)
        for k, v in d.items():
            if isinstance(v, float) and (v != v):
                d[k] = None
        return d

    return jsonify({
        "run_id":                   run.run_id,
        "screening_epoch":          run.screening_epoch,
        "alert_count":              sum(1 for a in run.alerts if a.screening_status == "ALERT"),
        "alerts":                   [_serialise_result(a) for a in run.alerts],
        "skipped":                  run.skipped,
        "runtime_seconds":          run.runtime_seconds,
        "config":                   run.config,
        "disclaimer":               run.disclaimer,
    })


# ---------------------------------------------------------------------------
# Grid-Based Collision Analysis Endpoints
# ---------------------------------------------------------------------------
from grid_analysis import GridConfig, analyse_encounter, analyse_all_alerts
from collision_avoidance import (
    AvoidanceConfig, generate_hybrid_avoidance_plan,
    evaluate_avoidance_candidate, serialise_avoidance_plan,
    STRATEGY_PROGRADE_ALONG_TRACK, STRATEGY_RETROGRADE_ALONG_TRACK,
    STRATEGY_POS_CROSS_TRACK, STRATEGY_NEG_CROSS_TRACK,
    STRATEGY_POS_RADIAL, STRATEGY_NEG_RADIAL, STRATEGY_CUSTOM,
)
import dataclasses


def _serialise_grid_result(r):
    """Convert GridAnalysisResult to a JSON-serialisable dict."""
    d = dataclasses.asdict(r)
    for k, v in list(d.items()):
        if isinstance(v, float) and (v != v):   # NaN -> None
            d[k] = None
    return d


@app.route("/api/analyse")
def analyse():
    """
    Run Phase-3 screening then apply grid-based TCA refinement to all alerts.

    Query parameters (Phase-3 screening):
      norad_ids             : comma-separated NORAD IDs (default: all stations + 200 active)
      limit                 : int, max satellites (default 222)
      horizon_minutes       : float (default 180)
      coarse_step_minutes   : float (default 1.0)
      threshold_km          : float (default 5.0)
      broad_phase_margin_km : float (default 50.0)

    Query parameters (Phase-4 grid):
      grid_half_window_min  : float, search window around TCA (default 2.0)
      grid_points           : int, grid points per level (default 20)
      grid_refinement_factor: int, points divisor per level (default 10)
      target_resolution_m   : float, target spatial resolution in metres (default 0.01)
      max_grid_iterations   : int (default 12)
      include_iteration_log : bool, "true"/"false" (default false)

    Returns schema:
      {
        "phase3_run_id":   str,
        "screening_epoch": str,
        "objects_screened": int,
        "phase3_alert_count": int,
        "grid_analyses": [ GridAnalysisResult... ],
        "runtime_seconds": float,
        "accuracy_disclaimer": str
      }
    """
    import math as _math

    # --- Phase-3 config ---
    try:
        from collision_screening import ScreeningConfig
        _h   = float(request.args.get("horizon_minutes", 180))
        _cs  = float(request.args.get("coarse_step_minutes", 1.0))
        _th  = float(request.args.get("threshold_km", 5.0))
        _bm  = float(request.args.get("broad_phase_margin_km", 50.0))
        for nm, v in [("horizon_minutes",_h),("coarse_step_minutes",_cs),
                      ("threshold_km",_th),("broad_phase_margin_km",_bm)]:
            if not _math.isfinite(v):
                raise ValueError(f"{nm}={v} is not finite")
        p3_cfg = ScreeningConfig(
            horizon_minutes=_h, coarse_step_minutes=_cs,
            screening_threshold_km=_th, broad_phase_margin_km=_bm,
        )
    except Exception as exc:
        return jsonify({"error": f"Bad Phase-3 config: {exc}"}), 400

    # --- Phase-4 grid config ---
    try:
        _hw  = float(request.args.get("grid_half_window_min", 2.0))
        _gp  = int(request.args.get("grid_points", 20))
        _rf  = int(request.args.get("grid_refinement_factor", 10))
        _tr  = float(request.args.get("target_resolution_m", 0.01))
        _mi  = int(request.args.get("max_grid_iterations", 12))
        _il  = request.args.get("include_iteration_log", "false").lower() == "true"
        for nm, v in [("grid_half_window_min",_hw),("target_resolution_m",_tr)]:
            if not _math.isfinite(v):
                raise ValueError(f"{nm}={v} is not finite")
        if _gp < 3:
            raise ValueError("grid_points must be >= 3")
        if _rf < 2:
            raise ValueError("grid_refinement_factor must be >= 2")
        g4_cfg = GridConfig(
            search_half_window_minutes=_hw,
            initial_grid_points=_gp,
            refinement_factor=_rf,
            target_spatial_resolution_m=_tr,
            max_iterations=_mi,
            include_iteration_log=_il,
        )
    except Exception as exc:
        return jsonify({"error": f"Bad Phase-4 grid config: {exc}"}), 400

    # --- Select satellites ---
    norad_ids_raw = request.args.get("norad_ids")
    limit = int(request.args.get("limit", 222))
    if norad_ids_raw:
        ids = [int(x.strip()) for x in norad_ids_raw.split(",") if x.strip().isdigit()]
        recs = [SATELLITES[nid] for nid in ids if nid in SATELLITES]
    else:
        station_recs = [s for s in SATELLITES.values() if s.get("source") == "stations"]
        active_recs  = [s for s in SATELLITES.values() if s.get("source") != "stations"]
        recs = station_recs + active_recs[:max(0, limit - len(station_recs))]

    t_total_start = datetime.now(tz=timezone.utc)

    # --- Phase-3 screening ---
    dt = t_total_start
    p3_run = screen_satellites(recs, dt, p3_cfg)

    # --- Phase-4 grid analysis on alerts ---
    g4_results = analyse_all_alerts(p3_run, recs, dt, g4_cfg, alerts_only=True)

    t_total_s = (datetime.now(tz=timezone.utc) - t_total_start).total_seconds()

    return jsonify({
        "phase3_run_id":       p3_run.run_id,
        "screening_epoch":     p3_run.screening_epoch,
        "objects_screened":    p3_run.objects_screened,
        "phase3_alert_count":  p3_run.alert_count if hasattr(p3_run, "alert_count") else len([a for a in p3_run.alerts if a.screening_status == "ALERT"]),
        "grid_analyses":       [_serialise_grid_result(r) for r in g4_results],
        "runtime_seconds":     round(t_total_s, 3),
        "accuracy_disclaimer": (
            "Grid resolution is a numerical property of the algorithm. "
            "It does NOT represent physical prediction accuracy. "
            "SGP4 TLE-based position errors typically range from 100 m to >10 km "
            "depending on TLE age. Results are not suitable for operational use."
        ),
    })


@app.route("/api/analyse/pair")
def analyse_pair():
    """
    Run Phase-3 screening on exactly two NORAD IDs, then apply grid refinement.

    Required parameters:
      norad_a, norad_b      : NORAD IDs (integers)

    All Phase-3 and Phase-4 grid parameters from /api/analyse also apply.

    This endpoint is the primary route for controlled test scenarios where
    exact NORAD IDs are known.
    """
    import math as _math
    from collision_screening import ScreeningConfig

    try:
        norad_a = int(request.args.get("norad_a", ""))
        norad_b = int(request.args.get("norad_b", ""))
    except (TypeError, ValueError):
        return jsonify({"error": "norad_a and norad_b must be integer NORAD IDs"}), 400

    if norad_a not in SATELLITES or norad_b not in SATELLITES:
        missing = [n for n in [norad_a, norad_b] if n not in SATELLITES]
        return jsonify({"error": f"NORAD IDs not found: {missing}"}), 404

    recs = [SATELLITES[norad_a], SATELLITES[norad_b]]

    try:
        _h  = float(request.args.get("horizon_minutes", 180))
        _cs = float(request.args.get("coarse_step_minutes", 1.0))
        _th = float(request.args.get("threshold_km", 5.0))
        _bm = float(request.args.get("broad_phase_margin_km", 50.0))
        for nm, v in [("horizon_minutes",_h),("coarse_step_minutes",_cs)]:
            if not _math.isfinite(v): raise ValueError(f"{nm}={v}")
        p3_cfg = ScreeningConfig(horizon_minutes=_h, coarse_step_minutes=_cs,
                                  screening_threshold_km=_th, broad_phase_margin_km=_bm)

        _hw = float(request.args.get("grid_half_window_min", 2.0))
        _gp = int(request.args.get("grid_points", 20))
        _tr = float(request.args.get("target_resolution_m", 0.01))
        _mi = int(request.args.get("max_grid_iterations", 12))
        _il = request.args.get("include_iteration_log", "false").lower() == "true"
        g4_cfg = GridConfig(search_half_window_minutes=_hw, initial_grid_points=_gp,
                             target_spatial_resolution_m=_tr, max_iterations=_mi,
                             include_iteration_log=_il)
    except Exception as exc:
        return jsonify({"error": f"Bad config: {exc}"}), 400

    dt = datetime.now(tz=timezone.utc)
    p3_run = screen_satellites(recs, dt, p3_cfg)
    g4_results = analyse_all_alerts(p3_run, recs, dt, g4_cfg, alerts_only=True)

    return jsonify({
        "phase3_run_id":       p3_run.run_id,
        "screening_epoch":     p3_run.screening_epoch,
        "norad_a":             norad_a,
        "norad_b":             norad_b,
        "phase3_miss_distance_km": p3_run.alerts[0].miss_distance_km if p3_run.alerts else None,
        "phase3_tca_minutes":  p3_run.alerts[0].tca_minutes_from_start if p3_run.alerts else None,
        "grid_analyses":       [_serialise_grid_result(r) for r in g4_results],
        "accuracy_disclaimer": (
            "Grid resolution is a numerical property, not a measure of physical accuracy. "
            "Not suitable for operational collision avoidance."
        ),
    })
# ---------------------------------------------------------------------------
# Hybrid Collision Avoidance Endpoints
# ---------------------------------------------------------------------------

@app.route("/api/avoidance/plan")
def avoidance_plan():
    """
    Generate hybrid collision avoidance plan comparing multiple maneuver strategies.

    Required parameters:
      norad_a, norad_b : integer NORAD catalog IDs

    Optional parameters:
      delta_v_m_s           : float (default 0.5)
      target_clearance_km   : float (default 15.0)
      max_delta_v_m_s       : float (default 5.0)
      min_lead_time_minutes : float (default 15.0)
      lead_time_minutes     : float (optional)
      threshold_km          : float (default 15.0)
      horizon_minutes       : float (default 180.0)
      spacecraft_area_min_m2: float (optional)
      spacecraft_area_max_m2: float (optional)
      spacecraft_mass_kg    : float (optional)
      spacecraft_drag_coeff : float (optional)
    """
    import math as _math
    from collision_screening import ScreeningConfig, screen_satellites

    try:
        norad_a = int(request.args.get("norad_a", ""))
        norad_b = int(request.args.get("norad_b", ""))
    except (TypeError, ValueError):
        return jsonify({"error": "norad_a and norad_b must be integer NORAD IDs"}), 400

    if norad_a not in SATELLITES or norad_b not in SATELLITES:
        missing = [n for n in [norad_a, norad_b] if n not in SATELLITES]
        return jsonify({"error": f"NORAD IDs not found: {missing}"}), 404

    recs = [SATELLITES[norad_a], SATELLITES[norad_b]]

    try:
        dv_m_s = float(request.args.get("delta_v_m_s", 0.5))
        clearance_km = float(request.args.get("target_clearance_km", 15.0))
        max_dv = float(request.args.get("max_delta_v_m_s", 5.0))
        min_lead = float(request.args.get("min_lead_time_minutes", 15.0))
        h_min = float(request.args.get("horizon_minutes", 180.0))
        th_km = float(request.args.get("threshold_km", 15.0))

        lead_min_raw = request.args.get("lead_time_minutes")
        lead_min = float(lead_min_raw) if lead_min_raw is not None else None

        for nm, v in [("delta_v_m_s", dv_m_s), ("target_clearance_km", clearance_km),
                      ("max_delta_v_m_s", max_dv), ("min_lead_time_minutes", min_lead)]:
            if not _math.isfinite(v):
                raise ValueError(f"{nm}={v} is not finite")

        geom = None
        if request.args.get("spacecraft_area_min_m2") and request.args.get("spacecraft_area_max_m2") and request.args.get("spacecraft_mass_kg"):
            geom = {
                "area_min_m2": float(request.args.get("spacecraft_area_min_m2")),
                "area_max_m2": float(request.args.get("spacecraft_area_max_m2")),
                "mass_kg": float(request.args.get("spacecraft_mass_kg")),
                "drag_coeff": float(request.args.get("spacecraft_drag_coeff", 2.2)),
            }

        av_cfg = AvoidanceConfig(
            delta_v_m_s=dv_m_s,
            target_clearance_km=clearance_km,
            max_delta_v_m_s=max_dv,
            min_lead_time_minutes=min_lead,
            lead_time_minutes=lead_min,
            spacecraft_geometry=geom,
        )
    except Exception as exc:
        return jsonify({"error": f"Bad avoidance config: {exc}"}), 400

    dt = datetime.now(tz=timezone.utc)
    p3_cfg = ScreeningConfig(
        horizon_minutes=h_min,
        coarse_step_minutes=0.5,
        screening_threshold_km=th_km,
        broad_phase_margin_km=200.0,
    )
    p3_run = screen_satellites(recs, dt, p3_cfg)

    if not p3_run.alerts:
        return jsonify({
            "status": "SAFE",
            "message": "No collision alert detected between specified satellites",
            "norad_a": norad_a,
            "norad_b": norad_b,
            "screening_epoch": p3_run.screening_epoch,
            "horizon_minutes": h_min,
        }), 200

    alert = p3_run.alerts[0]
    all_recs = list(SATELLITES.values())[:50]
    plan = generate_hybrid_avoidance_plan(alert, recs[0], recs[1], dt, av_cfg, all_recs)

    return jsonify(serialise_avoidance_plan(plan)), 200


@app.route("/api/avoidance/evaluate")
def avoidance_evaluate():
    """
    Evaluate a specific candidate maneuver direction against an encounter alert.

    Required parameters:
      norad_a, norad_b : integer NORAD catalog IDs
      direction        : string maneuver direction

    Optional parameters:
      delta_v_m_s           : float (default 0.5)
      target_clearance_km   : float (default 15.0)
      lead_time_minutes     : float (optional)
      threshold_km          : float (default 15.0)
    """
    import math as _math
    from collision_screening import ScreeningConfig, screen_satellites

    try:
        norad_a = int(request.args.get("norad_a", ""))
        norad_b = int(request.args.get("norad_b", ""))
        direction = request.args.get("direction", "").strip()
        if not direction:
            raise ValueError("direction parameter is required")
    except (TypeError, ValueError) as exc:
        return jsonify({"error": f"Bad parameters: {exc}"}), 400

    if norad_a not in SATELLITES or norad_b not in SATELLITES:
        missing = [n for n in [norad_a, norad_b] if n not in SATELLITES]
        return jsonify({"error": f"NORAD IDs not found: {missing}"}), 404

    recs = [SATELLITES[norad_a], SATELLITES[norad_b]]

    try:
        dv_m_s = float(request.args.get("delta_v_m_s", 0.5))
        clearance_km = float(request.args.get("target_clearance_km", 15.0))
        max_dv = float(request.args.get("max_delta_v_m_s", 5.0))
        min_lead = float(request.args.get("min_lead_time_minutes", 15.0))
        h_min = float(request.args.get("horizon_minutes", 180.0))
        th_km = float(request.args.get("threshold_km", 15.0))

        lead_min_raw = request.args.get("lead_time_minutes")
        lead_min = float(lead_min_raw) if lead_min_raw is not None else None

        av_cfg = AvoidanceConfig(
            delta_v_m_s=dv_m_s,
            maneuver_direction=direction,
            target_clearance_km=clearance_km,
            max_delta_v_m_s=max_dv,
            min_lead_time_minutes=min_lead,
            lead_time_minutes=lead_min,
        )
    except Exception as exc:
        return jsonify({"error": f"Bad avoidance config: {exc}"}), 400

    dt = datetime.now(tz=timezone.utc)
    p3_cfg = ScreeningConfig(
        horizon_minutes=h_min,
        coarse_step_minutes=0.5,
        screening_threshold_km=th_km,
        broad_phase_margin_km=200.0,
    )
    p3_run = screen_satellites(recs, dt, p3_cfg)

    if not p3_run.alerts:
        return jsonify({
            "status": "SAFE",
            "message": "No collision alert detected between specified satellites",
            "norad_a": norad_a,
            "norad_b": norad_b,
        }), 200

    alert = p3_run.alerts[0]
    lead_time = lead_min if lead_min is not None else max(min_lead, min(92.9, alert.tca_minutes_from_start * 0.7))

    cand = evaluate_avoidance_candidate(
        alert, recs[0], recs[1], dt, av_cfg, direction, dv_m_s, lead_time
    )

    # Convert candidate to dict
    cand_dict = {
        "event_id": alert.event_id,
        "primary_norad": norad_a,
        "secondary_norad": norad_b,
        "strategy_name": cand.strategy_name,
        "maneuver_direction": cand.maneuver_direction,
        "delta_v_m_s": cand.delta_v_m_s,
        "delta_v_vector_rtn_m_s": list(cand.delta_v_vector_rtn_m_s),
        "maneuver_lead_time_minutes": cand.maneuver_lead_time_minutes,
        "maneuver_time_minutes": cand.maneuver_time_minutes,
        "maneuver_epoch_utc": cand.maneuver_epoch_utc,
        "before_miss_distance_km": cand.before_miss_distance_km,
        "before_tca_minutes": cand.before_tca_minutes,
        "before_tca_utc": cand.before_tca_utc,
        "after_miss_distance_km": cand.after_miss_distance_km,
        "after_tca_minutes": cand.after_tca_minutes,
        "after_tca_utc": cand.after_tca_utc,
        "miss_distance_improvement_km": cand.miss_distance_improvement_km,
        "tca_delta_seconds": cand.tca_delta_seconds,
        "clears_threshold": cand.clears_threshold,
        "is_feasible": cand.is_feasible,
        "feasibility_reasons": cand.feasibility_reasons,
        "semi_major_axis_change_m": cand.semi_major_axis_change_m,
        "orbital_period_change_s": cand.orbital_period_change_s,
        "perigee_altitude_km": cand.perigee_altitude_km,
    }
    return jsonify(cand_dict), 200


@app.route("/api/avoidance/trajectory")
def avoidance_trajectory():
    """
    Computes the 3D post-burn orbit path (in ECEF km) for a chosen avoidance maneuver candidate.

    Query parameters:
      norad_id         : integer NORAD catalog ID of the maneuvering satellite (required)
      direction        : string maneuver direction ('prograde_along_track', 'retrograde_along_track',
                         'positive_radial', 'negative_radial', 'positive_cross_track', 'negative_cross_track')
      delta_v_m_s      : float delta-v in m/s (default 0.5)
      lead_time_minutes: float lead time before TCA / from start (default 15.0)
      steps            : integer number of orbit steps (default 120)
    """
    from datetime import timedelta
    from collision_avoidance import (
        calculate_maneuvered_elements, ManeuveredSatrec,
        STRATEGY_PROGRADE_ALONG_TRACK, STRATEGY_RETROGRADE_ALONG_TRACK,
        STRATEGY_POS_RADIAL, STRATEGY_NEG_RADIAL,
        STRATEGY_POS_CROSS_TRACK, STRATEGY_NEG_CROSS_TRACK
    )

    norad_str = request.args.get("norad_id") or request.args.get("norad_a")
    if not norad_str:
        return jsonify({"error": "norad_id (or norad_a) required"}), 400

    try:
        nid = int(norad_str)
    except ValueError:
        return jsonify({"error": "norad_id must be an integer"}), 400

    if nid not in SATELLITES:
        return jsonify({"error": f"NORAD ID {nid} not found"}), 404

    s = SATELLITES[nid]
    direction = request.args.get("direction", STRATEGY_PROGRADE_ALONG_TRACK)
    dv_m_s = float(request.args.get("delta_v_m_s", 0.5))
    lead_min = float(request.args.get("lead_time_minutes", 15.0))
    steps = max(30, min(240, int(request.args.get("steps", 120))))

    if dv_m_s <= 0.0 or dv_m_s > 100.0:
        return jsonify({"error": f"delta_v_m_s must be positive and <= 100 m/s, got {dv_m_s}"}), 400

    dir_lower = direction.lower()
    if "prograde" in dir_lower or dir_lower == "+t":
        dvr, dvt, dvw = 0.0, dv_m_s / 1000.0, 0.0
        normalized_dir = "prograde"
    elif "retrograde" in dir_lower or dir_lower == "-t":
        dvr, dvt, dvw = 0.0, -dv_m_s / 1000.0, 0.0
        normalized_dir = "retrograde"
    elif "pos_rad" in dir_lower or "+r" in dir_lower or ("radial" in dir_lower and "neg" not in dir_lower and "in" not in dir_lower and "south" not in dir_lower):
        dvr, dvt, dvw = dv_m_s / 1000.0, 0.0, 0.0
        normalized_dir = "radial_out"
    elif "neg_rad" in dir_lower or "-r" in dir_lower or ("radial" in dir_lower and "in" in dir_lower):
        dvr, dvt, dvw = -dv_m_s / 1000.0, 0.0, 0.0
        normalized_dir = "radial_in"
    elif "pos_cross" in dir_lower or "+w" in dir_lower or "north" in dir_lower or ("cross" in dir_lower and "neg" not in dir_lower and "south" not in dir_lower):
        dvr, dvt, dvw = 0.0, 0.0, dv_m_s / 1000.0
        normalized_dir = "cross_track_north"
    elif "neg_cross" in dir_lower or "-w" in dir_lower or "south" in dir_lower:
        dvr, dvt, dvw = 0.0, 0.0, -dv_m_s / 1000.0
        normalized_dir = "cross_track_south"
    else:
        return jsonify({"error": f"Invalid maneuver direction: {direction}"}), 400

    dt = datetime.now(tz=timezone.utc)
    jd0, jdf0 = _epoch_to_jd(dt.strftime("%Y-%m-%dT%H:%M:%S.%f"))
    epoch_days = jd0 + jdf0 - 2433281.5

    try:
        sat_man, changes = calculate_maneuvered_elements(s["satrec"], epoch_days, lead_min, dvr, dvt, dvw)
        man_rec = ManeuveredSatrec(s["satrec"], sat_man, lead_min, jd0, jdf0, dv_rtn_km_s=(dvr, dvt, dvw))
    except Exception as exc:
        return jsonify({"error": f"Failed to compute maneuvered orbit: {exc}"}), 500

    period = s.get("period_min") or 92.0
    orbit_path = []
    for i in range(steps):
        t_step = i * (period / steps)
        t_dt = dt + timedelta(minutes=t_step)
        jd_i, jdf_i = _epoch_to_jd(t_dt.strftime("%Y-%m-%dT%H:%M:%S.%f"))
        gmst_i = _greenwich_sidereal_angle(jd_i, jdf_i)
        err, pos, _ = man_rec.sgp4(jd_i, jdf_i)
        if err == 0:
            ecef = _teme_to_ecef(pos[0], pos[1], pos[2], gmst_i)
            orbit_path.append([round(c, 3) for c in ecef])

    return jsonify({
        "norad_id": nid,
        "name": s["name"],
        "direction": normalized_dir,
        "delta_v_m_s": dv_m_s,
        "delta_v_vector_rtn_m_s": [round(dvr * 1000.0, 3), round(dvt * 1000.0, 3), round(dvw * 1000.0, 3)],
        "lead_time_minutes": lead_min,
        "orbit_path": orbit_path,
        "steps": len(orbit_path),
        "semi_major_axis_change_m": round(changes.get("da_m", 0.0), 3),
        "orbital_period_change_s": round(changes.get("period_change_s", 0.0), 3),
        "perigee_altitude_km": round(changes.get("perigee_altitude_km", 0.0), 3),
        "disclaimer": (
            "POST-BURN TRAJECTORY ESTIMATE ONLY. Based on impulsive Gauss Variational Equations "
            "and Clohessy-Wiltshire relative motion superposition. Does not guarantee real-world "
            "collision avoidance clearance."
        ),
    }), 200


# ---------------------------------------------------------------------------
# ML-Based Collision Risk Prediction Endpoints
# ---------------------------------------------------------------------------

_ml_predictor = None


def _get_ml_predictor():
    global _ml_predictor
    if _ml_predictor is None:
        try:
            try:
                from backend.ml_risk_engine import MLRiskPredictor
            except ImportError:
                from ml_risk_engine import MLRiskPredictor
            _ml_predictor = MLRiskPredictor.load(BACKEND_DIR / "models" / "phase6")
        except Exception as exc:
            import traceback
            with open('ml_err.log', 'w', encoding='utf-8') as ef:
                ef.write(traceback.format_exc())
            return None
    return _ml_predictor


@app.route("/api/v1/ml/model_info", methods=["GET"])
def ml_model_info():
    """
    Returns metadata, training metrics, and operational disclaimers for the
    ML-Based Collision Risk Prediction model.
    """
    predictor = _get_ml_predictor()
    if predictor is None:
        return jsonify({
            "error": "ML model artifacts not loaded. Run backend/scripts/train_ml_risk.py to generate them."
        }), 503

    meta = predictor.metadata or {}
    return jsonify({
        "status": "available",
        "model_version": meta.get("model_version", "1.0.0-phase6"),
        "training_timestamp": meta.get("training_timestamp"),
        "n_features": meta.get("n_features", len(predictor.preprocessor.feature_cols) if predictor.preprocessor else 0),
        "evaluation_summary": meta.get("evaluation_summary", {}),
        "top_features": meta.get("top_features", []),
        "action_threshold": meta.get("action_threshold", -6.0),
        "censored_risk_floor": meta.get("censored_risk_floor", -30.0),
        "scientific_caveat": (
            "Prototype ML model trained on historical ESA Kelvins competition CDMs. "
            "Outputs provide relative risk estimates and do not represent certified operational "
            "collision probabilities for active on-orbit assets. "
            "Anonymized ESA records must never be cross-joined with live CelesTrak NORAD catalog IDs."
        ),
    }), 200


@app.route("/api/v1/ml/predict_risk", methods=["POST"])
def ml_predict_risk():
    """
    Predicts final collision risk (log10 probability) and high-risk classification
    for a given Conjunction Data Message (CDM).

    Payload: JSON dictionary of CDM features (e.g. time_to_tca, miss_distance,
    relative_speed, mahalanobis_distance, c_position_covariance_det, etc.).
    """
    predictor = _get_ml_predictor()
    if predictor is None:
        return jsonify({
            "error": "ML model artifacts not loaded. Run backend/scripts/train_ml_risk.py to generate them."
        }), 503

    if not request.is_json:
        return jsonify({"error": "Request body must be valid JSON"}), 400

    payload = request.get_json()
    if not isinstance(payload, dict):
        return jsonify({"error": "JSON payload must be an object/dictionary"}), 400

    cdm_data = payload.get("cdm", payload)
    if not isinstance(cdm_data, dict) or not cdm_data:
        return jsonify({"error": "CDM record cannot be empty"}), 400

    try:
        result = predictor.predict_cdm(cdm_data)
        return jsonify(result), 200
    except Exception as exc:
        return jsonify({"error": f"Prediction failed: {str(exc)}"}), 400


@app.route("/api/v1/ml/predict_batch", methods=["POST"])
def ml_predict_batch():
    """
    Predicts collision risk for a batch of Conjunction Data Messages (CDMs).

    Payload: JSON object containing 'cdms': list of CDM feature dictionaries.
    """
    predictor = _get_ml_predictor()
    if predictor is None:
        return jsonify({
            "error": "ML model artifacts not loaded. Run backend/scripts/train_ml_risk.py to generate them."
        }), 503

    if not request.is_json:
        return jsonify({"error": "Request body must be valid JSON"}), 400

    payload = request.get_json()
    if not isinstance(payload, dict) or "cdms" not in payload:
        return jsonify({"error": "Payload must contain a 'cdms' array of objects"}), 400

    cdms = payload["cdms"]
    if not isinstance(cdms, list) or len(cdms) == 0:
        return jsonify({"error": "'cdms' must be a non-empty list of CDM objects"}), 400

    if len(cdms) > 1000:
        return jsonify({"error": "Batch size cannot exceed 1000 CDMs"}), 400

    try:
        results = predictor.predict_batch(cdms)
        return jsonify({
            "count": len(results),
            "predictions": results,
            "model_version": predictor.metadata.get("model_version", "1.0.0-phase6"),
        }), 200
    except Exception as exc:
        return jsonify({"error": f"Batch prediction failed: {str(exc)}"}), 400


# ---------------------------------------------------------------------------
# Startup


# ===========================================================================
# AI Assistant Endpoints (Groq-powered)
# ===========================================================================
# Grounding context builder
# ---------------------------------------------------------------------------

def _build_grounding_context(
    selected_norad_id=None,
    selected_event_id=None,
) -> str:
    """
    Assemble a factual context string from live backend data.
    This is injected into the AI system prompt (never echoed to the user)
    so the LLM can ground answers in real catalogue values.
    """
    lines = []

    # 1. Catalogue summary
    total = len(SATELLITES)
    station_count = sum(1 for s in SATELLITES.values() if s["source"] == "stations")
    active_count  = sum(1 for s in SATELLITES.values() if s["source"] == "active_satellites")
    stale_count   = sum(1 for s in SATELLITES.values() if s.get("stale"))
    err_count     = sum(1 for s in SATELLITES.values() if s.get("init_error"))
    lines.append(
        f"CATALOGUE SUMMARY: {total} total objects loaded "
        f"({station_count} space stations, {active_count} active satellites). "
        f"Stale TLE (>14 days): {stale_count}. Init errors: {err_count}."
    )

    # 2. Selected satellite details
    if selected_norad_id is not None:
        try:
            nid = int(selected_norad_id)
        except (TypeError, ValueError):
            nid = None
        if nid and nid in SATELLITES:
            s = SATELLITES[nid]
            lines.append(
                f"SELECTED SATELLITE: {s['name']} (NORAD {nid}), "
                f"Source: {s['source']}, "
                f"Epoch: {s['epoch']} (age: {s.get('epoch_age_days', 'unknown')} days), "
                f"Inclination: {s['inclination']}°, "
                f"Eccentricity: {s['eccentricity']:.6f}, "
                f"Mean Motion: {s['mean_motion']:.6f} rev/day, "
                f"Period: {s.get('period_min', 'N/A')} min, "
                f"Stale: {s.get('stale', False)}, "
                f"Init error: {s.get('init_error', None)}."
            )
        else:
            lines.append(f"SELECTED SATELLITE: NORAD ID {selected_norad_id} not found in catalogue.")

    # 3. Last screening run summary
    global _last_screening_run
    if _last_screening_run is not None:
        run = _last_screening_run
        alert_count = sum(1 for a in run.alerts if a.screening_status == "ALERT")
        lines.append(
            f"LAST SCREENING RUN: {run.screening_epoch}, "
            f"Horizon: {run.horizon_minutes} min, "
            f"Threshold: {run.screening_threshold_km} km, "
            f"Objects screened: {run.objects_screened}, "
            f"Alerts: {alert_count}."
        )
        # Top 3 alerts by miss distance
        real_alerts = [a for a in run.alerts if a.screening_status == "ALERT"]
        real_alerts.sort(key=lambda a: a.miss_distance_km)
        for i, a in enumerate(real_alerts[:3], 1):
            lines.append(
                f"  Alert {i}: {a.name_1} (NORAD {a.norad_id_1}) vs "
                f"{a.name_2} (NORAD {a.norad_id_2}), "
                f"Miss distance: {a.miss_distance_km:.3f} km, "
                f"TCA: {a.tca_utc}, Event ID: {a.event_id}."
            )
        # Specific event if selected
        if selected_event_id:
            match = next(
                (a for a in run.alerts if a.event_id == selected_event_id), None
            )
            if match:
                lines.append(
                    f"SELECTED EVENT ({selected_event_id}): "
                    f"{match.name_1} vs {match.name_2}, "
                    f"Miss distance: {match.miss_distance_km:.3f} km, "
                    f"Relative speed: {match.relative_speed_km_s:.3f} km/s, "
                    f"TCA: {match.tca_utc}, Status: {match.screening_status}."
                )
            else:
                lines.append(f"SELECTED EVENT {selected_event_id}: not found in last screening run.")
    else:
        lines.append("LAST SCREENING RUN: No screening run performed yet.")

    # 4. ML model status
    predictor = _get_ml_predictor()
    if predictor and predictor.metadata:
        meta = predictor.metadata
        lines.append(
            f"ML MODEL: version {meta.get('model_version', 'N/A')}, "
            f"trained {meta.get('training_timestamp', 'unknown')}, "
            f"action_threshold log10(P)={meta.get('action_threshold', -6.0)}, "
            f"dataset: anonymized ESA Kelvins Collision Avoidance Challenge CDMs."
        )
    else:
        lines.append("ML MODEL: Not loaded. Run backend/scripts/train_ml_risk.py to generate artifacts.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# AI Routes
# ---------------------------------------------------------------------------

@app.route("/api/ai/status")
def ai_status():
    """Return provider, assistant name, and safe configuration status."""
    try:
        from ai_service import get_status
    except ImportError:
        from backend.ai_service import get_status
    return jsonify(get_status()), 200


@app.route("/api/ai/suggestions")
def ai_suggestions():
    """Return predefined suggested questions. Does not require Groq."""
    try:
        from ai_service import SUGGESTIONS
    except ImportError:
        from backend.ai_service import SUGGESTIONS
    return jsonify({"suggestions": SUGGESTIONS}), 200


@app.route("/api/ai/ask", methods=["POST"])
def ai_ask():
    """
    POST /api/ai/ask

    Body (JSON):
    {
      "question": str,                   // required, max 2000 chars
      "history": [                       // optional, max 10 turns
        {"role": "user"|"assistant", "content": str},
        ...
      ],
      "selected_norad_id": int|null,     // optional selection context
      "selected_event_id": str|null      // optional event context
    }

    Response:
    {
      "answer": str,
      "sources": str,
      "data_timestamp": str (UTC ISO),
      "model": str,
      "configured": bool
    }

    Error:
    {
      "error": str,      // user-facing message (no secrets)
      "configured": bool
    }
    """
    try:
        from ai_service import (
            ask, get_status, GroqServiceError,
            _MAX_QUESTION_CHARS, _MAX_HISTORY_TURNS
        )
    except ImportError:
        from backend.ai_service import (
            ask, get_status, GroqServiceError,
            _MAX_QUESTION_CHARS, _MAX_HISTORY_TURNS
        )

    # --- Input validation ---
    if not request.is_json:
        return jsonify({"error": "Request body must be JSON.", "configured": get_status()["configured"]}), 400

    MAX_REQUEST_BYTES = 32_768
    if request.content_length and request.content_length > MAX_REQUEST_BYTES:
        return jsonify({"error": "Request body too large.", "configured": get_status()["configured"]}), 413

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "JSON payload must be an object.", "configured": get_status()["configured"]}), 400

    question = payload.get("question", "")
    if not isinstance(question, str) or not question.strip():
        return jsonify({"error": "Question cannot be empty.", "configured": get_status()["configured"]}), 400

    question = question.strip()[:_MAX_QUESTION_CHARS]

    history = payload.get("history", [])
    if not isinstance(history, list):
        history = []

    selected_norad_id = payload.get("selected_norad_id")
    selected_event_id = payload.get("selected_event_id")

    # Validate event_id is a known safe string (no injection risk from arbitrary content)
    if selected_event_id is not None and not isinstance(selected_event_id, str):
        selected_event_id = None
    if selected_event_id:
        selected_event_id = selected_event_id[:128]  # bounded

    # --- Build grounding context from trusted backend sources ---
    context = _build_grounding_context(
        selected_norad_id=selected_norad_id,
        selected_event_id=selected_event_id,
    )

    # --- Call Groq ---
    try:
        answer = ask(question=question, history=history, context=context)
    except GroqServiceError as exc:
        status_info = get_status()
        return jsonify({
            "error": str(exc),
            "configured": status_info["configured"],
        }), exc.http_status

    return jsonify({
        "answer": answer,
        "sources": "Akash Setu backend catalogue + Groq LLM (general astrodynamics knowledge)",
        "data_timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "model": get_status().get("model"),
        "configured": get_status()["configured"],
    }), 200

# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Akash Setu - Loading satellite data...")
    print("=" * 60)
    global SATELLITES
    SATELLITES = _load_satellites()

    station_count = sum(1 for s in SATELLITES.values() if s["source"] == "stations")
    active_count  = sum(1 for s in SATELLITES.values() if s["source"] == "active_satellites")
    err_count     = sum(1 for s in SATELLITES.values() if s["init_error"])
    print(f"  Stations:           {station_count}")
    print(f"  Active satellites:  {active_count}")
    print(f"  Init errors:        {err_count}")
    print(f"  Total (deduplicated): {len(SATELLITES)}")
    print("=" * 60)
    print(f"  Frontend: http://localhost:5000/")
    print(f"  API:      http://localhost:5000/api/health")
    print("=" * 60)

    app.run(host="0.0.0.0", port=5000, debug=False)


if __name__ == "__main__":
    main()






