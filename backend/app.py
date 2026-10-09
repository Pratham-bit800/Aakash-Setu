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

SATELLITES = {}  # populated at startup

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
        return jsonify({"error": "norad_ids required (comma-separated)"}), 400

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
# Phase 3: Collision Screening Endpoints
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
# Startup
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



