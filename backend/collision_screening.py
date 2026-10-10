"""
collision_screening.py -- Akash Setu: Collision Screening Engine
=========================================================================
Screens candidate satellite pairs for close approaches over a configurable
future time horizon using SGP4-propagated TEME states.

ALGORITHM OVERVIEW
------------------
1. Broad-phase filter  (O(N) per satellite, eliminates most pairs)
   - Compute apogee/perigee of each object from mean-motion & eccentricity.
   - For each pair, test if their altitude bands overlap with a margin equal
     to the screening threshold.  Only overlapping pairs proceed.
2. Coarse time-step scan  (O(P x T) where P = candidate pairs, T = time steps)
   - Propagate both objects in TEME at each time step.
   - Compute inter-object separation in TEME (same inertial frame -- no
     coordinate conversion needed for relative distance).
   - Record time steps where separation < coarse_threshold_km.
3. Refinement (bisection, O(P_close x log2(T_refine)))
   - For each coarse-flagged window, bisect to find minimum separation more
     accurately than the coarse time step allows.
4. Event construction
   - Assign stable event IDs (deterministic hash of pair + epoch).
   - Return structured ScreeningResult records.

COORDINATE FRAMES & UNITS
--------------------------
- SGP4 output: TEME (True Equator Mean Equinox), position in km, velocity km/s.
- Relative distance is computed in TEME (inertial); this is correct because
  both objects share the same frame at each time step.
- ECEF conversion (TEME -> ECEF via GMST) is NOT applied for distance
  calculations -- only for position reporting in results.
- All distances: km.  All velocities: km/s.  Time: minutes (propagation) or
  seconds (relative velocity reporting).
- miss_distance_km  = ||r1 - r2|| at TCA  (km)
- relative_speed_km_s = ||v1 - v2|| at TCA  (km/s)

LIMITATIONS & DISCLAIMERS
--------------------------
- This is a SCREENING tool, not a conjunction assessment or probability tool.
- SGP4 accuracy degrades with TLE age; errors can exceed tens of km after days.
- TEME->ECEF uses approximate GMST (IAU 1982); position error < 1 km for
  visualization but NOT suitable for conjunction probability calculation.
- No atmospheric drag perturbations beyond what SGP4 models.
- No manoeuvre modelling; only ballistic trajectories are screened.
- No calibrated collision probability (Pc) is computed.
- A screening ALERT means the objects' propagated trajectories come within the
  threshold -- it is NOT a confirmed collision risk.
- Stale TLEs (epoch_age > stale_days) generate warnings; results are retained
  but flagged.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

# Re-use the propagation helpers already in app.py by importing them when
# called from Flask context.  In standalone / test contexts we reproduce the
# minimal subset needed here so the module is self-contained.

# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class ScreeningConfig:
    """All tunable parameters for a screening run."""
    horizon_minutes: float = 180.0       # How far ahead to screen (minutes)
    coarse_step_minutes: float = 1.0     # Coarse propagation step (minutes)
    screening_threshold_km: float = 5.0  # Alert threshold (km)
    broad_phase_margin_km: float = 50.0  # Extra margin for broad-phase filter
    refinement_steps: int = 20           # Bisection iterations for TCA refine
    stale_epoch_days: float = 14.0       # Days after which TLE is considered stale
    max_pairs: int = 500_000             # Safety cap on candidate pairs
    threshold_comparison: str = "lt"     # "lt" (< threshold) or "lte" (<= threshold)

    def alert_triggered(self, distance_km: float) -> bool:
        """Return True if distance triggers an alert per the configured rule."""
        if self.threshold_comparison == "lte":
            return distance_km <= self.screening_threshold_km
        return distance_km < self.screening_threshold_km


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ObjectScreeningInfo:
    """Metadata about one object as used in screening."""
    norad_id: int
    name: str
    epoch: str
    epoch_age_days: Optional[float]
    stale: bool
    perigee_km: float
    apogee_km: float
    propagation_available: bool
    init_error: Optional[str] = None


@dataclass
class ScreeningResult:
    """A single close-approach event found by the screening engine."""
    event_id: str
    norad_id_1: int
    name_1: str
    norad_id_2: int
    name_2: str
    # Screening epoch: UTC ISO string (start of screening window)
    screening_epoch: str
    # TCA: time of closest approach (UTC ISO string)
    tca_utc: str
    # Minutes from screening start to TCA
    tca_minutes_from_start: float
    miss_distance_km: float
    relative_speed_km_s: float
    # Approach geometry
    radial_separation_km: float      # along r1 direction at TCA
    along_track_separation_km: float # NOT computed (would need full ECI frame)
    # Data quality
    object_1_stale: bool
    object_2_stale: bool
    screening_status: str            # "ALERT", "SAFE", "SKIPPED_STALE", "ERROR"
    # Limitations notice (always included)
    disclaimer: str = (
        "SCREENING ALERT ONLY. Not a confirmed collision or calibrated probability. "
        "SGP4 accuracy degrades with TLE age. No manoeuvre modelling applied."
    )
    notes: list = field(default_factory=list)


@dataclass
class ScreeningRun:
    """Complete output of one screening run."""
    run_id: str
    screening_epoch: str
    horizon_minutes: float
    coarse_step_minutes: float
    screening_threshold_km: float
    total_objects_input: int
    objects_screened: int
    objects_skipped: int
    total_pairs_possible: int
    pairs_after_broad_phase: int
    pairs_evaluated: int
    alerts: list                  # list of ScreeningResult
    skipped: list                 # list of {norad_id, reason}
    runtime_seconds: float
    config: dict
    disclaimer: str = (
        "Collision screening results are propagated estimates only. "
        "This tool does not produce calibrated collision probabilities. "
        "TLE-based SGP4 propagation is subject to orbital error growth."
    )


# ---------------------------------------------------------------------------
# Core math helpers (self-contained, no Flask dependency)
# ---------------------------------------------------------------------------

EARTH_RADIUS_KM = 6371.0
REVDAY2RADMIN   = 2.0 * math.pi / 1440.0
DEG2RAD         = math.pi / 180.0


def _vec_norm(v) -> float:
    return math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])


def _vec_sub(a, b) -> tuple:
    return (a[0]-b[0], a[1]-b[1], a[2]-b[2])


def _vec_dot(a, b) -> float:
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]


def _parse_epoch_to_dt(epoch_str: str) -> Optional[datetime]:
    if not isinstance(epoch_str, str):
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(epoch_str, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return None


def _epoch_to_jd(epoch_str: str):
    """Convert ISO epoch string to (jd_int, jd_frac)."""
    dt = _parse_epoch_to_dt(epoch_str)
    if dt is None:
        raise ValueError(f"Bad epoch: {epoch_str!r}")
    y, m, d = dt.year, dt.month, dt.day
    h = dt.hour + dt.minute/60.0 + dt.second/3600.0 + dt.microsecond/3.6e9
    if m <= 2:
        y -= 1; m += 12
    A = int(y/100); B = 2 - A + int(A/4)
    jd = int(365.25*(y+4716)) + int(30.6001*(m+1)) + d + h/24.0 + B - 1524.5
    jd_int = int(jd)
    return float(jd_int), float(jd - jd_int)


def _sgp4_epoch_days(epoch_str: str) -> float:
    """SGP4 epoch: days since 1949-12-31 00:00 UT (JD 2433281.5)."""
    jd, jdf = _epoch_to_jd(epoch_str)
    return (jd + jdf) - 2433281.5


def _jd_add_minutes(jd: float, jdf: float, minutes: float):
    """Add minutes to a Julian Date pair, returning normalised (jd, jdf)."""
    frac = jdf + minutes / 1440.0
    return jd + int(frac), frac - int(frac)


def _stable_event_id(norad1: int, norad2: int, epoch_iso: str, tca_min: float) -> str:
    """Deterministic event ID: hash of sorted NORAD pair + epoch + TCA bucket."""
    a, b = sorted([norad1, norad2])
    raw = f"{a}-{b}-{epoch_iso}-{tca_min:.1f}"
    return "EVT-" + hashlib.sha1(raw.encode()).hexdigest()[:12].upper()


# ---------------------------------------------------------------------------
# Broad-phase filter helpers
# ---------------------------------------------------------------------------

def _perigee_apogee(mean_motion_rev_day: float, eccentricity: float):
    """
    Estimate perigee and apogee altitude (km above Earth surface).
    semi-major axis a = (mu / n^2)^(1/3), mu = 398600.4418 km^3/s^2
    n in rad/s = mean_motion_rev_day * 2*pi / 86400
    """
    mu = 398600.4418
    if mean_motion_rev_day <= 0:
        return 0.0, 50000.0  # unknown -- wide band
    n_rad_s = mean_motion_rev_day * 2.0 * math.pi / 86400.0
    try:
        a = (mu / (n_rad_s * n_rad_s)) ** (1.0/3.0)
    except Exception:
        return 0.0, 50000.0
    perigee_km = a * (1.0 - eccentricity) - EARTH_RADIUS_KM
    apogee_km  = a * (1.0 + eccentricity) - EARTH_RADIUS_KM
    return perigee_km, apogee_km


def _altitude_bands_overlap(perigee1, apogee1, perigee2, apogee2, margin_km: float) -> bool:
    """Return True if the two altitude bands overlap when expanded by margin_km."""
    lo1, hi1 = perigee1 - margin_km, apogee1 + margin_km
    lo2, hi2 = perigee2 - margin_km, apogee2 + margin_km
    return hi1 >= lo2 and hi2 >= lo1


# ---------------------------------------------------------------------------
# Satrec wrapper for screening (avoids importing Flask app at test time)
# ---------------------------------------------------------------------------

def _make_satrec_from_record(rec: dict):
    """
    Build a Satrec from a CelesTrak-style dict.
    Returns (satrec, error_str).  Mirrors _init_satrec in app.py.
    """
    try:
        from sgp4.api import Satrec, WGS72
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


def _propagate_teme(satrec, jd: float, jdf: float):
    """
    Propagate satrec to (jd, jdf).
    Returns (r_teme_km, v_teme_km_s) or raises RuntimeError.
    r and v are tuples of 3 floats.
    """
    e, r, v = satrec.sgp4(jd, jdf)
    if e != 0:
        raise RuntimeError(f"SGP4 error code {e}")
    return tuple(r), tuple(v)


# ---------------------------------------------------------------------------
# Main screening engine
# ---------------------------------------------------------------------------

def screen_satellites(
    satellite_records: list[dict],
    screening_dt: datetime,
    config: ScreeningConfig | None = None,
) -> ScreeningRun:
    """
    Run the collision screening pipeline.

    Parameters
    ----------
    satellite_records : list of dicts, each containing at minimum:
        {
          "NORAD_CAT_ID": int-or-str,
          "OBJECT_NAME": str,
          "EPOCH": "YYYY-MM-DDTHH:MM:SS.ffffff",
          "MEAN_MOTION": float (rev/day),
          "ECCENTRICITY": float,
          "INCLINATION": float (deg),
          "RA_OF_ASC_NODE": float (deg),
          "ARG_OF_PERICENTER": float (deg),
          "MEAN_ANOMALY": float (deg),
          "BSTAR": float,
          "MEAN_MOTION_DOT": float,
          "MEAN_MOTION_DDOT": float,
        }
        OR a pre-built internal record (from SATELLITES dict in app.py):
        {
          "norad_id": int,
          "name": str,
          "epoch": str,
          "epoch_age_days": float,
          "mean_motion": float,
          "eccentricity": float,
          "satrec": Satrec-or-None,
          "init_error": str-or-None,
          "stale": bool,
          ...
        }

    screening_dt : datetime (UTC-aware)
    config       : ScreeningConfig (defaults used if None)

    Returns
    -------
    ScreeningRun dataclass
    """
    import time as _time_mod
    t_start = _time_mod.monotonic()

    if config is None:
        config = ScreeningConfig()

    epoch_iso = screening_dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
    jd0, jdf0 = _epoch_to_jd(epoch_iso)

    # -----------------------------------------------------------------------
    # 1. Parse / validate all input records
    # -----------------------------------------------------------------------
    objects: list[dict] = []       # normalised records
    skipped: list[dict] = []
    seen_norads: set = set()

    for raw in satellite_records:
        # Support both CelesTrak-style (NORAD_CAT_ID) and internal (norad_id)
        if "NORAD_CAT_ID" in raw:
            norad = int(raw["NORAD_CAT_ID"])
            name  = raw.get("OBJECT_NAME", f"NORAD-{norad}")
            epoch = raw.get("EPOCH", "")
            mm    = float(raw.get("MEAN_MOTION", 0))
            ecc   = float(raw.get("ECCENTRICITY", 0))
            ea    = None
            ep_dt = _parse_epoch_to_dt(epoch)
            if ep_dt:
                ea = (screening_dt - ep_dt).total_seconds() / 86400.0
            stale = ea is not None and ea > config.stale_epoch_days
            # Build satrec
            satrec, init_err = _make_satrec_from_record(raw)
        else:
            # Internal format
            norad    = int(raw["norad_id"])
            name     = raw.get("name", f"NORAD-{norad}")
            epoch    = raw.get("epoch", "")
            mm       = raw.get("mean_motion", 0) or 0
            ecc      = raw.get("eccentricity", 0) or 0
            ea       = raw.get("epoch_age_days")
            stale    = raw.get("stale", False)
            satrec   = raw.get("satrec")
            init_err = raw.get("init_error")

        # Deduplicate by NORAD ID
        if norad in seen_norads:
            skipped.append({"norad_id": norad, "name": name, "reason": "duplicate"})
            continue
        seen_norads.add(norad)

        if init_err or satrec is None:
            skipped.append({
                "norad_id": norad, "name": name,
                "reason": f"init_error: {init_err or 'satrec=None'}",
            })
            continue

        perigee, apogee = _perigee_apogee(mm, ecc)

        objects.append({
            "norad_id": norad, "name": name, "epoch": epoch,
            "epoch_age_days": ea, "stale": stale,
            "satrec": satrec, "perigee_km": perigee, "apogee_km": apogee,
        })

    n = len(objects)
    total_pairs = n * (n - 1) // 2

    # -----------------------------------------------------------------------
    # 2. Broad-phase filter: altitude band overlap
    # -----------------------------------------------------------------------
    candidate_pairs: list[tuple] = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = objects[i], objects[j]
            if _altitude_bands_overlap(
                a["perigee_km"], a["apogee_km"],
                b["perigee_km"], b["apogee_km"],
                config.broad_phase_margin_km + config.screening_threshold_km,
            ):
                candidate_pairs.append((i, j))

    if len(candidate_pairs) > config.max_pairs:
        candidate_pairs = candidate_pairs[:config.max_pairs]

    # -----------------------------------------------------------------------
    # 3. Coarse time-step scan
    # -----------------------------------------------------------------------
    n_steps = int(config.horizon_minutes / config.coarse_step_minutes) + 1
    time_steps = [i * config.coarse_step_minutes for i in range(n_steps)]

    # Pre-propagate all objects at all time steps (one state cache per object)
    # Only propagate objects that appear in at least one candidate pair
    active_indices: set = set()
    for i, j in candidate_pairs:
        active_indices.add(i); active_indices.add(j)

    # state_cache[obj_index][step_index] = (r_teme, v_teme) or None
    state_cache: dict[int, list] = {}
    for idx in active_indices:
        obj = objects[idx]
        states = []
        for t_min in time_steps:
            jd_t, jdf_t = _jd_add_minutes(jd0, jdf0, t_min)
            try:
                r, v = _propagate_teme(obj["satrec"], jd_t, jdf_t)
                states.append((r, v))
            except Exception:
                states.append(None)
        state_cache[idx] = states

    # Scan pairs
    coarse_threshold = config.screening_threshold_km * 5.0  # wider coarse window
    close_windows: list[dict] = []

    for i, j in candidate_pairs:
        states_i = state_cache[i]
        states_j = state_cache[j]
        min_sep = float("inf")
        min_step = 0
        for s, (si, sj) in enumerate(zip(states_i, states_j)):
            if si is None or sj is None:
                continue
            dr = _vec_sub(si[0], sj[0])
            sep = _vec_norm(dr)
            if sep < min_sep:
                min_sep = sep
                min_step = s
        if min_sep < coarse_threshold:
            close_windows.append({
                "i": i, "j": j,
                "coarse_min_sep_km": min_sep,
                "coarse_min_step": min_step,
            })

    # -----------------------------------------------------------------------
    # 4. Refinement: bisection around minimum coarse step
    # -----------------------------------------------------------------------
    alerts: list[ScreeningResult] = []

    for cw in close_windows:
        i, j = cw["i"], cw["j"]
        obj_a, obj_b = objects[i], objects[j]
        s_min = cw["coarse_min_step"]

        # Bisection window: one step before/after coarse minimum
        t_lo = time_steps[max(0, s_min - 1)]
        t_hi = time_steps[min(len(time_steps) - 1, s_min + 1)]

        best_sep  = float("inf")
        best_t    = (t_lo + t_hi) / 2.0
        best_r_a  = None
        best_v_a  = None
        best_v_b  = None

        for _ in range(config.refinement_steps):
            t_mid = (t_lo + t_hi) / 2.0
            jd_lo_pt, jdf_lo_pt = _jd_add_minutes(jd0, jdf0, t_lo)
            jd_hi_pt, jdf_hi_pt = _jd_add_minutes(jd0, jdf0, t_hi)
            jd_mid,   jdf_mid   = _jd_add_minutes(jd0, jdf0, t_mid)

            try:
                r_a_lo, v_a_lo = _propagate_teme(obj_a["satrec"], jd_lo_pt, jdf_lo_pt)
                r_b_lo, v_b_lo = _propagate_teme(obj_b["satrec"], jd_lo_pt, jdf_lo_pt)
                r_a_hi, v_a_hi = _propagate_teme(obj_a["satrec"], jd_hi_pt, jdf_hi_pt)
                r_b_hi, v_b_hi = _propagate_teme(obj_b["satrec"], jd_hi_pt, jdf_hi_pt)
            except Exception as exc:
                break

            sep_lo = _vec_norm(_vec_sub(r_a_lo, r_b_lo))
            sep_hi = _vec_norm(_vec_sub(r_a_hi, r_b_hi))

            if sep_lo < sep_hi:
                # minimum is in [t_lo, t_mid]
                t_hi = t_mid
                if sep_lo < best_sep:
                    best_sep = sep_lo; best_t = t_lo
                    best_r_a = r_a_lo; best_v_a = v_a_lo; best_v_b = v_b_lo
            else:
                # minimum is in [t_mid, t_hi]
                t_lo = t_mid
                if sep_hi < best_sep:
                    best_sep = sep_hi; best_t = t_hi
                    best_r_a = r_a_hi; best_v_a = v_a_hi; best_v_b = v_b_hi

        # Compute relative velocity at TCA
        if best_v_a is not None and best_v_b is not None:
            dv     = _vec_sub(best_v_a, best_v_b)
            rel_v  = _vec_norm(dv)
        else:
            rel_v = float("nan")

        # Radial separation component (projection of dr onto r_a unit vector)
        radial_sep = float("nan")
        if best_r_a is not None and best_r_a:
            r_a_norm = _vec_norm(best_r_a)
            if r_a_norm > 0:
                # TCA position of B relative to A in A's radial direction
                # We use the coarse-minimum positions for this
                s_min_idx = cw["coarse_min_step"]
                if state_cache[i][s_min_idx] and state_cache[j][s_min_idx]:
                    ra_c = state_cache[i][s_min_idx][0]
                    rb_c = state_cache[j][s_min_idx][0]
                    dr_c = _vec_sub(rb_c, ra_c)
                    r_unit = tuple(x / r_a_norm for x in best_r_a)
                    radial_sep = abs(_vec_dot(dr_c, r_unit))

        # Construct TCA datetime
        tca_dt = datetime.fromtimestamp(
            screening_dt.timestamp() + best_t * 60.0,
            tz=timezone.utc
        )
        tca_iso = tca_dt.strftime("%Y-%m-%dT%H:%M:%S.%f")

        status = "ALERT" if config.alert_triggered(best_sep) else "SAFE"
        notes  = []
        if obj_a["stale"]:
            notes.append(f"Object 1 ({obj_a['name']}) has stale TLE (>{config.stale_epoch_days}d old)")
        if obj_b["stale"]:
            notes.append(f"Object 2 ({obj_b['name']}) has stale TLE (>{config.stale_epoch_days}d old)")

        eid = _stable_event_id(obj_a["norad_id"], obj_b["norad_id"], epoch_iso, best_t)

        result = ScreeningResult(
            event_id=eid,
            norad_id_1=obj_a["norad_id"], name_1=obj_a["name"],
            norad_id_2=obj_b["norad_id"], name_2=obj_b["name"],
            screening_epoch=epoch_iso,
            tca_utc=tca_iso,
            tca_minutes_from_start=round(best_t, 4),
            miss_distance_km=round(best_sep, 4),
            relative_speed_km_s=round(rel_v, 6),
            radial_separation_km=round(radial_sep, 4) if not math.isnan(radial_sep) else float("nan"),
            along_track_separation_km=float("nan"),  # Not computed in this version
            object_1_stale=obj_a["stale"],
            object_2_stale=obj_b["stale"],
            screening_status=status,
            notes=notes,
        )
        alerts.append(result)

    # Sort alerts: ALERT first, then by miss distance
    alerts.sort(key=lambda r: (0 if r.screening_status == "ALERT" else 1, r.miss_distance_km))

    t_end = _time_mod.monotonic()

    run_id = "RUN-" + hashlib.sha1(epoch_iso.encode()).hexdigest()[:10].upper()

    return ScreeningRun(
        run_id=run_id,
        screening_epoch=epoch_iso,
        horizon_minutes=config.horizon_minutes,
        coarse_step_minutes=config.coarse_step_minutes,
        screening_threshold_km=config.screening_threshold_km,
        total_objects_input=len(satellite_records),
        objects_screened=n,
        objects_skipped=len(skipped),
        total_pairs_possible=total_pairs,
        pairs_after_broad_phase=len(candidate_pairs),
        pairs_evaluated=len(close_windows),
        alerts=alerts,
        skipped=skipped,
        runtime_seconds=round(t_end - t_start, 3),
        config={
            "horizon_minutes": config.horizon_minutes,
            "coarse_step_minutes": config.coarse_step_minutes,
            "screening_threshold_km": config.screening_threshold_km,
            "broad_phase_margin_km": config.broad_phase_margin_km,
            "refinement_steps": config.refinement_steps,
            "stale_epoch_days": config.stale_epoch_days,
            "threshold_comparison": config.threshold_comparison,
        },
    )
