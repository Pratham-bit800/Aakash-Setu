"""
grid_analysis.py  --  Akash Setu Phase 4 (validation pass 2026-10-10)
=======================================================================

REFERENCE-DOCUMENT STATUS
--------------------------
No external reference document was provided with the Phase 4 specification.
If a specific reference is later supplied this module must be reviewed against
it and any deviations corrected before operational use.

LITERATURE BASIS  (none of these is "the supplied reference")
  Alfano 2005   -- uniform time-grid scan as Step 1 of TCA determination
  Hoots 1984    -- TCA bracketing by interval narrowing
  Vallado 2013  -- relative motion and conjunction geometry in TEME, Sec 9.5

ALGORITHM
---------
Level 0:
  Grid on [t_tca - hw, t_tca + hw], n equally spaced points, dt_0 = 2*hw/(n-1)
  Find t* = argmin  rho(t)  where  rho(t) = ||r1(t) - r2(t)||   [km, TEME]

Level k >= 1:
  dt_k = dt_{k-1} / refinement_factor          (Eq 3: refinement_factor is used here)
  hw_k = dt_k * (n - 1) / 2                    (same n, smaller window)
  center <- previous t*
  Find new t*, new best_sep

Stop when (evaluated in priority order):
  (a) sigma_spatial(dt_k) <= target  =>  stop_reason = STOP_RESOLUTION; converged = True
  (b) |sep_k - sep_{k-1}| < tol     =>  stop_reason = STOP_TOLERANCE;   converged = False
  (c) hw_next <= 0 or numeric underflow => stop_reason = STOP_WINDOW; converged = False
  (d) max_iterations reached         =>  stop_reason = STOP_MAX_ITER;   converged = False

converged is True ONLY for (a).  The other three stops are explicitly distinguished.

EQUATIONS
---------
  sigma_spatial [m] = dt [s] * v_rel [km/s] * 1000          (Eq 1)
  dt_target [s]     = target_m / (v_rel [km/s] * 1000)      (Eq 2)
  dt_{k+1} [min]    = dt_k [min] / refinement_factor         (Eq 3)
  hw_{k+1} [min]    = dt_{k+1} * (n - 1) / 2               (Eq 4)

UNITS
-----
  Time: minutes (propagation), seconds (dt output field)
  Distance: km (propagation), metres (spatial_resolution_m output)
  Speed: km/s
"""

from __future__ import annotations

import math
import time as _time_mod
from dataclasses import dataclass, field
from datetime import datetime, timezone

from collision_screening import (
    _epoch_to_jd,
    _jd_add_minutes,
    _propagate_teme,
    _vec_norm,
    _vec_sub,
    ScreeningResult,
    screen_satellites,
)

# ---------------------------------------------------------------------------
# Stop-reason constants  (machine-readable, mutually exclusive)
# ---------------------------------------------------------------------------

STOP_RESOLUTION = "RESOLUTION_ACHIEVED"    # target sigma_spatial met; converged=True
STOP_TOLERANCE  = "TOLERANCE_CONVERGENCE"  # |sep_change| < tol; converged=False
STOP_WINDOW     = "WINDOW_DEGENERATE"      # numeric underflow; converged=False
STOP_MAX_ITER   = "MAX_ITERATIONS"         # safety cap; converged=False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class GridConfig:
    """Configuration for the nested-grid TCA refinement.

    refinement_factor: int
        Ratio by which dt shrinks each level.  Must be >= 2.
        dt_{k+1} = dt_k / refinement_factor  (Eq 3).
        Previously declared but ignored; now enforced in the narrowing step.

    convergence_tolerance_km: float
        Stop with STOP_TOLERANCE when |sep_k - sep_{k-1}| < tol.
        Set to float('inf') to disable; useful for isolating STOP_RESOLUTION.

    target_spatial_resolution_m: float
        Stop with STOP_RESOLUTION (converged=True) when sigma_spatial <= this.
        Set to float('inf') to disable; algorithm will stop by tolerance or
        max_iterations.
    """
    search_half_window_minutes: float = 2.0
    initial_grid_points: int = 20
    refinement_factor: int = 10
    target_spatial_resolution_m: float = 0.01
    max_iterations: int = 12
    convergence_tolerance_km: float = 1e-9
    include_iteration_log: bool = False


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class GridAnalysisResult:
    """Result of one grid-analysis refinement on a single satellite pair."""

    # Phase-3 identity
    event_id: str
    norad_id_1: int
    name_1: str
    norad_id_2: int
    name_2: str
    screening_epoch: str

    # Phase-3 inputs
    phase3_tca_minutes: float
    phase3_miss_distance_km: float

    # Refined outputs
    refined_tca_minutes: float
    refined_tca_utc: str
    refined_miss_distance_km: float
    refined_miss_distance_m: float

    # Grid resolution
    final_dt_seconds: float
    spatial_resolution_m: float      # sigma_spatial = final_dt_s * v_rel * 1000

    # Stop condition (distinct, mutually exclusive)
    stop_reason: str                 # STOP_RESOLUTION | STOP_TOLERANCE | STOP_WINDOW | STOP_MAX_ITER
    converged: bool                  # True ONLY when stop_reason == STOP_RESOLUTION
    convergence_reason: str          # Human-readable elaboration

    # Kinematics
    relative_speed_km_s: float

    # Delta vs Phase-3
    tca_delta_seconds: float
    miss_distance_improvement_m: float

    # Quality flags
    phase3_stale_warning: bool

    # Optional  (have defaults)
    iterations_used: int = 0
    iteration_log: list = field(default_factory=list)
    runtime_seconds: float = 0.0
    accuracy_disclaimer: str = (
        "Grid resolution is a NUMERICAL property of the algorithm, NOT a measure "
        "of physical prediction accuracy. SGP4 propagation errors for typical TLEs "
        "range from ~100 m (fresh TLE, 1 day) to >10 km (stale TLE, 14 days). "
        "A 1-cm grid resolution does not imply 1-cm positional knowledge. "
        "Results are not suitable for operational collision avoidance."
    )


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _grid_tca_search(satrec_a, satrec_b, jd0, jdf0, center_minutes, half_window_minutes, n_points):
    """Uniform grid search over [center-hw, center+hw].

    Returns (best_t_min, min_sep_km, rel_speed_km_s, dt_min).
    """
    if n_points < 2 or half_window_minutes <= 0.0:
        jd_t, jdf_t = _jd_add_minutes(jd0, jdf0, center_minutes)
        try:
            r_a, v_a = _propagate_teme(satrec_a, jd_t, jdf_t)
            r_b, v_b = _propagate_teme(satrec_b, jd_t, jdf_t)
            return center_minutes, _vec_norm(_vec_sub(r_a, r_b)), _vec_norm(_vec_sub(v_a, v_b)), 0.0
        except RuntimeError:
            return center_minutes, float("inf"), float("nan"), 0.0

    dt = (2.0 * half_window_minutes) / (n_points - 1)
    t_lo = center_minutes - half_window_minutes
    best_t, best_sep, best_rv = center_minutes, float("inf"), float("nan")

    for i in range(n_points):
        t = t_lo + i * dt
        jd_t, jdf_t = _jd_add_minutes(jd0, jdf0, t)
        try:
            r_a, v_a = _propagate_teme(satrec_a, jd_t, jdf_t)
            r_b, v_b = _propagate_teme(satrec_b, jd_t, jdf_t)
        except RuntimeError:
            continue
        sep = _vec_norm(_vec_sub(r_a, r_b))
        if sep < best_sep:
            best_sep = sep
            best_t = t
            best_rv = _vec_norm(_vec_sub(v_a, v_b))

    return best_t, best_sep, best_rv, dt


# ---------------------------------------------------------------------------
# Core refinement
# ---------------------------------------------------------------------------

def refine_tca(satrec_a, satrec_b, jd0, jdf0, phase3_tca_minutes,
               relative_speed_km_s, config=None):
    """Nested-grid TCA refinement.

    Returns
    -------
    (refined_t_min, refined_sep_km, final_dt_s, spatial_res_m,
     iterations, converged, stop_reason, convergence_reason, iteration_log)
    """
    if config is None:
        config = GridConfig()

    n = max(config.initial_grid_points, 2)
    rf = max(config.refinement_factor, 2)
    rel_speed_safe = max(relative_speed_km_s, 1e-9)

    # Eq 2: target dt for desired spatial resolution
    target_dt_s   = config.target_spatial_resolution_m / (rel_speed_safe * 1000.0)
    target_dt_min  = target_dt_s / 60.0

    hw   = config.search_half_window_minutes
    center = phase3_tca_minutes
    current_dt_min = (2.0 * hw) / (n - 1)

    best_t   = center
    best_sep = float("inf")
    prev_sep = float("inf")
    best_rv  = relative_speed_km_s

    stop_reason        = STOP_MAX_ITER
    convergence_reason = "max_iterations reached without meeting any stopping criterion"
    iteration_log      = []
    iterations         = 0

    for it in range(config.max_iterations):
        t_new, sep_new, rv_new, dt_used = _grid_tca_search(
            satrec_a, satrec_b, jd0, jdf0, center, hw, n
        )
        current_dt_min = dt_used
        iterations = it + 1

        if config.include_iteration_log:
            iteration_log.append({
                "iteration": it,
                "center_min":        round(center, 10),
                "half_window_min":   round(hw, 10),
                "dt_seconds":        round(dt_used * 60.0, 12),
                "spatial_resolution_m": round(dt_used * 60.0 * rel_speed_safe * 1000.0, 8),
                "best_sep_km":       round(sep_new, 10),
                "n_points":          n,
            })

        if sep_new < best_sep:
            best_sep = sep_new
            best_t   = t_new
            if math.isfinite(rv_new):
                best_rv = rv_new

        # ---- Stopping criterion (a): spatial resolution achieved ----
        achieved_dt_s  = current_dt_min * 60.0
        spatial_res_m  = achieved_dt_s * rel_speed_safe * 1000.0
        if (math.isfinite(config.target_spatial_resolution_m)
                and spatial_res_m <= config.target_spatial_resolution_m):
            stop_reason = STOP_RESOLUTION
            convergence_reason = (
                f"RESOLUTION_ACHIEVED: target {config.target_spatial_resolution_m:.4g} m "
                f"met (sigma_spatial = {spatial_res_m:.4g} m) after {iterations} iteration(s)"
            )
            break

        # ---- Stopping criterion (b): flat range function ----
        sep_change = abs(prev_sep - best_sep)
        if (it >= 1 and math.isfinite(config.convergence_tolerance_km)
                and sep_change < config.convergence_tolerance_km):
            stop_reason = STOP_TOLERANCE
            convergence_reason = (
                f"TOLERANCE_CONVERGENCE: |sep_change| = {sep_change:.3e} km "
                f"< tol {config.convergence_tolerance_km:.3e} km after {iterations} iteration(s). "
                f"Target spatial resolution {config.target_spatial_resolution_m:.4g} m "
                f"was NOT achieved (current {spatial_res_m:.4g} m). "
                "Range function is numerically flat at this scale; further refinement "
                "is limited by double-float arithmetic or SGP4 internal precision."
            )
            break
        prev_sep = best_sep

        # ---- Narrow window for next level (Eq 3 and Eq 4) ----
        dt_next_min = current_dt_min / rf              # Eq 3: refinement_factor USED HERE
        hw_next     = dt_next_min * (n - 1) / 2.0     # Eq 4

        if hw_next <= 0.0 or not math.isfinite(hw_next):
            stop_reason = STOP_WINDOW
            convergence_reason = (
                f"WINDOW_DEGENERATE: dt_next = {dt_next_min:.3e} min is non-positive or "
                f"non-finite after {iterations} iteration(s); likely floating-point underflow."
            )
            break

        hw     = hw_next
        center = best_t

    final_dt_s    = current_dt_min * 60.0
    spatial_res_m = final_dt_s * rel_speed_safe * 1000.0
    converged     = (stop_reason == STOP_RESOLUTION)

    return (best_t, best_sep, final_dt_s, spatial_res_m,
            iterations, converged, stop_reason, convergence_reason, iteration_log)


# ---------------------------------------------------------------------------
# Top-level public functions
# ---------------------------------------------------------------------------

def analyse_encounter(phase3_result, satellite_records, screening_dt, grid_config=None):
    """Apply grid-based TCA refinement to a single Phase-3 ScreeningResult."""
    t_start = _time_mod.monotonic()
    if grid_config is None:
        grid_config = GridConfig()

    epoch_iso = screening_dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
    jd0, jdf0 = _epoch_to_jd(epoch_iso)
    norad1, norad2 = phase3_result.norad_id_1, phase3_result.norad_id_2

    def _find_satrec(records, norad):
        for rec in records:
            nid = int(rec.get("NORAD_CAT_ID", rec.get("norad_id", -1)))
            if nid == norad:
                if "satrec" in rec:
                    return rec["satrec"]
                from collision_screening import _make_satrec_from_record
                sat, err = _make_satrec_from_record(rec)
                if err:
                    raise ValueError(f"Cannot build satrec for NORAD {norad}: {err}")
                return sat
        raise KeyError(f"NORAD {norad} not found in satellite_records")

    def _err_result(exc):
        return GridAnalysisResult(
            event_id=phase3_result.event_id,
            norad_id_1=norad1, name_1=phase3_result.name_1,
            norad_id_2=norad2, name_2=phase3_result.name_2,
            screening_epoch=epoch_iso,
            phase3_tca_minutes=phase3_result.tca_minutes_from_start,
            phase3_miss_distance_km=phase3_result.miss_distance_km,
            refined_tca_minutes=phase3_result.tca_minutes_from_start,
            refined_tca_utc=phase3_result.tca_utc,
            refined_miss_distance_km=float("nan"), refined_miss_distance_m=float("nan"),
            final_dt_seconds=float("nan"), spatial_resolution_m=float("nan"),
            stop_reason=STOP_MAX_ITER, converged=False,
            convergence_reason=f"ERROR: {exc}",
            relative_speed_km_s=phase3_result.relative_speed_km_s,
            tca_delta_seconds=0.0, miss_distance_improvement_m=0.0,
            phase3_stale_warning=(phase3_result.object_1_stale or phase3_result.object_2_stale),
            iterations_used=0,
            runtime_seconds=round(_time_mod.monotonic() - t_start, 6),
        )

    try:
        satrec_a = _find_satrec(satellite_records, norad1)
        satrec_b = _find_satrec(satellite_records, norad2)
    except (KeyError, ValueError) as exc:
        return _err_result(exc)

    rel_speed = phase3_result.relative_speed_km_s
    tca_min   = phase3_result.tca_minutes_from_start

    (refined_t, refined_sep, final_dt_s, spatial_res_m,
     iters, converged, stop_reason, convergence_reason, iter_log) = refine_tca(
        satrec_a, satrec_b, jd0, jdf0, tca_min, rel_speed, grid_config
    )

    refined_tca_dt  = datetime.fromtimestamp(
        screening_dt.timestamp() + refined_t * 60.0, tz=timezone.utc
    )
    return GridAnalysisResult(
        event_id=phase3_result.event_id,
        norad_id_1=norad1, name_1=phase3_result.name_1,
        norad_id_2=norad2, name_2=phase3_result.name_2,
        screening_epoch=epoch_iso,
        phase3_tca_minutes=tca_min,
        phase3_miss_distance_km=phase3_result.miss_distance_km,
        refined_tca_minutes=round(refined_t, 10),
        refined_tca_utc=refined_tca_dt.strftime("%Y-%m-%dT%H:%M:%S.%f"),
        refined_miss_distance_km=round(refined_sep, 9),
        refined_miss_distance_m=round(refined_sep * 1000.0, 6),
        final_dt_seconds=round(final_dt_s, 12),
        spatial_resolution_m=round(spatial_res_m, 6),
        stop_reason=stop_reason,
        converged=converged,
        convergence_reason=convergence_reason,
        relative_speed_km_s=round(rel_speed, 6),
        tca_delta_seconds=round((refined_t - tca_min) * 60.0, 6),
        miss_distance_improvement_m=round((phase3_result.miss_distance_km - refined_sep) * 1000.0, 6),
        phase3_stale_warning=(phase3_result.object_1_stale or phase3_result.object_2_stale),
        iterations_used=iters,
        iteration_log=iter_log if grid_config.include_iteration_log else [],
        runtime_seconds=round(_time_mod.monotonic() - t_start, 6),
    )


def analyse_all_alerts(screening_run, satellite_records, screening_dt,
                       grid_config=None, alerts_only=True):
    """Apply grid analysis to all (or ALERT-only) results from a Phase-3 run."""
    return [
        analyse_encounter(alert, satellite_records, screening_dt, grid_config)
        for alert in screening_run.alerts
        if not (alerts_only and alert.screening_status != "ALERT")
    ]

