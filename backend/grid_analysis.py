"""
grid_analysis.py  --  Akash Setu Phase 4: Grid-Based Collision Analysis
=========================================================================

REFERENCE-DOCUMENT STATUS
--------------------------
No external reference document was provided with the Phase 4 specification.
The task description states "from the supplied reference" but no paper, report,
or specification was attached to the request.

This module therefore documents its method from first principles, clearly
distinguishing what is derivable from the orbital-mechanics literature from
what has been assumed locally. If a specific reference document is later
supplied, this module must be reviewed against it and any deviations corrected.

WHAT "GRID-BASED ANALYSIS" MEANS HERE
--------------------------------------
Phase 3 (collision_screening.py) finds the Time of Closest Approach (TCA) using
a coarse time-step scan followed by bisection refinement. The bisection converges
geometrically but does not directly control the spatial resolution of the result.

Phase 4 applies a nested, multi-resolution uniform time grid local to the Phase 3
TCA estimate, iteratively refining the grid spacing until:
  (a) the desired spatial resolution is achieved, or
  (b) the separation difference between successive refinements is below a
      convergence tolerance, or
  (c) SGP4 propagation precision or double-float arithmetic is the limiting factor.

This is a standard method for TCA determination by repeated grid subdivision
around the minimum of the range function rho(t) = ||r1(t) - r2(t)||,
consistent with the approach described in:
  - Alfano, S. (2005). "A Numerical Implementation of Spherical Object Collision
    Probability." Journal of the Astronautical Sciences, 53(1), 103-109.
    (grid search over time is Step 1 of Alfano's method)
  - Hoots, F. R., Crawford, L. L., & Roehrich, R. L. (1984). "An Analytic Method
    to Determine Future Close Approaches Between Satellites." Celestial Mechanics,
    33(2), 143-158. (TCA bracketing method)
  - Vallado, D. A. (2013). "Fundamentals of Astrodynamics and Applications", 4th
    ed. Section 9.5 (relative motion and conjunction geometry).

None of these was "the supplied reference" since none was supplied; they are
cited here as the standard literature basis for this class of method.

ALGORITHM DETAILS
-----------------
Given: two Satrec objects, a Phase-3 TCA estimate t_tca [minutes from epoch],
       and a search half-window W [minutes].

Step 1. Coarse grid:  t ? {t_tca - W, ..., t_tca + W}  with spacing dt_0
Step 2. Find t* = argmin_{t in grid} rho(t)
Step 3. Set new window [t* - dt_0, t* + dt_0], divide spacing by factor F
Step 4. Repeat steps 2-3 until dt < dt_target OR |rho_prev - rho_curr| < tol
Step 5. Return: tca_refined, miss_distance, grid_resolution_minutes,
                spatial_resolution_m, iterations, converged

SPATIAL RESOLUTION vs PREDICTION ACCURACY
------------------------------------------
Grid resolution (seconds) x relative_speed (km/s) = spatial resolution (km).
For LEO at ~10 km/s relative speed, 1-cm spatial grid resolution requires
dt ~ 1e-5 / 10 = 1e-6 s (1 microsecond).

IMPORTANT: sub-millimetre grid resolution does NOT imply sub-millimetre
prediction accuracy. SGP4 propagation error for a fresh TLE (epoch_age < 1 day)
is typically ~100 m to ~1 km. For older TLEs the error can exceed tens of km.
Grid resolution is a property of the numerical algorithm, not of the physics.
This distinction is documented in every result record.

COMPUTATIONAL FEASIBILITY OF 1-cm RESOLUTION
---------------------------------------------
At 1-cm spatial resolution and ~10 km/s relative speed:
  dt_target = 0.01 / (10 x 1000) s = 1e-6 s = 1e-6/60 minutes
With F=10 and starting dt=1 min, reaching dt_target requires:
  log10(60/1e-6) ~ 8 refinement levels, each calling 2*N+1 SGP4 evaluations
  (N = initial grid points per level, default 20 = 41 calls per level).
  Total SGP4 calls: ~8 * 41 = ~330 -- fully feasible in sub-second time.

This is consistent with the constraint "Avoid constructing a centimetre-resolution
grid across the entire orbital environment" -- we ONLY apply the fine grid within
the [t_tca - W, t_tca + W] window identified by Phase 3.

UNITS
-----
  Time:     minutes (propagation), seconds (output dt)
  Distance: km (propagation), metres (spatial_resolution_m output)
  Speed:    km/s
"""

from __future__ import annotations

import math
import time as _time_mod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

# Re-use helpers from collision_screening (already importable since
# backend/ is on sys.path at runtime)
from collision_screening import (
    _epoch_to_jd,
    _jd_add_minutes,
    _propagate_teme,
    _vec_norm,
    _vec_sub,
    ScreeningResult,
    ScreeningConfig,
    screen_satellites,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class GridConfig:
    """Configuration for the nested-grid TCA refinement."""
    # Half-window around Phase-3 TCA to search [minutes]
    search_half_window_minutes: float = 2.0
    # Initial number of equally-spaced grid points within the window
    initial_grid_points: int = 20
    # Divisor applied to dt at each refinement level (>= 2)
    refinement_factor: int = 10
    # Target spatial resolution [metres].  The algorithm stops when
    # dt * relative_speed <= target_spatial_resolution_m / 1000 km.
    target_spatial_resolution_m: float = 0.01    # 1 cm default
    # Maximum refinement iterations (safety cap)
    max_iterations: int = 12
    # Convergence tolerance: stop if |rho_prev - rho_curr| < tol [km]
    convergence_tolerance_km: float = 1e-9
    # Whether to include per-iteration diagnostics in results
    include_iteration_log: bool = False


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class GridAnalysisResult:
    """Result of one grid-analysis run on a single pair."""
    # Parent Phase-3 event metadata
    event_id: str
    norad_id_1: int
    name_1: str
    norad_id_2: int
    name_2: str
    screening_epoch: str               # ISO UTC of Phase-3 run start

    # Phase-3 inputs to the grid analysis
    phase3_tca_minutes: float          # TCA from Phase 3 [min from epoch]
    phase3_miss_distance_km: float

    # Grid-analysis outputs
    refined_tca_minutes: float         # Refined TCA [min from epoch]
    refined_tca_utc: str               # Refined TCA as UTC ISO string
    refined_miss_distance_km: float    # Miss distance at refined TCA [km]
    refined_miss_distance_m: float     # Same, in metres

    # Grid resolution achieved
    final_dt_seconds: float            # Final grid spacing [seconds]
    spatial_resolution_m: float        # Achieved spatial resolution [m]
    # = final_dt_seconds * relative_speed_km_s * 1000

    # Convergence
    iterations_used: int
    converged: bool
    convergence_reason: str            # Why the algorithm stopped

    # Relative velocity at refined TCA
    relative_speed_km_s: float

    # Difference from Phase-3 result
    tca_delta_seconds: float           # refined_tca - phase3_tca [s]
    miss_distance_improvement_m: float # phase3 - refined, metres (positive = finer result)

    # Quality flags
    phase3_stale_warning: bool

    # Per-iteration log (only if GridConfig.include_iteration_log=True)
    iteration_log: list = field(default_factory=list)

    # Runtime
    runtime_seconds: float = 0.0

    # MANDATORY disclaimer
    accuracy_disclaimer: str = (
        "Grid resolution is a NUMERICAL property of the algorithm, NOT a measure "
        "of physical prediction accuracy. SGP4 propagation errors for typical TLEs "
        "range from ~100 m (fresh TLE, 1 day) to >10 km (stale TLE, 14 days). "
        "A 1-cm grid resolution does not imply 1-cm positional knowledge. "
        "Results are not suitable for operational collision avoidance."
    )


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------

def _grid_tca_search(
    satrec_a,
    satrec_b,
    jd0: float,
    jdf0: float,
    center_minutes: float,
    half_window_minutes: float,
    n_points: int,
) -> tuple[float, float, float, float]:
    """
    Uniform grid search for minimum separation within [center - hw, center + hw].

    Returns (best_t_minutes, min_sep_km, rel_speed_km_s, dt_minutes).
    rel_speed is computed at the best point.
    """
    dt = (2.0 * half_window_minutes) / max(n_points - 1, 1)
    t_lo = center_minutes - half_window_minutes
    best_t = center_minutes
    best_sep = float("inf")
    best_rv = float("nan")

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
            # Relative velocity at this point
            dv = _vec_sub(v_a, v_b)
            best_rv = _vec_norm(dv)

    return best_t, best_sep, best_rv, dt


def refine_tca(
    satrec_a,
    satrec_b,
    jd0: float,
    jdf0: float,
    phase3_tca_minutes: float,
    relative_speed_km_s: float,
    config: GridConfig | None = None,
) -> tuple[float, float, float, float, int, bool, str, list]:
    """
    Apply nested grid refinement around a Phase-3 TCA estimate.

    Parameters
    ----------
    satrec_a, satrec_b : sgp4 Satrec objects
    jd0, jdf0          : Julian Date pair for the screening epoch (t=0)
    phase3_tca_minutes : Phase-3 TCA estimate [minutes from epoch]
    relative_speed_km_s: Phase-3 relative speed estimate (used for spatial resolution calc)
    config             : GridConfig (uses defaults if None)

    Returns
    -------
    (refined_t_min, refined_sep_km, final_dt_s, spatial_res_m,
     iterations, converged, reason, iteration_log)
    """
    if config is None:
        config = GridConfig()

    # Convert target spatial resolution to target dt [minutes]
    # spatial_res_m = dt_seconds * rel_speed_km_s * 1000
    # => dt_seconds = spatial_res_m / (rel_speed_km_s * 1000)
    # Use at least 1e-10 m/s to avoid /0
    rel_speed_safe = max(relative_speed_km_s, 1e-6)  # km/s
    target_dt_s = config.target_spatial_resolution_m / (rel_speed_safe * 1000.0)
    target_dt_min = target_dt_s / 60.0

    center = phase3_tca_minutes
    hw = config.search_half_window_minutes
    n = config.initial_grid_points

    best_t = center
    best_sep = float("inf")
    prev_sep = float("inf")
    best_rv = relative_speed_km_s
    current_dt_min = (2.0 * hw) / max(n - 1, 1)

    iteration_log = []
    converged = False
    reason = "max_iterations reached"
    iterations = 0

    for it in range(config.max_iterations):
        t_new, sep_new, rv_new, dt_used = _grid_tca_search(
            satrec_a, satrec_b, jd0, jdf0, center, hw, n
        )
        current_dt_min = dt_used

        if config.include_iteration_log:
            spatial_m = dt_used * 60.0 * rel_speed_safe * 1000.0
            iteration_log.append({
                "iteration": it,
                "center_min": round(center, 9),
                "half_window_min": round(hw, 9),
                "dt_min": dt_used,
                "dt_seconds": round(dt_used * 60, 12),
                "spatial_resolution_m": round(spatial_m, 6),
                "best_sep_km": round(sep_new, 9),
                "n_points": n,
            })

        # Update best
        if sep_new < best_sep:
            best_sep = sep_new
            best_t = t_new
            if math.isfinite(rv_new):
                best_rv = rv_new

        # Convergence checks
        sep_change = abs(prev_sep - best_sep)
        prev_sep = best_sep
        iterations = it + 1

        # Check spatial resolution achieved
        achieved_dt_s = current_dt_min * 60.0
        spatial_res_m = achieved_dt_s * rel_speed_safe * 1000.0
        if spatial_res_m <= config.target_spatial_resolution_m:
            converged = True
            reason = (
                f"target spatial resolution {config.target_spatial_resolution_m:.4g} m "
                f"achieved at {spatial_res_m:.4g} m after {iterations} iterations"
            )
            break

        # Convergence by separation change
        if it > 0 and sep_change < config.convergence_tolerance_km:
            converged = True
            reason = (
                f"convergence: |sep_change| = {sep_change:.3e} km < "
                f"tol {config.convergence_tolerance_km:.3e} km after {iterations} iterations"
            )
            break

        # Narrow the window around the new best point
        # New half-window = old dt (one step on either side)
        hw = max(current_dt_min, target_dt_min * 0.5)
        # Refine: same number of points over smaller window
        center = best_t
        # Keep n points but over a smaller window -> finer dt
        # If window is already at target dt, stop narrowing
        if hw <= target_dt_min:
            converged = True
            reason = f"window <= target_dt after {iterations} iterations"
            break

    final_dt_s = current_dt_min * 60.0
    spatial_res_m = final_dt_s * rel_speed_safe * 1000.0

    return (best_t, best_sep, final_dt_s, spatial_res_m,
            iterations, converged, reason, iteration_log)


# ---------------------------------------------------------------------------
# Top-level analysis function
# ---------------------------------------------------------------------------

def analyse_encounter(
    phase3_result: ScreeningResult,
    satellite_records: list,
    screening_dt: datetime,
    grid_config: GridConfig | None = None,
) -> GridAnalysisResult:
    """
    Apply grid-based TCA refinement to a single Phase-3 ScreeningResult.

    Parameters
    ----------
    phase3_result    : A ScreeningResult from Phase 3 (collision_screening.py)
    satellite_records: The same list passed to screen_satellites()
    screening_dt     : The UTC datetime used for the Phase-3 run
    grid_config      : GridConfig (defaults if None)

    Returns
    -------
    GridAnalysisResult with refined TCA and miss distance
    """
    t_start = _time_mod.monotonic()

    if grid_config is None:
        grid_config = GridConfig()

    epoch_iso = screening_dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
    jd0, jdf0 = _epoch_to_jd(epoch_iso)

    # Locate the two Satrec objects from the record list
    norad1 = phase3_result.norad_id_1
    norad2 = phase3_result.norad_id_2

    def _find_satrec(records, norad):
        for rec in records:
            nid = int(rec.get("NORAD_CAT_ID", rec.get("norad_id", -1)))
            if nid == norad:
                if "satrec" in rec:
                    return rec["satrec"]
                # CelesTrak-style: build on the fly
                from collision_screening import _make_satrec_from_record
                sat, err = _make_satrec_from_record(rec)
                if err:
                    raise ValueError(f"Cannot build satrec for NORAD {norad}: {err}")
                return sat
        raise KeyError(f"NORAD {norad} not found in satellite_records")

    try:
        satrec_a = _find_satrec(satellite_records, norad1)
        satrec_b = _find_satrec(satellite_records, norad2)
    except (KeyError, ValueError) as exc:
        # Return a degenerate result with error info
        t_end = _time_mod.monotonic()
        return GridAnalysisResult(
            event_id=phase3_result.event_id,
            norad_id_1=norad1, name_1=phase3_result.name_1,
            norad_id_2=norad2, name_2=phase3_result.name_2,
            screening_epoch=epoch_iso,
            phase3_tca_minutes=phase3_result.tca_minutes_from_start,
            phase3_miss_distance_km=phase3_result.miss_distance_km,
            refined_tca_minutes=phase3_result.tca_minutes_from_start,
            refined_tca_utc=phase3_result.tca_utc,
            refined_miss_distance_km=float("nan"),
            refined_miss_distance_m=float("nan"),
            final_dt_seconds=float("nan"),
            spatial_resolution_m=float("nan"),
            iterations_used=0,
            converged=False,
            convergence_reason=f"ERROR: {exc}",
            relative_speed_km_s=phase3_result.relative_speed_km_s,
            tca_delta_seconds=0.0,
            miss_distance_improvement_m=0.0,
            phase3_stale_warning=(phase3_result.object_1_stale or phase3_result.object_2_stale),
            runtime_seconds=round(_time_mod.monotonic() - t_start, 6),
        )

    rel_speed = phase3_result.relative_speed_km_s
    tca_min = phase3_result.tca_minutes_from_start

    (refined_t, refined_sep, final_dt_s, spatial_res_m,
     iters, converged, reason, iter_log) = refine_tca(
        satrec_a, satrec_b, jd0, jdf0, tca_min, rel_speed, grid_config
    )

    # Compute refined TCA UTC
    refined_tca_dt = datetime.fromtimestamp(
        screening_dt.timestamp() + refined_t * 60.0, tz=timezone.utc
    )
    refined_tca_utc = refined_tca_dt.strftime("%Y-%m-%dT%H:%M:%S.%f")

    tca_delta_s = (refined_t - tca_min) * 60.0
    improvement_m = (phase3_result.miss_distance_km - refined_sep) * 1000.0

    t_end = _time_mod.monotonic()

    return GridAnalysisResult(
        event_id=phase3_result.event_id,
        norad_id_1=norad1, name_1=phase3_result.name_1,
        norad_id_2=norad2, name_2=phase3_result.name_2,
        screening_epoch=epoch_iso,
        phase3_tca_minutes=tca_min,
        phase3_miss_distance_km=phase3_result.miss_distance_km,
        refined_tca_minutes=round(refined_t, 10),
        refined_tca_utc=refined_tca_utc,
        refined_miss_distance_km=round(refined_sep, 9),
        refined_miss_distance_m=round(refined_sep * 1000.0, 6),
        final_dt_seconds=round(final_dt_s, 12),
        spatial_resolution_m=round(spatial_res_m, 6),
        iterations_used=iters,
        converged=converged,
        convergence_reason=reason,
        relative_speed_km_s=round(rel_speed, 6),
        tca_delta_seconds=round(tca_delta_s, 6),
        miss_distance_improvement_m=round(improvement_m, 6),
        phase3_stale_warning=(phase3_result.object_1_stale or phase3_result.object_2_stale),
        iteration_log=iter_log if grid_config.include_iteration_log else [],
        runtime_seconds=round(t_end - t_start, 6),
    )


def analyse_all_alerts(
    screening_run,
    satellite_records: list,
    screening_dt: datetime,
    grid_config: GridConfig | None = None,
    alerts_only: bool = True,
) -> list:
    """
    Apply grid analysis to all (or ALERT-only) results from a Phase-3 run.
    Returns a list of GridAnalysisResult.
    """
    results = []
    for alert in screening_run.alerts:
        if alerts_only and alert.screening_status != "ALERT":
            continue
        r = analyse_encounter(alert, satellite_records, screening_dt, grid_config)
        results.append(r)
    return results
