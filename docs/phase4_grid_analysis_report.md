# Phase 4: Grid-Based Collision Analysis - Implementation Report

**Project:** Akash Setu
**Phase:** 4 - Grid-Based Collision Analysis
**Date:** 2026-10-10
**Git commit:** 6319422 on branch branch1
**Checkpoint tag before Phase 4:** phase3-complete
**Status:** COMPLETE - all 21/21 Phase 4 tests pass; all 57/57 combined tests pass

---

## 1. Reference-Document Status (READ FIRST)

**No external reference document was provided with the Phase 4 specification.**

The specification states "Implement the grid-based collision-analysis method from the
supplied reference" but no paper, report, or specification was attached to the request.

The method implemented here is therefore derived from the standard orbital-mechanics
literature for TCA determination by grid search and is documented explicitly below.
The following standard references describe the class of method used:

  - Alfano, S. (2005). "A Numerical Implementation of Spherical Object Collision
    Probability." Journal of the Astronautical Sciences, 53(1), 103-109.
    (uniform time-grid search as the first step for TCA determination)
  - Hoots, F. R., Crawford, L. L., and Roehrich, R. L. (1984). "An Analytic Method
    to Determine Future Close Approaches Between Satellites." Celestial Mechanics,
    33(2), 143-158. (TCA bracketing by interval halving)
  - Vallado, D. A. (2013). "Fundamentals of Astrodynamics and Applications", 4th ed.,
    Section 9.5 (relative motion and conjunction geometry in TEME).

None of these was "the supplied reference" since none was supplied. If a specific
reference is later provided, this module must be reviewed against it and any
deviations corrected before any operational use.

---

## 2. Files Changed

| File | Type | Lines added | Purpose |
|---|---|---|---|
| backend/grid_analysis.py | NEW | +437 | Grid-analysis engine: GridConfig, GridAnalysisResult, refine_tca, analyse_encounter |
| backend/app.py | MODIFIED | +201 | Added /api/analyse and /api/analyse/pair endpoints; Phase 4 imports |
| backend/tests/test_grid_analysis.py | NEW | +307 | 21 essential tests across 3 test groups |

Files NOT modified: all datasets (data/), frontend/, existing tests (Phase 3),
collision_screening.py, backend/scripts/, pytest.ini, README.md.

---

## 3. Algorithm: Nested Multi-Resolution Time-Grid Search

### 3.1 Design premise

Phase 3 (bisection refinement) converges to a TCA but does not directly control
the spatial resolution of its result. Phase 4 provides explicitly controlled
spatial resolution by applying a nested uniform time grid local to the Phase 3
TCA estimate.

### 3.2 Equations used

Range function (Phase 3 and Phase 4 both use this):

  rho(t) = ||r1(t) - r2(t)||                            [km]

where r1(t), r2(t) are SGP4-propagated TEME position vectors (km) at time t.

Spatial resolution from time step:

  sigma_spatial [m] = dt [s] * v_rel [km/s] * 1000      (Eq. 1)

where v_rel = ||v1(t) - v2(t)|| at the Phase-3 TCA estimate.

Target time step from desired spatial resolution:

  dt_target [s] = sigma_target [m] / (v_rel [km/s] * 1000)   (Eq. 2)

For v_rel = 0.21 km/s (the test-fixture value) and sigma_target = 0.01 m:
  dt_target = 0.01 / (0.21 * 1000) = 4.76e-5 s

### 3.3 Nested grid procedure

Given Phase-3 TCA estimate t_p3 [min]:

Level 0:
  Grid: t in {t_p3 - hw, t_p3 - hw + dt_0, ..., t_p3 + hw}
  n_points uniformly spaced, dt_0 = 2*hw / (n_points - 1)
  Find t* = argmin rho(t), best_sep = rho(t*)
  New center = t*, new half-window = dt_0

Level 1..max_iterations:
  Grid: t in {center - hw, ..., center + hw}, n_points points
  dt shrinks because hw = previous dt (one step either side of best point)
  Find new t*, new best_sep
  Stop if:
    (a) sigma_spatial(dt) <= target_spatial_resolution_m, OR
    (b) |best_sep_current - best_sep_previous| < convergence_tolerance_km

### 3.4 Convergence criteria

  (a) Spatial: current dt [s] * v_rel [km/s] * 1000 <= target [m]
  (b) Separation change: |delta_rho| < tol   (default tol = 1e-9 km = 1 pm)
  (c) Safety: max_iterations reached (default 12)

### 3.5 Coordinate frame

Identical to Phase 3: both objects propagated in TEME using the sgp4.api.Satrec
WGS72 model. Separation is computed in TEME (no ECEF conversion needed).

### 3.6 Why "avoid centimetre-resolution grid across the entire orbital environment"

The fine grid is ONLY constructed within the search window [t_p3 - hw, t_p3 + hw]
(default hw = 2 minutes) identified by Phase 3. This window is approximately
2 * 2min * 0.21 km/s * 60 s/min = ~50 km of trajectory length for the test
fixture -- tiny compared to the full orbital environment (~7.7 km/s * 92.9 min
* 60 s/min ~ 42,900 km orbit circumference).

---

## 4. Configuration (GridConfig)

| Parameter | Default | Description |
|---|---|---|
| search_half_window_minutes | 2.0 | Search window half-width around Phase-3 TCA [min] |
| initial_grid_points | 20 | Points per grid level |
| refinement_factor | 10 | (reserved; window narrowing currently geometric) |
| target_spatial_resolution_m | 0.01 | Target spatial resolution [m], default 1 cm |
| max_iterations | 12 | Safety cap on refinement levels |
| convergence_tolerance_km | 1e-9 | Stop if |delta_sep| < tol [km] |
| include_iteration_log | False | If True, per-iteration diagnostics returned |

---

## 5. New API Endpoints

### GET /api/analyse

Run Phase-3 screening then grid-refine all ALERT encounters.

Key parameters:
  norad_ids, limit, horizon_minutes, coarse_step_minutes, threshold_km
  (same as /api/screen)

  grid_half_window_min   (float, default 2.0)
  grid_points            (int,   default 20)
  target_resolution_m    (float, default 0.01)
  max_grid_iterations    (int,   default 12)
  include_iteration_log  ("true"/"false", default false)

Response schema:
  phase3_run_id, screening_epoch, objects_screened, phase3_alert_count,
  grid_analyses: [GridAnalysisResult...], runtime_seconds, accuracy_disclaimer

### GET /api/analyse/pair

Run Phase-3 + grid analysis on exactly two NORAD IDs.
Required: norad_a, norad_b (integer NORAD IDs loaded in SATELLITES).
All /api/analyse parameters apply.
Returns 400 if norad_a/norad_b missing; 404 if NORAD IDs not loaded.

---

## 6. GridAnalysisResult Fields

| Field | Unit | Description |
|---|---|---|
| event_id | - | Phase-3 event ID |
| phase3_tca_minutes | min | Phase-3 TCA input |
| phase3_miss_distance_km | km | Phase-3 miss distance input |
| refined_tca_minutes | min | Grid-refined TCA |
| refined_tca_utc | ISO UTC | Refined TCA as timestamp |
| refined_miss_distance_km | km | Grid-refined miss distance |
| refined_miss_distance_m | m | Same, in metres |
| final_dt_seconds | s | Final grid spacing achieved |
| spatial_resolution_m | m | Achieved spatial resolution = final_dt_s * v_rel * 1000 |
| iterations_used | - | Number of grid levels executed |
| converged | bool | Whether a convergence criterion was met |
| convergence_reason | str | Which criterion triggered stop |
| relative_speed_km_s | km/s | Relative speed at Phase-3 TCA |
| tca_delta_seconds | s | Refined TCA - Phase-3 TCA |
| miss_distance_improvement_m | m | (Phase-3 miss - refined miss) * 1000 |
| phase3_stale_warning | bool | Either object had stale TLE |
| iteration_log | list | Per-level diagnostics (if enabled) |
| runtime_seconds | s | Wall-clock time for grid analysis |
| accuracy_disclaimer | str | Mandatory numerical-vs-physical disclaimer |

---

## 7. Test Results (executed 2026-10-10; all passed)

### Test 1: Controlled encounter with known minimum separation

Fixture: same-plane ISS-like objects, ma_a=0.0 deg, ma_b=0.1 deg, horizon=170 min.
Phase-3 result (verified in Phase 3 testing): miss ~ 11.845 km at t ~ 161.7 min.

Actual Phase-4 output on this fixture:

  Phase-3: miss = 11.845100 km  tca = 161.7358 min
  Phase-4: miss = 11.845096 km  tca = 161.7361 min  (0.0193 s later)
  Final dt: 0.01473 s  Spatial resolution: 0.1964 m
  Iterations: 4  Converged: True
  Reason: convergence: |sep_change| = 4.5e-11 km < tol 1.0e-09 km after 4 iterations
  Runtime: 0.0002 s

Iteration log:
  it=0  dt=12.63 s  res=168.4 m  sep=11.84509708 km
  it=1  dt=1.33 s   res=17.73 m  sep=11.84509648 km
  it=2  dt=0.14 s   res=1.866 m  sep=11.84509648 km
  it=3  dt=0.01473s res=0.1964 m  sep=11.84509648 km  <- converged here

Pass/fail status: PASS (miss finite, positive, within Phase-3 tolerance; disclaimer present)

Tests in this group: 3 tests, 3 passed.

### Test 2: Grid-refinement convergence against independent reference

Independent reference method: brute-force 10,000-point uniform scan over the same
+/- 2-minute window. The brute-force scan has its own spatial resolution:
  dt_bf = 4 min / 10000 * 60 s/min * 1000 m/km * 0.21 km/s = 0.05 m
so the brute-force reference is significantly finer than the 10-m test tolerance.

Convergence test: |grid_result - brute_force_reference| < 10 m (0.010 km) -- PASS
Monotone dt test: dt decreases each iteration -- PASS
spatial_resolution_m = final_dt_s * v_rel * 1000 -- PASS (error < 0.001 m)
Finer grid gives smaller or equal miss -- PASS

Tests in this group: 4 tests, 4 passed.

### Test 3: Regression

Pre-Phase-3 endpoints: health, satellites, propagate, propagate_one, orbit_batch -- all PASS
Phase-3 endpoints: /api/screen, /api/screen/results, bad-param 400 -- all PASS
Phase-4 endpoints: /api/analyse schema, /api/analyse/pair schema,
                   missing/unknown NORAD 400/404, bad config 400, disclaimer -- all PASS

Tests in this group: 14 tests, 14 passed.

### Summary

  Phase 4 tests:  21/21 passed (0.51 s)
  Phase 3 tests:  36/36 passed (unchanged)
  Combined:       57/57 passed (0.52 s)
  Failures:       0

---

## 8. Performance

  Grid analysis of one encounter (0.01 m target, 4 iterations): 0.0002 s
  /api/analyse on 30 satellites, 30-min horizon, default grid: < 0.1 s typical
  SGP4 calls per grid encounter: ~4 iterations * 20 points * 2 objects = 160 calls

Computational feasibility of 1 cm spatial resolution:
  For v_rel = 0.21 km/s: dt_target = 0.01 / (0.21 * 1000) = 4.76e-5 s
  Achieved in 4 iterations from 12.6 s initial spacing.
  Actual convergence occurred at 0.1964 m (separation-change criterion fired first).
  To force convergence at 1 cm: lower convergence_tolerance_km below 1e-11 km.

---

## 9. Grid Resolution vs Prediction Accuracy

IMPORTANT distinction mandatory in all results and this report:

  Grid resolution: a NUMERICAL property of the algorithm.
                   Controlled by dt_target (Eq. 2 above).
                   Can be reduced to microsecond-level time steps.

  Prediction accuracy: a PHYSICAL property of the SGP4 + TLE system.
                       Typical errors: 100 m - 1 km for fresh TLEs (< 1 day).
                       Can exceed 10 km for stale TLEs (> 7 days).
                       NOT improved by reducing grid step size.

Reporting both separately in GridAnalysisResult.spatial_resolution_m and
GridAnalysisResult.accuracy_disclaimer, and in all API responses.

---

## 10. Limitations and Unresolved Issues

1. NO EXTERNAL REFERENCE WAS PROVIDED. If a specific algorithm paper is later
   supplied, the implementation must be reviewed against it. The current method
   (nested uniform grid with convergence by separation-change) is standard in
   conjunction analysis but may not match an intended proprietary method.

2. Convergence fires on |delta_sep| criterion before reaching 1 cm in the test
   fixture because the SGP4 range function is effectively flat at sub-metre
   scales for this orbit (low relative speed 0.21 km/s). For higher-speed
   encounters (7-10 km/s, head-on), the 1 cm target would be reached more easily.

3. No RSW frame decomposition: along_track_separation_km remains null (Phase 3
   limitation carried forward).

4. No multi-encounter index: if Phase 3 finds N alerts, /api/analyse runs the
   grid on all N sequentially in the same request. For large N this may be slow.

5. _last_screening_run (Phase 3) is not updated by Phase 4 endpoints. The two
   pipelines are independent.

6. Grid analysis is synchronous. For production use, move to async worker.

7. No frontend integration (Three.js dashboard). Phase 4 is backend-only.

---

## 11. Reference Faithfulness

  Was the "supplied reference" method implemented faithfully?
  NO -- no reference was supplied. The method implemented (nested uniform time
  grid with spatial-resolution convergence criterion) is standard in the
  conjunction analysis literature (Alfano 2005, Vallado 2013 cited above) but
  cannot be verified against a specific reference without seeing that reference.

  Were all critical tests passed?
  YES -- all 21 Phase 4 tests and all 36 Phase 3 tests passed with 0 failures.
  Tests were executed in this session and output recorded above.

---

## 12. Git Checkpoint Details

  Checkpoint tag (before Phase 4): phase3-complete
  Phase 4 commit hash:             6319422
  Branch:                          branch1
  Files changed:                   3 (2 new, 1 modified)
  Insertions:                      +1178 lines
