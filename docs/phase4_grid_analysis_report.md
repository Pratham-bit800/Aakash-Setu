# Phase 4: Grid-Based Collision Analysis - Validation Report

**Project:** Akash Setu
**Phase:** 4 (validation pass)
**Date:** 2026-10-10
**Git commits:** 6319422 (Phase 4 initial), 5ad5996 (validation fixes)
**Branch:** branch1
**Status:** COMPLETE - 64/64 tests pass, all critical issues resolved

---

## 1. Reference-Document Status

**No external reference document was provided with the Phase 4 specification.**

The specification states "from the supplied reference" but no paper, report, or
specification was attached. This report was generated after explicitly searching
the project workspace for any attached document; none was found.

The method is derived from standard conjunction-analysis literature:
  - Alfano (2005): uniform time-grid scan for TCA determination
  - Hoots, Crawford, Roehrich (1984): TCA bracketing by interval narrowing
  - Vallado (2013): relative motion geometry in TEME, Section 9.5

None of these is "the supplied reference." If a specific reference is later
provided, backend/grid_analysis.py must be reviewed against it. This report
does NOT claim reference fidelity because no reference exists to be faithful to.

---

## 2. Issues Found and Fixed

### Issue 1: converged=True conflated three distinct stopping conditions

Before fix: `converged = True` was set for three different stops:
  (a) sigma_spatial(dt) <= target  [desired]
  (b) |sep_change| < tol           [early stop, target NOT met]
  (c) hw <= target_dt              [window narrowed past target]

This made `converged` meaningless as a signal. A caller checking `converged=True`
could not distinguish "resolution achieved" from "stopped because the range
function is flat" -- which is an entirely different situation with different
implications for result quality.

After fix:
  `converged` is True ONLY when `stop_reason == STOP_RESOLUTION`.
  Four machine-readable stop-reason constants now distinguish all cases:
    STOP_RESOLUTION  = "RESOLUTION_ACHIEVED"   -- converged=True
    STOP_TOLERANCE   = "TOLERANCE_CONVERGENCE" -- converged=False
    STOP_WINDOW      = "WINDOW_DEGENERATE"     -- converged=False
    STOP_MAX_ITER    = "MAX_ITERATIONS"        -- converged=False

  `GridAnalysisResult.stop_reason` carries the constant.
  `GridAnalysisResult.convergence_reason` carries the human-readable elaboration.

### Issue 2: refinement_factor declared but never used

Before fix: GridConfig.refinement_factor = 10 was declared in the dataclass
but the window-narrowing loop used:
    hw = max(current_dt_min, target_dt_min * 0.5)
which completely ignores refinement_factor.

After fix: window narrowing now implements Eq 3 and Eq 4:
    dt_{k+1} [min] = dt_k [min] / refinement_factor      (Eq 3)
    hw_{k+1} [min] = dt_{k+1} * (n - 1) / 2             (Eq 4)

Verified numerically: with rf=5, n=10, hw=2 min, dt_0 = 26.67 s:
  L0: 26.67 s (expected 26.67 s) -- ok
  L1:  5.33 s (expected  5.33 s) -- ok
  L2:  1.07 s (expected  1.07 s) -- ok
  L3:  0.213 s (expected 0.213 s) -- ok
Relative error in all cases < 1e-9.

### Issue 3: float('inf') as target triggered STOP_RESOLUTION immediately

Before fix: setting target_spatial_resolution_m=float('inf') caused the check
    if sigma_spatial <= target:  [float <= inf is always True]
to fire at level 0, making it impossible to disable the resolution stop.

After fix: both stopping criteria are guarded with math.isfinite():
    if math.isfinite(target) and sigma_spatial <= target:  -> STOP_RESOLUTION
    if math.isfinite(tol) and sep_change < tol:           -> STOP_TOLERANCE
Setting either to float('inf') now reliably disables that criterion.

### Issue 4: iteration_log key name changed

Before fix: test expected entry["dt_min"]; log used entry["dt_seconds"].
After fix: test corrected to use entry["dt_seconds"] (the canonical key).

---

## 3. Algorithm (as implemented, updated)

Given: two Satrec objects, Phase-3 TCA t_p3, search_half_window hw, n points, rf factor.

Level 0:
  Grid on [t_p3 - hw, t_p3 + hw], n equally spaced points
  dt_0 = 2*hw / (n-1)
  Find t* = argmin rho(t) where rho(t) = ||r1(t) - r2(t)||  [km, TEME frame]

Level k >= 1:
  dt_k = dt_{k-1} / refinement_factor           (Eq 3 -- refinement_factor IS used)
  hw_k = dt_k * (n - 1) / 2                     (Eq 4)
  center <- previous t*
  Find new t*, new min_sep

Stop conditions (checked in priority order):
  (a) math.isfinite(target) and sigma_spatial(dt_k) <= target -> STOP_RESOLUTION
  (b) k>=1 and math.isfinite(tol) and |sep_k-sep_{k-1}| < tol -> STOP_TOLERANCE
  (c) hw_next <= 0 or non-finite                              -> STOP_WINDOW
  (d) k == max_iterations-1                                   -> STOP_MAX_ITER

Spatial resolution formula:
  sigma_spatial [m] = dt [s] * v_rel [km/s] * 1000    (Eq 1)

---

## 4. New Essential Tests Added

### TestCrossingEncounter (3 tests)

Fixture: two ISS-altitude objects, inclination 51.6 and 128.4 deg (retrograde),
same RAAN and ma=0. Both start at equatorial ascending node -> co-located at t=0.
Observed: miss=0.0009 km, v_rel=9.52 km/s, tca=0.025 min.

test_crossing_encounter_detected:
  Phase-3 detects the crossing at t<5 min with v_rel > 5 km/s. PASS.

test_grid_analysis_on_crossing_encounter:
  Grid refines crossing TCA to miss=0.000934 km with STOP_RESOLUTION at 0.30 m
  resolution in 6 iterations. Spatial resolution formula verified. PASS.

test_high_relative_speed_spatial_resolution:
  With the same dt, sigma_cross (9.52 km/s) >> sigma_coplanar (0.21 km/s) by 45x.
  Verifies Eq 1 is velocity-dependent. PASS.

### TestStopReasonDistinction (4 tests)

test_stop_tolerance_fires_before_resolution:
  Default config (tol=1e-9, target=0.01m): STOP_TOLERANCE fires at iteration 4,
  spatial_res=0.168 m > 0.01 m target. converged=False. PASS.

test_stop_resolution_achieved_when_tolerance_disabled:
  tol=inf, target=0.01m: STOP_RESOLUTION fires at iteration 6, spatial_res=0.00168 m.
  converged=True. PASS.

test_stop_max_iter_when_both_criteria_disabled:
  target=inf, tol=inf, max_iterations=3: STOP_MAX_ITER, iterations_used=3,
  converged=False. PASS.

test_refinement_factor_controls_dt_reduction:
  rf=5, n=10: dt values at levels 0-3 match dt_0/rf^k exactly (rel_err < 1e-9).
  PASS.

---

## 5. Full Test Results (executed 2026-10-10)

Phase 3 tests (test_collision_screening.py):  36/36 passed
Phase 4 tests (test_grid_analysis.py):        28/28 passed
Combined total:                                64/64 passed
Runtime:                                       0.56 s
Failures:                                      0

---

## 6. Observed Stopping Behavior on Test Fixtures

### Low-speed coplanar (v_rel=0.013 km/s, miss=11.845 km)

Note: Phase-3 v_rel=0.013 km/s (relative speed at TCA, not orbital speed).

  Default config (tol=1e-9, target=0.01 m):
    Stop: TOLERANCE_CONVERGENCE at iteration 4, sigma=0.1684 m
    The range function is numerically flat at sub-0.17-m scales.
    The 1-cm target was NOT achieved. converged=False.

  Tolerance disabled (tol=inf, target=0.01 m):
    Stop: RESOLUTION_ACHIEVED at iteration 6, sigma=0.00168 m
    L0: 12.63s  168.4m       L1: 1.263s   16.84m
    L2: 0.1263s  1.684m      L3: 0.01263s  0.1684m
    L4: 0.001263s 0.01684m   L5: 0.0001263s 0.001684m  <- RESOLUTION MET
    converged=True.

### Crossing (v_rel=9.52 km/s, miss=0.0009 km at t=0.025 min)

  Grid (target=1.0 m, half_window=0.5 min, rf=10):
    Stop: RESOLUTION_ACHIEVED at iteration 6, sigma=0.3005 m
    L0: 3.158s  30050m  L1: 0.316s  3005m  L2: 0.0316s  300.5m
    L3: 3.16e-3s  30.05m  L4: 3.16e-4s  3.005m  L5: 3.16e-5s  0.3005m  <- MET

---

## 7. Remaining Scientific Limitations

1. NO REFERENCE DOCUMENT PROVIDED. If a specific algorithm paper is later
   supplied, the implementation must be compared and any deviations fixed.

2. STOP_TOLERANCE frequently fires before STOP_RESOLUTION for low-relative-speed
   encounters. At v_rel < ~0.1 km/s, the range function rho(t) is so flat near
   the minimum that successive grid levels find essentially the same minimum.
   This is an inherent SGP4/double-float arithmetic limit, not a bug.
   To reach 1-cm spatial resolution in such cases: disable convergence_tolerance_km.

3. along_track_separation_km is still null (requires full RSW frame decomposition,
   not implemented).

4. No collision probability (Pc). Phase 4 does not add covariance modelling.

5. Synchronous execution: analyse_all_alerts blocks the Flask worker thread.

6. No frontend dashboard panel for grid-analysis results.

7. Phase-3 relative speed (relative_speed_km_s) is used as the v_rel input to
   Eq 1 and Eq 2 in the grid analysis. If Phase-3 bisection gives a poor v_rel
   estimate (e.g., for very slow encounters), the target_dt computed via Eq 2 and
   the reported sigma_spatial may be inaccurate. At the final TCA, a fresh v_rel
   is not re-evaluated inside grid analysis (only the Phase-3 value is reused).

8. TLE freshness: all results degrade with TLE age. Fresh TLE < 1 day: ~100 m.
   Stale TLE > 7 days: >10 km. Grid resolution (sub-mm) is orders of magnitude
   finer than this physical error floor.

---

## 8. Reference Faithfulness Statement

  Was the supplied reference method implemented faithfully?
  CANNOT BE DETERMINED -- no reference was supplied.
  The method implemented (nested uniform grid with refinement_factor-controlled
  window narrowing and three distinct stopping criteria) is consistent with
  standard TCA grid-search literature but has not been verified against any
  specific document.

  Were all critical tests passed?
  YES -- 64/64 tests passed with 0 failures on 2026-10-10.
  Tests were executed in this session; output is recorded in Section 6 above.

---

## 9. Git Checkpoint

  Checkpoint before Phase 4: phase3-complete
  Phase 4 initial commit:    6319422
  Validation fix commit:     5ad5996
  Branch:                    branch1
  Files changed (total):     3 (grid_analysis.py, test_grid_analysis.py, this report)
