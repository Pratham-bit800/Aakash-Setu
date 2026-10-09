# Phase 4: Grid-Based Collision Analysis - Scientific Correctness & Validation Report

**Project:** Akash Setu
**Phase:** 4 (scientific review & validation pass)
**Date:** 2026-10-10
**Git branch:** branch1
**Status:** COMPLETE - 66/66 tests pass, scientific correctness validated, Phase 4 finalized

---

## 1. Reference-Document Status

**No external reference document was provided with the Phase 4 specification.**

The specification states "from the supplied reference" but no paper, report, or
specification was attached. The project workspace was searched thoroughly; no
external PDF, document, or equation specification was provided.

The method is derived from standard conjunction-analysis literature:
  - Alfano (2005): uniform time-grid scan for TCA determination
  - Hoots, Crawford, Roehrich (1984): TCA bracketing by interval narrowing
  - Vallado (2013): relative motion geometry in TEME, Section 9.5

None of these is "the supplied reference." If a specific reference is later
provided, `backend/grid_analysis.py` must be reviewed against it. This report
does NOT claim reference fidelity because no reference exists in the repository
to be faithful to.

---

## 2. Issues Found and Resolved

### Issue 1: converged=True conflated three distinct stopping conditions
Before fix: `converged = True` was set for three different stops:
  (a) sigma_spatial(dt) <= target  [desired]
  (b) |sep_change| < tol           [early stop, target NOT met]
  (c) hw <= target_dt              [window narrowed past target]

This conflation obscured algorithmic status. A caller checking `converged=True`
could not distinguish "resolution achieved" from "stopped because the range
function is flat".

After fix:
  `converged` is True ONLY when `stop_reason == STOP_RESOLUTION`.
  Four machine-readable stop-reason constants distinguish all cases:
    STOP_RESOLUTION  = "RESOLUTION_ACHIEVED"   -- converged=True
    STOP_TOLERANCE   = "TOLERANCE_CONVERGENCE" -- converged=False
    STOP_WINDOW      = "WINDOW_DEGENERATE"     -- converged=False
    STOP_MAX_ITER    = "MAX_ITERATIONS"        -- converged=False

  `GridAnalysisResult.stop_reason` carries the constant.
  `GridAnalysisResult.convergence_reason` carries human-readable elaboration.

### Issue 2: refinement_factor declared but never used
Before fix: `GridConfig.refinement_factor = 10` was declared in the dataclass
but the window-narrowing loop used `hw = max(current_dt_min, target_dt_min * 0.5)`,
ignoring `refinement_factor`.

After fix: window narrowing implements Eq 3 and Eq 4:
    dt_{k+1} [min] = dt_k [min] / refinement_factor      (Eq 3)
    hw_{k+1} [min] = dt_{k+1} * (n - 1) / 2             (Eq 4)

Verified numerically: with rf=5, n=10, hw=2 min, dt_0 = 26.67 s:
  L0: 26.67 s | L1: 5.33 s | L2: 1.07 s | L3: 0.213 s (rel_err < 1e-9).

### Issue 3: float('inf') as target triggered STOP_RESOLUTION immediately
Before fix: setting `target_spatial_resolution_m=float('inf')` caused
`sigma_spatial <= target` to fire at level 0 (float <= inf is True), preventing
isolation of other stopping conditions.

After fix: both stopping criteria are guarded with `math.isfinite()`:
    if math.isfinite(target) and sigma_spatial <= target:  -> STOP_RESOLUTION
    if math.isfinite(tol) and sep_change < tol:           -> STOP_TOLERANCE
Setting either to `float('inf')` reliably disables that criterion.

### Issue 4: iteration_log key name consistency
Fixed key naming in `iteration_log` to `dt_seconds` (canonical key) and added
`spatial_resolution_m`, `best_sep_km`, and `best_rv_km_s`.

### Issue 5: Relative velocity reused from Phase 3 instead of recalculated at refined TCA
Before fix: `relative_speed_km_s` from Phase 3 was reused throughout refinement
and stored directly in `GridAnalysisResult.relative_speed_km_s`. The final
`spatial_resolution_m` was computed using Phase-3 relative speed rather than the
kinematic relative velocity at the refined TCA.
After fix:
  - At each grid iteration, `best_rv` tracks the velocity norm at `best_t`.
  - Upon loop completion, $v_{	ext{rel}}$ is explicitly recalculated via fresh
    SGP4 TEME propagation at `refined_tca` (`best_t`):
    $$v_{	ext{rel}} = \|\mathbf{v}_1(t^*) - \mathbf{v}_2(t^*)\|$$
  - `spatial_resolution_m` is computed consistently using the refined relative
    velocity: $\sigma_{	ext{spatial}} = \Delta t_{	ext{final}} 	imes v_{	ext{rel}} 	imes 1000$.
  - `GridAnalysisResult.relative_speed_km_s` stores the recalculated refined
    relative speed.
  - `refine_tca()` returns `refined_rv` as a 10th tuple element.

---

## 3. Algorithm (as implemented and verified)

Given: two `Satrec` objects, Phase-3 TCA $t_{	ext{p3}}$, half-window $hw$, $n$ points, refinement factor $rf$.

Level 0:
  Grid on $[t_{	ext{p3}} - hw, t_{	ext{p3}} + hw]$, $n$ equally spaced points.
  $\Delta t_0 = 2 \cdot hw / (n - 1)$
  Find $t^* = \operatorname{argmin} ho(t)$ where $ho(t) = \|\mathbf{r}_1(t) - \mathbf{r}_2(t)\|$ [km, TEME frame].
  $v_{	ext{rel}} = \|\mathbf{v}_1(t^*) - \mathbf{v}_2(t^*)\|$

Level $k \ge 1$:
  $\Delta t_k = \Delta t_{k-1} / rf$ (Eq 3)
  $hw_k = \Delta t_k \cdot (n - 1) / 2$ (Eq 4)
  $	ext{center} \leftarrow 	ext{previous } t^*$
  Find new $t^*$, new minimum separation $ho(t^*)$, and $v_{	ext{rel}}(t^*)$.

Stop conditions (checked in priority order):
  (a) `math.isfinite(target)` and $\sigma_{	ext{spatial}}(\Delta t_k) \le 	ext{target} ightarrow$ `STOP_RESOLUTION` (`converged=True`)
  (b) $k \ge 1$ and `math.isfinite(tol)` and $|ho_k - ho_{k-1}| < 	ext{tol} ightarrow$ `STOP_TOLERANCE` (`converged=False`)
  (c) $hw_{k+1} \le 0$ or non-finite $ightarrow$ `STOP_WINDOW` (`converged=False`)
  (d) $k = 	ext{max\_iterations} - 1 ightarrow$ `STOP_MAX_ITER` (`converged=False`)

Post-refinement kinematics:
  Propagate both objects at $t^*$ to obtain $\mathbf{v}_1(t^*)$ and $\mathbf{v}_2(t^*)$.
  $v_{	ext{rel,refined}} = \|\mathbf{v}_1(t^*) - \mathbf{v}_2(t^*)\|$
  $\sigma_{	ext{spatial}} = \Delta t_{	ext{final}} \cdot v_{	ext{rel,refined}} \cdot 1000$ (Eq 1)

---

## 4. Test Suite Summary

The test suite contains 66 automated tests across Phase 3 and Phase 4:

### TestCrossingEncounter (3 tests)
Fixture: two ISS-altitude objects, inclination 51.6° and 128.4° (retrograde), same RAAN and ma=0.
  - `test_crossing_encounter_detected`: Phase-3 detects crossing with $v_{	ext{rel}} > 5$ km/s. PASS.
  - `test_grid_analysis_on_crossing_encounter`: Grid refines crossing TCA to miss=0.000934 km with `STOP_RESOLUTION`. PASS.
  - `test_high_relative_speed_spatial_resolution`: Verifies velocity-dependent scaling of spatial resolution ($\sigma_{	ext{cross}} \gg \sigma_{	ext{coplanar}}$). PASS.

### TestStopReasonDistinction (4 tests)
  - `test_stop_tolerance_fires_before_resolution`: Default config triggers `STOP_TOLERANCE` at iteration 4 on flat range function, `converged=False`. PASS.
  - `test_stop_resolution_achieved_when_tolerance_disabled`: `tol=inf` triggers `STOP_RESOLUTION` at iteration 6, `converged=True`. PASS.
  - `test_stop_max_iter_when_both_criteria_disabled`: `target=inf`, `tol=inf`, max 3 iters triggers `STOP_MAX_ITER`. PASS.
  - `test_refinement_factor_controls_dt_reduction`: Verifies $\Delta t_k = \Delta t_0 / rf^k$ across levels 0–3 with relative error < 1e-9. PASS.

### TestRefinedTcaScientificCorrectness (2 essential tests)
  - `test_refined_tca_velocity_recalculated_and_used_consistently`:
    Independently propagates both satellites at `result.refined_tca_minutes` and verifies:
    1. `result.relative_speed_km_s` matches independent propagation ($|\Delta| < 10^{-5}$ km/s).
    2. `result.spatial_resolution_m` is strictly equal to $\Delta t_{	ext{final}} 	imes v_{	ext{rel}} 	imes 1000$ ($|\Delta| < 10^{-4}$ m).
    3. Direct `refine_tca()` invocation returns 10-element tuple with matching refined $v_{	ext{rel}}$. PASS.
  - `test_stopping_behavior_and_numerical_resolution_distinction`:
    Verifies that when `STOP_TOLERANCE` fires:
    1. `converged` is `False`, cleanly distinguishing tolerance convergence from spatial resolution achievement.
    2. `spatial_resolution_m` is strictly greater than `target_spatial_resolution_m`.
    3. `accuracy_disclaimer` explicitly distinguishes numerical grid resolution from physical prediction accuracy. PASS.

---

## 5. Full Test Results (executed 2026-10-10)

| Suite | File | Tests | Passed | Failed |
|---|---|---|---|---|
| Phase 3 Screening | `backend/tests/test_collision_screening.py` | 36 | 36 | 0 |
| Phase 4 Grid Analysis | `backend/tests/test_grid_analysis.py` | 30 | 30 | 0 |
| **Total** | | **66** | **66** | **0** |

**Execution time:** 0.68 s
**Result:** 100% PASS

---

## 6. Numerical Resolution vs. Physical Prediction Accuracy

A critical scientific distinction enforced in this phase:

1. **Numerical Grid Resolution ($\sigma_{	ext{spatial}}$):**
   A mathematical discretization parameter defining the step size of the search
   grid along the orbit trajectory:
   $$\sigma_{	ext{spatial}} = \Delta t 	imes v_{	ext{rel}} 	imes 1000$$
   Achieving $\sigma_{	ext{spatial}} \le 0.01$ m (1 cm) means only that the
   time discretization is fine enough that the relative position changes by
   less than 1 cm per grid step.

2. **Physical Prediction Accuracy:**
   Determined by physical force models, numerical integrator errors, and above all,
   TLE observational uncertainty:
   - Fresh TLE (< 1 day): typical SGP4 ephemeris error ~100 m to 1 km.
   - Stale TLE (7–14 days): ephemeris error exceeds 5 km to 20 km.
   - SGP4 does not model solar radiation pressure variations, atmospheric density
     fluctuations, or third-body perturbations with precision beyond several hundred metres.

Therefore, achieving a 1-cm grid resolution **does not** imply 1-cm physical positional
accuracy. The `accuracy_disclaimer` field in `GridAnalysisResult` explicitly states this
limitation on every API response.

---

## 7. Remaining Scientific Limitations

1. **No External Reference Document Supplied:** If a specific reference paper
   or standard is provided for Phase 5 or orbital conjunction assessment, the
   algorithm must be evaluated against it.
2. **Tolerance Convergence on Low-Speed Encounters:** At $v_{	ext{rel}} < 0.1$ km/s,
   the quadratic range minimum $ho(t)$ is extremely flat. Double-precision arithmetic
   and SGP4 roundoff cause $|ho_k - ho_{k-1}|$ to drop below $10^{-9}$ km before
   reaching sub-centimetre time steps. In these cases, the algorithm stops with
   `STOP_TOLERANCE` and reports `converged=False`.
3. **RSW Frame Decomposition:** `along_track_separation_km` remains null until
   full Radial-Intrack-Crosstrack (RSW/NTW) frame transformations are implemented.
4. **Collision Probability ($P_c$):** Covariance matrices and collision probability
   calculations (e.g., Foster, Akella-Alfriend, or Hall methods) are not part of Phase 4.
5. **Synchronous Execution:** Batch analysis `/api/analyse` runs synchronously on the
   Flask worker thread.

---

## 8. Reference Faithfulness Statement

  Was the supplied reference method implemented faithfully?
  **CANNOT BE DETERMINED** -- no reference document was provided.
  The method implemented (nested uniform grid refinement with refinement_factor-controlled
  window narrowing, refined-TCA velocity recalculation, and machine-readable stopping criteria)
  is derived from established astrodynamics literature (Alfano 2005, Hoots 1984, Vallado 2013).

  Were all tests passed?
  **YES** -- 66/66 tests passed with 0 failures on 2026-10-10.
