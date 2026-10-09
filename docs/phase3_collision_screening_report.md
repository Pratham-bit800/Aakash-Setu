# Phase 3: Collision Screening Engine - Implementation Report

**Project:** Akash Setu  
**Phase:** 3 - Collision Screening Engine  
**Date:** 2026-10-10  
**Git commit:** 135cf92 on branch branch1  
**Checkpoint tag before changes:** phase2-complete  
**Status:** COMPLETE - all 36/36 tests pass, 9/9 live API endpoints verified

---

## 1. Summary

Phase 3 adds a modular, configurable collision-screening engine to the Akash Setu project.
The engine screens candidate satellite pairs over a configurable future time horizon using
SGP4-propagated TEME states, identifies close-approach events, and returns structured result
records with stable event IDs. Two new Flask API endpoints expose the engine. All 36 unit,
integration, and regression tests pass. All existing endpoints and the Three.js dashboard
continue to function without change.

IMPORTANT DISCLAIMER (present in every API response):
  "Screening alerts are propagated estimates only. This is NOT a confirmed collision or a
  calibrated collision probability. SGP4 accuracy degrades with TLE age. No manoeuvre
  modelling is applied."

---

## 2. Files Created and Modified

| File | Type | Purpose |
|---|---|---|
| backend/collision_screening.py | NEW | Self-contained screening engine |
| backend/app.py | MODIFIED | Added /api/screen and /api/screen/results; NaN validation |
| backend/tests/__init__.py | NEW | Python package marker for pytest discovery |
| backend/tests/test_collision_screening.py | NEW | 36 tests covering 14 acceptance criteria |
| pytest.ini | NEW | Registers slow custom mark; sets rootdir |
| docs/phase3_collision_screening_report.md | NEW | This report |

Files NOT modified: all datasets (data/), frontend/, backend/scripts/, notebooks/, README.md

---

## 3. New API Endpoints

### GET /api/screen

Run collision screening against a selected subset of loaded satellites.

Query parameters:
  horizon_minutes       (float, default 180)  - Screening time window
  coarse_step_minutes   (float, default 1.0)  - Propagation step
  threshold_km          (float, default 5.0)  - Alert threshold distance
  broad_phase_margin_km (float, default 50.0) - Altitude-band filter margin
  refinement_steps      (int,   default 20)   - Bisection iterations
  stale_epoch_days      (float, default 14.0) - TLE age warning threshold
  threshold_comparison  (str,   default "lt") - "lt" or "lte"
  norad_ids             (str,   optional)     - Comma-separated NORAD IDs
  limit                 (int,   default 222)  - Max satellites

Error responses:
  400 - invalid or non-finite config parameter (e.g. horizon_minutes=NaN)

### GET /api/screen/results

Return the most recent screening run result (idempotent, no new run triggered).
Returns 404 if no run has been performed since server start.

---

## 4. Algorithm

Step 1 - Broad-phase filter (O(N) per satellite)
  - Compute perigee/apogee from mean_motion + eccentricity
  - Eliminate pairs whose altitude bands cannot overlap within (threshold + margin) km

Step 2 - Coarse time-step scan (O(P x T))
  - Propagate both objects in TEME at each coarse time step
  - Record pairs where minimum separation < 5 * threshold_km

Step 3 - Bisection refinement (O(W x log2(refinement_steps)))
  - For each flagged window, bisect the [t_min-1, t_min+1] interval
  - Tracks the endpoint with smaller separation each iteration

Step 4 - Result construction
  - Compute relative velocity and radial separation
  - Assign stable deterministic event IDs (SHA1 hash of sorted pair + epoch + TCA bucket)
  - Apply threshold comparison rule

---

## 5. Coordinate Frames and Units

  Propagation frame:      TEME (True Equator Mean Equinox)
  Distance calculations:  TEME (both objects in same inertial frame)
  Position reporting:     ECEF via approximate GMST (IAU 1982) - same as existing API
  Distance unit:          km
  Velocity unit:          km/s
  Earth radius:           6371.0 km
  Gravitational param:    398600.4418 km3/s2
  SGP4 model:             WGS72, via sgp4.api.Satrec

along_track_separation_km is null in all results (not implemented - requires full RSW decomposition).
No calibrated collision probability (Pc) is computed.

---

## 6. Test Results (all 36 tests, executed 2026-10-10)

T1  TestT1HeadOn::test_finds_encounter
    Fixture: same-plane ISS-like, ma_b=0.1 deg, threshold=15 km, horizon=170 min
    Expected: >=1 alert, miss < 14 km, TCA in window, rel_speed > 0
    Actual: miss=11.85 km, TCA=161.7 min, rel_speed=0.21 km/s
    Tolerance: +-2 km  PASS

T2  TestT2Parallel::test_wide_raan_no_alert
    Fixture: ISS-like, RAAN offset 90 deg, threshold=1 km, horizon=30 min
    Expected: 0 ALERT events
    Actual: 0 ALERT events  PASS

T3  TestT3Crossing::test_converging_approach_found
    Fixture: same-plane, ma_b=0.2 deg, threshold=30 km, horizon=170 min
    Expected: >=1 alert, miss < 27 km
    Actual: miss=23.69 km, ALERT  PASS

T4  TestT4Safe::test_leo_gso_no_alert
    Fixture: LEO (15.5 rev/day) + GSO (1.0027 rev/day), threshold=5 km
    Expected: 0 ALERT events (broad-phase eliminates pair)
    Actual: 0 ALERT events  PASS

T5  TestT5Threshold (2 tests)
    lt rule:  alert_triggered(4.99)=T, (5.00)=F, (5.01)=F  PASS
    lte rule: alert_triggered(4.99)=T, (5.00)=T, (5.01)=F  PASS

T6  TestT6Refinement::test_refined_le_coarse
    Fixture: same as T1, coarse 1.0 min vs fine 0.1 min
    Expected: coarse+refinement miss <= fine-step miss + 10 km
    Actual: 11.85 km <= 11.85 km + 10 km  PASS

T7  TestT7Invalid (2 tests)
    test_bad_satrec_skipped: NORAD 30 (satrec=None) in skipped, 1 screened  PASS
    test_no_crash_on_malformed: no exception, NORAD 99 in skipped  PASS

T8  TestT8Stale::test_stale_flag_in_result
    Fixture: epoch_age_days=30, stale=True
    Expected: object_1_stale=True, notes contain "stale"
    Actual: flag set, note present  PASS

T9  TestT9Duplicates (2 tests)
    test_dedup: NORAD 50 x2 + NORAD 51 -> objects_screened=2, 50 in skipped  PASS
    test_pairs_unique: 5 objects -> pairs_after_broad_phase <= 10  PASS

T10 TestT10Units (5 tests)
    ISS perigee/apogee in [200,700] km: PASS
    GSO perigee in [30000,40000] km: PASS
    Band overlap logic correct: PASS
    rel_speed in [0,16] km/s: PASS
    JD helpers: error < 1e-9: PASS

T11 TestT11API (9 tests)
    health, satellites, propagate, propagate_one: all PASS (existing schemas unchanged)
    screen schema: all required keys present  PASS
    screen NaN param: HTTP 400, error key present  PASS
    results 404 before run: PASS
    results 200 after run: PASS
    alert fields: all required fields present  PASS

T12 TestT12Performance (2 tests - includes @pytest.mark.slow)
    10 LEO + 5 GSO: pairs_after_broad_phase (45) < total_pairs (105): PASS
    200-sat, 30-min, 2-min step: runtime=0.35s < 120s, screened=200: PASS

T13 TestT13Reproducibility::test_same_results_twice
    Same event IDs, miss distances, TCA on both runs
    Distance error < 1e-9 km  PASS

T14 TestT14Regression (7 tests)
    health, satellites, propagate_all, propagate_one, orbit_batch, 404 unknown, 400 bad ts
    All existing endpoints return original schemas  PASS x7

TOTAL: 36 passed, 0 failed, 1 warning (mark registration - fixed by pytest.ini)
Runtime: 0.71 seconds

---

## 7. Live API Regression (server with 16,684 satellites)

  [PASS] HTTP 200  GET /api/health                        sats=16684
  [PASS] HTTP 200  GET /api/satellites?limit=5            total=16684
  [PASS] HTTP 200  GET /api/propagate?ts=...&limit=3      count=16116
  [PASS] HTTP 200  GET /api/propagate_one?norad_id=25544  position + orbit_path
  [PASS] HTTP 200  GET /api/orbit_batch?norad_ids=25544   91 orbit points
  [PASS] HTTP 404  GET /api/screen/results (no run)       error message correct
  [PASS] HTTP 200  GET /api/screen?...&limit=30           alerts=38, rt=0.011s
  [PASS] HTTP 400  GET /api/screen?horizon_minutes=NaN    error: not a finite number
  [PASS] HTTP 200  GET /api/screen/results (after run)    alerts=38

  9/9 PASSED

---

## 8. Performance Measurements

22 stations, 180-min horizon, 1-min coarse step, 5 km threshold:
  objects_screened:      22
  total_pairs_possible:  231
  pairs_after_broad_phase: 189 (18% eliminated by broad-phase)
  pairs_evaluated:       38
  alerts:                38
  runtime:               0.021 s

30 mixed satellites, 30-min horizon:
  total_pairs: 435, bp_pairs: 202 (54% pass), runtime: 0.011 s

200 synthetic LEO sats, 30-min horizon, 2-min step:
  total_pairs: 19900, runtime: ~0.35 s

Scalability note: For the full 16,684-satellite dataset, use norad_ids parameter
to select a subset. Screening all pairs without filtering is not recommended in
the dev server.

---

## 9. Errors Encountered and Fixes Made

1. /api/screen returned 404 on live server
   Cause: Flask catch-all route matching /api/* (Phase 2 bug, already fixed)
   Fix: Catch-all replaced with explicit /css/, /js/, /assets/ routes

2. float("NaN") passed validation, crashed at int(NaN) inside screen_satellites
   Cause: float("NaN") is valid Python; try/except on float() does not catch NaN
   Fix: Added math.isfinite() check for all float config params -> HTTP 400

3. T1/T3 fixtures returned 0 alerts
   Cause: ma=170 deg offset -> min separation 13,535 km; never close approach
   Fix: Replaced with verified fixtures ma_b=0.1 deg (11.85 km) and 0.2 deg (23.69 km)

4. PytestUnknownMarkWarning for @pytest.mark.slow
   Cause: Custom mark not registered
   Fix: Added pytest.ini with markers = slow: marks tests as slow

---

## 10. Remaining Limitations

1. along_track_separation_km always null - needs full RSW frame decomposition
2. No calibrated collision probability (Pc)
3. Bisection assumes single closest-approach point per coarse window
4. screen_satellites() is synchronous - blocks Flask worker for large N
5. _last_screening_run stores only the most recent run (no keyed store)
6. No frontend dashboard panel for screening results (Phase 4 scope)
7. TLE freshness: all results degrade with TLE age

---

## 11. Git Checkpoint Details

  Checkpoint tag (before Phase 3): phase2-complete
  Phase 3 commit hash:             135cf92
  Branch:                          branch1
  Files changed:                   7 (6 new, 1 modified)
  Insertions:                      +1156 lines
  Deletions:                       -9 lines

---

## 12. Phase 3 Completion Statement

Phase 3 is COMPLETE.

All acceptance criteria have been met and verified by actual execution:
  [x] Collision-screening module implemented and modular
  [x] Broad-phase filter reduces pair evaluations
  [x] Coarse scan + bisection refinement for TCA estimation
  [x] All results carry explicit screening disclaimers
  [x] Stable, deterministic event IDs
  [x] Stale TLE detection and flagging
  [x] Duplicate NORAD ID deduplication
  [x] Invalid/missing satrec handled gracefully
  [x] Two new Flask endpoints: /api/screen, /api/screen/results
  [x] All 14 test cases implemented (36 individual tests)
  [x] 36/36 tests PASSED (executed 2026-10-10)
  [x] 9/9 live API endpoints VERIFIED
  [x] Existing endpoints, datasets, and Three.js dashboard unmodified
  [x] Git commit made: 135cf92

Do not proceed to Phase 4 without explicit user instruction.
