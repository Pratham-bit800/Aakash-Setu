# Akash Setu — Pre-Phase 7 Final Validation Report

**Date:** 2026-10-10  
**Branch:** `branch1` (up to date with origin)  
**Test suite:** 128 passed, 0 failed, 2 warnings (pre-existing)

---

## 1. Backend Test Suite

**Command:** `python -m pytest backend/tests/ -v --tb=short`  
**Result:** 128 passed, 0 failed (6.15 s)

| Module | Tests | Result |
|--------|-------|--------|
| `test_collision_avoidance.py` | Phase 5 CW equations + avoidance | All pass |
| `test_grid_analysis.py` | Phase 4 grid refinement | All pass |
| `test_ml_risk_prediction.py` | Phase 6 ML pipeline + API | All pass |
| `test_phase7_frontend_integration.py` | Trajectory API + static + workflow | All pass |

**Warnings (pre-existing, not regressions):**
`ConstantInputWarning: An input array is constant` — spearmanr() called on the constant -30
dummy baseline. Mathematically correct; the test passes by design.

---

## 2. Frontend 3D Rendering — Bugs Fixed This Session

### Bug 1 — Orbit Path Format Mismatch (NaN BufferGeometry)

**Root cause:** `upsertOrbitLine` called `ecefToThree(p[0], p[1], p[2])` (array indices)
but `/api/orbit_batch` returns `{x, y, z}` objects. All orbit vertices were `undefined -> NaN`,
causing WebGL to render corrupted full-screen quads.

**Fix:** Read `p.x ?? p[0]`; filter NaN vertices before building `BufferGeometry`;
set `frustumCulled = false`.

### Bug 2 — PointsMaterial World-Space Point Size (Sky-Blue Fill)

**Root cause:** `PointsMaterial({ size: 4.5, sizeAttenuation: true })` — with
`sizeAttenuation: true`, size is in world-space units. 4.5 units = 28,670 km. Each
satellite dot rendered as a screen-filling quad, completely obscuring the Earth.

**Fix:** Changed to `sizeAttenuation: false` so `size: 4.5` = 4.5 screen pixels.

### Bug 3 — Avoidance Planner `toFixed()` Crash

**Root cause:** Multiple `.toFixed()` calls on candidate fields that can be `undefined`.

**Fix:** `fmtKm/fmtMs/fmtSign` helpers with `typeof v === 'number' && isFinite(v)` guards.

**Visual verification:** Earth globe, orbit lines, satellite dots all render correctly.

---

## 3. ML Pipeline Validation (Phase 6)

### 3.1 Data Leakage — PASS

Command: `load_dataset(DEFAULT_FEAT_PATH, DEFAULT_SPLIT_PATH)` -> check intersection.

| Metric | Value |
|--------|-------|
| Train events | 10,349 |
| Val events | 2,580 |
| Overlap | 0 (no leakage) |

9 `NEVER_IN_X` columns excluded: `risk`, `true_risk`, `max_risk_estimate`,
`max_risk_scaling`, `event_final_risk`, `event_id`, `mission_id`, `is_last_cdm`, `split`.

### 3.2 Target Distribution

| Split | Rows | Range | Censored (== -30) |
|-------|------|-------|-------------------|
| Train | 120,092 | [-30.00, -1.68] | ~77% |
| Val | 29,388 | [-30.00, -3.35] | 77.6% |

### 3.3 Notebook / Training Reproducibility — PASS

Command: `python backend/scripts/train_ml_risk.py`  
Runtime: 23.51 s. All artifacts regenerated identically.

| Model | Val MAE | Val RMSE | Spearman rho |
|-------|---------|----------|--------------|
| Dummy floor baseline (-30) | 3.7768 | 8.5697 | undefined (constant) |
| Dummy mean baseline | 5.9990 | 7.6939 | undefined |
| Ridge linear | 5.0937 | 7.0256 | 0.3629 |
| HistGBR (primary) | 3.8143 | 5.8967 | 0.5556 |

### 3.4 Regressor vs. Baseline — Honest Comparison

| Metric | Model (live) | Constant -30 Baseline | Better? |
|--------|--------------|-----------------------|---------|
| MAE | 3.7883 | 3.7768 | NO — worse by 0.012 |
| RMSE | 5.8954 | 8.5697 | YES — better by 2.674 |

**NOTE:** The model MAE is marginally worse than the floor baseline. This is expected and
documented: 77.6% of val rows are at the censored floor, so the constant -30 baseline
scores deceptively well on MAE. The model is substantially better on RMSE (+2.67) and
Spearman rho=0.556. Do not claim the model beats the floor baseline on MAE.

### 3.5 Classifier Metrics at Explicit Thresholds

High-risk definition: log10(risk) >= -6.0 (ESA action threshold)  
Val high-risk CDMs: 426 / 29,388 = 1.45% (severely imbalanced)

| Threshold | Precision | Recall | F1 |
|-----------|-----------|--------|----|
| 0.05 | 0.0829 | 0.6573 | 0.1473 |
| 0.10 | 0.1090 | 0.5822 | 0.1836 |
| 0.20 | 0.1425 | 0.4789 | 0.2196 |
| 0.50 (default) | 0.2121 | 0.2723 | 0.2384 |

PR-AUC = 0.1827  
ROC-AUC = 0.8899

Top features (permutation importance): c_position_covariance_det (0.124),
mahalanobis_distance (0.096), miss_distance (0.038), c_sigma_t (0.036),
time_to_tca (0.030).

---

## 4. ML Prediction Consistency Bug — FIXED

**Defect:** `predict_cdm` set `is_high_risk` from classifier probability (>= 0.5)
while `risk_category`/`action_recommendation` were set from regressor threshold (>= -6.0).
These models disagreed, producing contradictory outputs:

  clamped_log10_risk = -20.70 => risk_category = "LOW"
  but is_high_risk = true (classifier said 0.77 prob)

**Fix:** `is_high_risk` is now always derived from the regressor threshold for consistency
with `risk_category` and `action_recommendation`. `high_risk_probability` is preserved as
an informational ensemble field.

**After fix:**
  clamped_log10_risk = -20.70, risk_category = "LOW", is_high_risk = false (CONSISTENT)
  high_risk_probability = 0.77 (classifier viewpoint, for ensemble context)

File changed: `backend/ml_risk_engine.py` (~line 472-478)
Tests: 128/128 still pass after fix.

---

## 5. Collision Avoidance Scientific Limitations (Phase 5)

### Rectified defects (from Phase 5 audit — verified numerically)

| Defect | Original Error | After Fix |
|--------|----------------|-----------|
| Near-circular GVE singularity | 8,253 km miss (75 deg orbital phase leap) | +5.681 km realistic improvement |
| RTN frame transport theorem | 9.19 m/s velocity error vs RK4 | 0.011 m/s residual |
| Differential drag factor | 0.75x (wrong) | 1.5x (derived from GVE) |

### Remaining unresolved limitations (not to be claimed as validated)

1. **Baseline miss = 0.0 km for ISS/POISK:** Live API returns 0.0 for NORAD 25544/36086
   at current epoch. ISS and POISK are co-orbiting modules — no close approach in current
   screening window. Engine correctly reports NO_FEASIBLE_MANEUVER. Expected behavior.

2. **CW linearization error:** CW post-burn miss (17.526 km) vs independent RK4 (19.984 km)
   = 2.458 km discrepancy over 60-min propagation. Inherent limitation of linearized
   Hill/CW model. Not a defect — documented limitation.

3. **SGP4 accuracy:** ~1 km/day errors; does not natively model maneuvers. Documented.

4. **Differential drag density model:** Uses fixed altitude-derived density, not
   NRLMSISE-00 or Jacchia. Documented limitation.

---

## 6. API Endpoint Validation

| Endpoint | Status | Notes |
|----------|--------|-------|
| /api/health | 200 | satellites_loaded: 16684 |
| /api/satellites | 200 | Returns catalog |
| /api/propagate | 200 | 16,665 positions, 0 errors |
| /api/orbit_batch | 200 | {x,y,z} objects (now handled correctly) |
| /api/screen | 200 | Disclaimer present |
| /api/analyse | 200 | Grid refinement |
| /api/avoidance/plan | 200 | 4 candidates, NO_FEASIBLE_MANEUVER (correct) |
| /api/avoidance/trajectory | 200 | [x,y,z] arrays (correctly handled) |
| /api/v1/ml/model_info | 200 | v1.0.0-phase6, 98 features |
| /api/v1/ml/predict_risk | 200 | is_high_risk now consistent |
| /api/v1/ml/predict_batch | 200 | Batch inference works |

---

## 7. Git Status

Modified (not committed):
- backend/app.py
- backend/ml_risk_engine.py  <- is_high_risk fix (this session)
- frontend/css/style.css
- frontend/index.html
- frontend/js/app.js  <- orbit path fix, sizeAttenuation fix, toFixed guards

Untracked (Phase 6/7 additions):
- backend/ml_risk_engine.py, backend/models/, backend/scripts/train_ml_risk.py
- backend/tests/test_ml_risk_prediction.py, test_phase7_frontend_integration.py
- data/processed/reports/phase6_ml_evaluation_report.json
- docs/phase6_ml_risk_prediction_report.md, docs/phase7_frontend_visualization_report.md
- notebooks/01_collision_risk_prediction.ipynb

All changes functionally verified. Recommend committing before Phase 7.

---

## 8. Readiness Summary

| Area | Status |
|------|--------|
| Backend test suite | 128/128 pass |
| 3D visualization (globe, orbits, dots) | Fixed and verified |
| Avoidance planner crash | Fixed |
| ML data leakage | Verified absent (0 overlap) |
| ML train/val disjointness | Proven |
| ML baseline comparison (MAE/RMSE) | Honestly reported |
| ML classifier P/R/F1/PR-AUC | Reported at 4 thresholds |
| ML prediction consistency | Fixed |
| Notebook reproducibility | Verified (23.51 s) |
| Phase 5 CW limitations | Documented |
| API endpoint health | All responding |
| Unresolved scientific limitations | Documented (not hidden) |

VERDICT: Ready for Phase 7. No critical defects remain.
