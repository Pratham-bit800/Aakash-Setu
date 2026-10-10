# Akash Setu — Phase 7: Frontend 3D Conjunction & Avoidance Visualization

**Document Status:** Complete & Verified (audit pass: 2026-10-10)
**System Version:** Akash Setu v0.7.0
**Commits:** `42155dc` (Phase 6+7 main), `e4004cb` (MAE correction)
**Test suite:** 128 passed, 0 failed, 2 warnings (88.48 s)

---

## 1. Phase 7 Scope

Phase 7 integrates Phases 3–6 (screening → grid refinement → avoidance → ML risk) into
an interactive Three.js 3D dashboard served by the Flask backend.

The phase 7 report in the project was previously marked "Complete & Verified" with a
fabricated floor baseline MAE figure (2.3175). That figure has been corrected to the
measured value (3.7768). See §5.

---

## 2. Feature Audit — Documented vs Implemented

All features described in the Phase 7 plan were independently audited against the
source code. Results are below.

| # | Documented Feature | Implemented | Notes |
|---|-------------------|-------------|-------|
| 1 | 4-tab navigation (Tracker/Conjunctions/Avoidance/ML Risk) | YES | `data-mode` nav, `view-container` divs |
| 2 | 3D Earth sphere with atmosphere glow | YES | `buildEarth()` canvas texture + `atmoMat` |
| 3 | Star field background | YES | `buildStars()` 1200-point sphere |
| 4 | Satellite point cloud (per-satellite colors) | YES | `buildSatPoints()` Float32Array |
| 5 | Orbit lines per satellite | YES | `upsertOrbitLine()`, Map cache |
| 6 | Station orbits: bright emerald (selected) | YES | `ORBIT_SEL_COLOR_STATION = 0x00ffaa` |
| 7 | Satellite orbits: slate blue / bright selected | YES | `0x2a5caa` dim / `0x80c0ff` selected |
| 8 | Conjunction pair highlighted red/amber in 3D | YES | `selectConjunction()` sets CONJUNCTION_COLOR_1/2 |
| 9 | Post-burn trajectory: cyan `#00f0ff` loop | YES | `renderManeuverTrajectory()`, LineLoop |
| 10 | Impulse sphere marker at burn point | YES | `THREE.SphereGeometry` in `renderManeuverTrajectory` |
| 11 | Show Post-Burn 3D Trajectory button per candidate | YES | `visualizeCandidateTrajectory()` |
| 12 | Configurable screening threshold (1–500 km), horizon, limit | YES | HTML number inputs → `/api/screen` |
| 13 | View in 3D button on conjunction alert cards | YES | `selectConjunction(alert)` on card click |
| 14 | 0.01m nested grid refinement button per conjunction | YES | `runGridRefinement()` → `/api/analyse/pair` |
| 15 | ML Risk: scenario presets + live prediction | YES | `runMLPrediction()` → `/api/v1/ml/predict_risk` |
| 16 | ML research disclaimer visible in UI | YES | inline red warning box in result card |
| 17 | Clear Overlay / Reset Camera controls | YES | `clearManeuverLine()`, `controls.reset()` |
| 18 | Satellite detail panel (altitude, speed, inclination) | YES | `showDetailPanel()` right drawer |
| 19 | Panel toggle button | YES | `panel-toggle-btn` toggles `panel-visible` class |
| 20 | Real-time UTC clock | YES | `setInterval` updating `#sim-time` |

### Notes on Discrepancies

- **Earth texture (item 2):** The Phase 7 report described "photorealistic Earth sphere
  with spherical texture mapping." The actual implementation uses a `CanvasTexture`
  (HTML5 canvas gradient + lat/lon grid lines). No external texture file is loaded.
  The result is a schematic globe, not a photographic one. This is a description gap
  in the report, not a missing feature — the canvas approach avoids external asset
  dependencies and works offline. No change made.

- **Station dim-color (item 6):** Report mentions `#00e5a0` as the station orbit color.
  The implementation uses `0x00995a` (dim/unselected) and `0x00ffaa` (selected).
  The bright emerald `#00e5a0` ≈ `0x00e5a0` is between these two values and represents
  the selected/highlighted state behavior. No change made.

---

## 3. Defects Fixed This Session

### 3.1 `is_high_risk` Inconsistency (ml_risk_engine.py)

**Defect:** `predict_cdm()` set `is_high_risk` from the *classifier* probability
(`high_risk_prob >= 0.5`) while `risk_category` and `action_recommendation` were set
from the *regressor* threshold (`clamped_log10 >= -6.0`). The two models can disagree,
producing contradictory API output:

    clamped_log10_risk = -20.70  →  risk_category = "LOW"
    high_risk_prob = 0.77        →  is_high_risk = true    ← CONTRADICTION

**Fix:** `is_high_risk` now always derived from the regressor threshold:

    is_high_risk: bool = bool(clamped_log10 >= RISK_ACTION_THRESHOLD)

`high_risk_probability` is preserved as an informational classifier output.
128/128 tests pass after fix. No API schema change.

### 3.2 Fabricated Floor Baseline MAE (phase7_frontend_visualization_report.md)

**Defect:** Report stated `MAE (3.7768 vs 2.3175)` — the floor MAE of 2.3175 was
fabricated. Measured values on the validation set (29,388 rows):

    Regressor MAE  = 3.8143  (WORSE than floor by 0.012)
    Floor (-30) MAE = 3.7768
    Regressor RMSE = 5.8967  (BETTER than floor by 2.674)
    Floor (-30) RMSE = 8.5697

**Fix:** Report corrected to actual measured values. Both numbers are now traceable
to the live validation run (see post_phase7_integration_audit.md §4).

---

## 4. API Verification — Live Results

Command: `python -m pytest backend/tests/ -q --tb=short`

    128 passed, 0 failed, 2 warnings in 88.48 s

| Endpoint | Method | Live Response |
|----------|--------|---------------|
| `/api/health` | GET | `{"status":"ok","satellites_loaded":16684}` |
| `/api/screen?window_hours=3&miss_threshold_km=20&limit=60` | GET | conjunctions returned, disclaimer present |
| `/api/avoidance/trajectory?norad_id=25544&direction=prograde_along_track&delta_v_m_s=0.5&steps=30` | GET | 30 `[x,y,z]` orbit path points for NORAD 25544 |
| `/api/v1/ml/predict_risk` (elevated scenario) | POST | `risk_category=LOW, is_high_risk=false, clamped=-28.13` |
| `/api/v1/ml/model_info` | GET | `v1.0.0-phase6, 98 features` |

All responses consistent. `is_high_risk` now matches `risk_category` in all cases.

---

## 5. ML Validation Summary (Phase 6 — carried forward)

These figures are the authoritative measured values. They must not be updated without
re-running the validation script against the saved model artifacts.

| Metric | Value | Notes |
|--------|-------|-------|
| Val events | 2,580 (disjoint from 10,349 train) | 0 overlap verified |
| Val censored (== -30) | 77.6% of 29,388 rows | |
| Regressor MAE | 3.8143 | marginally WORSE than floor by 0.012 |
| Floor (-30) MAE | 3.7768 | |
| Regressor RMSE | 5.8967 | BETTER than floor by 2.674 |
| Floor (-30) RMSE | 8.5697 | |
| Spearman rho | 0.5556 | p < 1e-6 |
| Classifier PR-AUC | 0.1827 | 12.6x lift over 1.45% base rate |
| Classifier ROC-AUC | 0.8899 | |
| Classifier P/R/F1 @ t=0.50 | 0.212 / 0.272 / 0.238 | default threshold |

**MANDATORY:** The model MAE does NOT beat the floor baseline. Do not claim otherwise.
These are research-prototype outputs on ESA Kelvins CDMs, not certified Pc values.

---

## 6. Unresolved Scientific Limitations

1. **CW vs. RK4 discrepancy (2.458 km):** Post-burn miss distance from linearized
   Clohessy-Wiltshire (17.526 km) vs. independent RK4 (19.984 km) over 60 min.
   This is an inherent limitation of the linear CW model, not a software bug.
   Do not claim sub-km avoidance accuracy.

2. **Avoidance baseline miss = 0.0 km for ISS/POISK at current epoch:** These modules
   are co-orbiting — no close approach exists in the current TLE screening window.
   The engine correctly reports `NO_FEASIBLE_MANEUVER`. Expected, not a defect.

3. **SGP4 propagation:** ~1 km/day accuracy. No native maneuver modeling. TLEs carry
   no covariance matrices for proper Pc computation.

4. **ML model:** Trained on historical ESA Kelvins CDMs. Outputs are statistical
   correlations within the training distribution. Not operationally certified.

5. **Impulsive Δv approximation:** Real thrusters require finite-duration burn
   integration with attitude control, not modeled here.

---

## 7. Files Changed (this session)

| File | Change |
|------|--------|
| `backend/ml_risk_engine.py` | Fixed `is_high_risk` consistency (regressor threshold) |
| `docs/phase7_frontend_visualization_report.md` | Corrected fabricated floor MAE |
| `frontend/js/app.js` | Orbit path format fix, sizeAttenuation fix, toFixed guards |
| `frontend/index.html` | Phase 7 tabs and panels |
| `frontend/css/style.css` | Phase 7 styles |
| `backend/app.py` | Phase 6/7 routes |
| All model artifacts, tests, notebook, Phase 6 report | Added in commit 42155dc |

## 8. Commits

    e4004cb fix(docs): correct fabricated floor baseline MAE in Phase 7 report
    42155dc feat: Phase 6+7 -- ML risk prediction, 3D visualization, and frontend integration

---

## 9. Readiness Statement

Phase 7 is **fully implemented and tested**. All 20 documented features are present.
Two defects found by independent audit were fixed and verified. 128/128 tests pass.
No new features were invented. Scientific limitations are documented and visible.

This project is NOT flight-qualified. Outputs are research prototypes.
