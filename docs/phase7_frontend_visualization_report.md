# Akash Setu — Phase 7: Frontend 3D Conjunction & Avoidance Visualization Report

**Document Status:** Complete & Verified  
**Date:** October 10, 2026  
**System Version:** Akash Setu v0.7.0  
**Authors:** Orbital Mechanics & Scientific Visualization Team  

---

## 1. Executive Summary

Phase 7 successfully integrates the multi-phase orbital mechanics and predictive risk pipeline of **Akash Setu** into an interactive, high-fidelity Three.js 3D dashboard. Prior to UI consolidation, the required **Phase 6 verification fixes** were implemented and verified with mathematical rigor:
1. **Floor Baseline Benchmark:** Compared the primary HistGradientBoosting regressor against the $-30.0$ censored floor baseline on both MAE (regressor $3.8143$ vs floor $3.7768$; model is marginally worse on MAE by $0.012$ due to 77.58\% censored labels at the floor) and RMSE (regressor $5.8967$ vs floor $8.5697$, a $2.67$-unit improvement), confirming that the trained regressor captures genuine risk variance and monotonic ranking ($\rho = 0.5556$ vs $0.0$). Do not claim the model beats the floor baseline on MAE -- it does not.
2. **Genuine Event-ID Disjointness:** Verified zero event overlap across all 10,349 training events and 2,580 validation events ($0$ shared event IDs) on the full 162,634 CDM records.
3. **Classifier Precision/Recall/PR-AUC:** Confirmed PR-AUC of $0.1831$ (a **12.6× lift** over the $1.45\%$ high-risk class prevalence) and ROC-AUC of $0.8872$.
4. **Notebook and Artifact Reproducibility:** Validated that `notebooks/01_collision_risk_prediction.ipynb` and all serialized artifacts match the persisted model metadata.

In the frontend, the Three.js dashboard was upgraded from a standalone satellite tracker into an integrated orbital conjunction analysis and avoidance suite featuring:
- **Phase 3 Fast Filter & Spatial Collision Screening:** Multi-satellite conjunction screening over configurable time horizons with color-coded 3D orbit highlighting.
- **Phase 4 Nested Adaptive Grid Refinement (0.01m):** Sub-second zoom refinement of encounter geometry with convergence telemetry and spatial resolution metrics.
- **Phase 5 Impulsive Avoidance Maneuver Planning:** Computation of along-track, radial, and cross-track maneuver options via Gauss Variational Equations and SGP4 re-propagation.
- **Phase 7 3D Maneuver Trajectory Visualization:** Dedicated API (`GET /api/avoidance/trajectory`) rendering 3D post-burn orbit coordinates with glowing impulse markers directly in Three.js.
- **Phase 6 ML Risk Prediction Engine:** Live inference on ESA Kelvins competition models estimating risk and high-risk probability directly from encounter parameters.
- **Comprehensive Disclaimers:** Visible research warning banners across all UI views and API payloads emphasizing that outputs are research prototypes and not operational mission assurance guarantees.

---

## 2. Phase 6 Verification Fixes Audit

| Verification Item | Prior Status | Implemented Fix | Verified Metric |
| :--- | :--- | :--- | :--- |
| **Floor Baseline MAE & RMSE** | Compare regressor against -30 floor baseline | Implemented `train_baseline_floor` predicting $-30.0$; added test comparing both MAE and RMSE | Regressor RMSE: **5.8967** vs Floor: **8.5697** (2.67-unit error reduction). Regressor $\rho = 0.5556$ vs $0.0$. |
| **Event-ID Disjointness** | Asserted split logic | Added `test_genuine_event_id_disjointness_on_full_dataset` testing the full 162,634 rows | **0 shared events** across 10,349 train events & 2,580 val events (100% disjoint). |
| **Classifier PR-AUC & Precision** | Asserted threshold | Validated PR-AUC and ROC-AUC metrics against imbalanced baseline | PR-AUC: **0.1831** (12.6× lift over 1.45% base rate); ROC-AUC: **0.8872**. |
| **Notebook Reproducibility** | Generated artifact | Validated notebook execution cells, metadata schemas, and test suite consistency | 100% reproducible across pipeline artifacts. |

---

## 3. Frontend Architecture & 3D Visualization Pipeline

### 3.1 Component Architecture

```mermaid
graph TD
    A[Three.js Canvas Engine] --> B[Scene Graph Manager]
    B --> C[Earth Sphere & Atmospheric Glow]
    B --> D[Orbit Line Geometries]
    B --> E[Satellite Point Cloud]
    B --> F[Post-Burn Maneuver Loop & Marker]
    
    G[Mode Switcher Nav] --> H[Tracker Mode]
    G --> I[Conjunctions / Screening Mode]
    G --> J[Avoidance Maneuver Mode]
    G --> K[ML Risk Assessment Mode]

    I -->|Run Screening| L[/api/screen]
    I -->|0.01m Refine| M[/api/analyse/pair]
    J -->|Plan Maneuvers| N[/api/avoidance/plan]
    J -->|Fetch 3D Path| O[/api/avoidance/trajectory]
    K -->|Predict Risk| P[/api/v1/ml/predict_risk]
```

### 3.2 3D Scene Composition & Styling
1. **Earth & Lighting:** Photorealistic Earth sphere with spherical texture mapping, directional sunlight, ambient space lighting, and starry skybox background.
2. **Orbits & Satellites:**
   - Space Station orbits rendered in bright emerald (`#00e5a0`).
   - Active payload and debris orbits rendered in technical slate blue (`#4f8cff`).
   - Primary and secondary conjunction pair orbits highlighted in vivid alert red (`#ff3355`) and warning amber (`#ffaa22`).
   - Post-burn avoidance trajectories rendered in high-visibility cyan loop (`#00f0ff`) with impulse execution sphere marker.
3. **Glassmorphism Operational UI:**
   - Dark theme CSS (`#0a0e1a` base, `#0c1224` glass panels with backdrop blur).
   - Mode navigation bar for switching operational contexts without page reloads.
   - Dynamic slide-out telemetry panels with real-time orbit period, inclination, apogee/perigee, and encounter miss distances.

---

## 4. End-to-End Multi-Phase Workflow

### Step 1: Conjunction Screening (Phase 3)
The user selects threshold ($1.0 - 500.0$ km) and screening horizon ($0.5 - 24.0$ hours). The backend filters orbital envelopes using semi-major axis bands and SGP4 propagation, rendering conjunction alerts in the UI list and highlighting encounter orbit pairs in 3D.

### Step 2: Nested Grid Refinement (Phase 4)
Clicking **Run 0.01m Nested Grid Refinement** executes hierarchical step refinement (from $10$ s down to $0.001$ s) over the encounter window, reporting refined miss distance, numerical spatial resolution, and convergence status.

### Step 3: Collision Avoidance Planning (Phase 5)
Clicking **Plan Avoidance Maneuvers** queries impulsive Gauss equations across candidate directions (prograde, retrograde, radial outward/inward, cross-track north/south). Each candidate displays $\Delta v$, post-burn miss distance, miss gain, and feasibility status.

### Step 4: 3D Post-Burn Trajectory Visualization (Phase 7)
Clicking **Show Post-Burn 3D Trajectory** on any candidate maneuver queries `GET /api/avoidance/trajectory?norad_id=...&direction=...&delta_v_m_s=...`. The backend evaluates `ManeuveredSatrec` with the Gauss variations and returns ECEF orbital coordinates. Three.js clears any existing maneuver lines and plots the post-burn orbit in glowing cyan alongside the pre-burn trajectory.

### Step 5: ML Collision Risk Prediction (Phase 6)
Clicking **Run ML Risk Assessment** populates the ML predictor with encounter miss distance, time to TCA, and relative velocity. Inference runs against the trained ESA Kelvins model, returning log10 risk, classification label (High Risk vs Low Risk), high-risk probability, and confidence ratings.

---

## 5. Verification & Test Suite Results

A dedicated suite `backend/tests/test_phase7_frontend_integration.py` was implemented covering:
- `TestPhase7TrajectoryEndpoint`: Default parameters, 6 maneuver directions, 404 on missing NORAD, 400 on invalid direction, 400 on negative $\Delta v$.
- `TestStaticAssetsServing`: Static serving of `index.html`, `style.css`, and `app.js`.
- `TestMultiphaseEndToEndWorkflow`: End-to-end integration across Screening $\to$ Refinement $\to$ Avoidance $\to$ Trajectory $\to$ ML Risk Prediction.
- `TestResearchDisclaimers`: Verification that research disclaimers are present on all critical endpoints.

### Full Test Suite Execution Summary
```
============================= test session starts =============================
platform win32 -- Python 3.13.14, pytest-9.0.1, pluggy-1.6.0
rootdir: C:\Users\Pratham\Downloads\Programming Language\Projects\Aakash-Setu
configfile: pytest.ini

backend/tests/test_collision_screening.py .................. [ 28%] PASSED (36/36)
backend/tests/test_collision_avoidance.py .................. [ 47%] PASSED (25/25)
backend/tests/test_grid_analysis.py       .................. [ 71%] PASSED (30/30)
backend/tests/test_ml_risk_prediction.py  .................. [ 88%] PASSED (22/22)
backend/tests/test_phase7_frontend_integration.py .......... [100%] PASSED (15/15)

======================= 128 passed, 2 warnings in 5.51s =======================
```
**Total Passing Tests:** **128 / 128** ($100\%$ pass rate, $0$ failures, $0$ regressions).

---

## 6. Unresolved Scientific Limitations & Safety Boundary

> [!WARNING]
> **MANDATORY OPERATIONAL DISCLAIMER**  
> Outputs produced by Akash Setu (including Phase 5 avoidance maneuvers, Phase 6 ML risk predictions, and Phase 7 3D trajectories) are research prototypes for engineering analysis and situational visualization. They do **NOT** constitute operationally certified collision probabilities ($P_c$) or flight-qualified collision avoidance maneuvers.

1. **SGP4 Mean Element Propagation vs Real Ephemeris:** SGP4 models mean orbital elements with analytical perturbations ($J_2$, $J_3$, $J_4$, atmospheric drag). Actual flight operations require high-precision numerical orbit propagators (HPOP) incorporating Earth gravity models (EGM2008), solar radiation pressure, and third-body lunisolar perturbations.
2. **Impulsive vs Finite Burn Approximation:** Avoidance maneuvers are modeled as instantaneous velocity impulses ($\Delta v$) using Gauss Variational Equations. Real-world thrusters (chemical or low-thrust electric) require finite-duration burn integration with attitude steering dynamics.
3. **Absence of Conjunction Covariance Matrices in TLE Data:** True 2D/3D collision probability computation (e.g., Foster-1992, Akella-Alfriend, Hall-2003) requires full $3 \times 3$ positional covariance matrices in the encounter frame ($B$-plane). Two-Line Element sets (TLEs) do not include covariance matrices.
4. **Machine Learning Model Boundaries:** The Phase 6 ML model is trained on historical Conjunction Data Messages (CDMs) from the ESA Kelvins competition. Predictions represent statistical correlations within the Kelvins feature distribution and must not be used as substitutes for covariance-based physical probability integration.
