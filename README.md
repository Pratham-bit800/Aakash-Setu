# Aakash Setu

**Aakash Setu** (Sanskrit: *Bridge to the Sky*) is a research prototype for real-time satellite
orbit tracking, conjunction screening, collision avoidance planning, and ML-based risk assessment.

> **Research Prototype Disclaimer:** All outputs are mathematical estimates derived from TLE
> ephemerides, statistical models trained on historical ESA Kelvins CDMs, and linearized orbital
> mechanics. They are **not** operationally certified collision probability (Pc) values and must
> not be used for mission-critical decision-making.

---

## Features

| Feature | Description |
|---------|-------------|
| **3D Orbit Tracker** | Three.js visualization of 16,000+ satellites with SGP4 propagation |
| **Conjunction Screening** | Configurable threshold and horizon screening over the live catalog |
| **Encounter Grid Refinement** | Nested adaptive grid refinement to 0.01 m resolution |
| **Avoidance Planning** | Impulsive Δv candidates (prograde, retrograde, cross-track, radial) via CW equations |
| **Post-Burn Trajectory** | 3D rendering of post-maneuver orbit in the live scene |
| **ML Risk Prediction** | HistGradientBoosting regressor + classifier trained on ESA Kelvins CDMs |

---

## Quick Start

### Backend
```bash
cd backend
pip install flask flask-cors sgp4 numpy scipy scikit-learn joblib pyarrow
python app.py
```
Flask runs on `http://127.0.0.1:5000`. The frontend is served by Flask at the same URL.

---

## Project Structure

```
backend/
  app.py                    Flask API server (all routes)
  collision_screening.py    Fast orbital envelope screening engine
  grid_analysis.py          Nested adaptive grid refinement
  collision_avoidance.py    Hybrid avoidance planner (CW + SGP4)
  ml_risk_engine.py         ML pipeline: preprocessor, regressor, classifier
  models/phase6/            Saved model artifacts (joblib + metadata)
  scripts/
    train_ml_risk.py        Retrain ML models from processed data
    inspect_esa.py          ESA dataset inspection utility
    validate_celestrak.py   Celestrak TLE validation utility
  tests/                    Full test suite (128 tests)

frontend/
  index.html                Single-page application
  css/style.css             Styles
  js/app.js                 Three.js 3D scene + all UI logic

data/
  raw/                      Source TLE and ESA CDM datasets
  processed/                Parquet features, split indices, reports

notebooks/
  01_collision_risk_prediction.ipynb   ML training and evaluation notebook

docs/                       Technical reports and audit documentation
```

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Server health and catalog count |
| `/api/satellites` | GET | Full satellite catalog |
| `/api/orbit_batch` | POST | Batch orbit propagation |
| `/api/screen` | GET | Conjunction screening |
| `/api/analyse/pair` | GET | Grid refinement for a satellite pair |
| `/api/avoidance/plan` | GET | Avoidance maneuver candidates |
| `/api/avoidance/trajectory` | GET | Post-burn 3D orbit path |
| `/api/v1/ml/predict_risk` | POST | ML risk prediction |
| `/api/v1/ml/predict_batch` | POST | Batch ML predictions |
| `/api/v1/ml/model_info` | GET | Model version and feature metadata |

---

## Scientific Limitations

- **SGP4 accuracy:** ~1 km/day. TLEs carry no covariance matrices; formal Pc requires 6×6 covariance.
- **CW model:** Linearized Clohessy-Wiltshire equations valid for near-circular orbits (e ≪ 1).
  Validated against independent RK4(J₂) within 188 m over 60-min arcs under identical conditions.
- **ML model:** MAE slightly worse than the −30 floor baseline (3.79 vs 3.78).
  RMSE better by 2.67. Spearman ρ = 0.56 vs 0.0 for the floor. Not certified for operational use.
- **Atmosphere:** Fixed exponential density model, not NRLMSISE-00/JB2008.
- **Impulsive Δv:** Real burns require finite-duration modeling with attitude control.

---

## Retraining the ML Models

```bash
python backend/scripts/train_ml_risk.py
```
Requires `data/processed/esa_train_features.parquet` and `data/processed/esa_event_split.parquet`.

---

## Running Tests

```bash
python -m pytest backend/tests/ -q
# Expected: 128 passed, 0 failed, 2 warnings
```
