# Phase 5: Hybrid Collision Avoidance - Implementation & Scientific Report

**Project:** Akash Setu
**Phase:** 5 - Hybrid Collision Avoidance
**Date:** 2026-10-10
**Git branch:** branch1
**Status:** COMPLETE - 80/80 tests pass, zero regressions, full API integration verified

---

## 1. Executive Summary

Phase 5 introduces a scientifically rigorous **Hybrid Collision Avoidance Engine** to Akash Setu.
The engine addresses close-approach encounters identified by Phase 3 screening and refined by Phase 4
nested-grid analysis.

The engine implements a dual-path hybrid architecture:
1. **Attitude Reorientation Assessment (Aerodynamic Differential Drag):**
   - Evaluated strictly when spacecraft 3D geometry and attitude constraints are available.
   - When spacecraft geometry is absent (the standard TLE catalog condition), the engine explicitly
     documents the physical data gap and reports `UNAVAILABLE_NO_GEOMETRY`, avoiding fabrication of
     unvalidated spacecraft physical parameters.
   - When geometry is provided (cross-sectional area variation, mass, drag coefficient), differential
     drag acceleration and along-track drift are evaluated quantitatively over the available lead time.
2. **Impulsive $\Delta v$ Orbital Maneuvers (Validated Astrodynamics):**
   - Applies impulsive burns in the local Radial-Transverse-Normal (RTN/RSW) orbital frame.
   - Formulated via Gauss's Variational Equations (GVE) on mean Keplerian elements.
   - Supports prograde along-track, retrograde along-track, out-of-plane cross-track, radial, and custom 3D vectors.
   - Integrates with the existing SGP4 propagator via `ManeuveredSatrec`, preserving the unmaneuvered
     state before burn epoch $t_{\text{man}}$ and propagating the maneuvered state thereafter.
   - Re-propagates and re-screens candidate maneuvers using the existing Phase 3 screening pipeline and
     Phase 4 grid-refinement engine.
   - Evaluates operational feasibility, clearance thresholds, propellant budgets, re-entry safety,
     and collateral conjunction risks.

Two new Flask API endpoints expose the engine:
- `GET /api/avoidance/plan`: Multi-strategy comparison, ranking, and hybrid recommendation.
- `GET /api/avoidance/evaluate`: Single-maneuver candidate evaluation.

All 80 tests in the project suite pass with zero failures. All existing endpoints and the Three.js dashboard
continue to function without disruption.

---

## 2. Reference-Document Status

**No external reference document was provided with the Phase 5 specification.**

The specification references "the reference-supported avoidance method," but no external PDF,
paper, or document was attached or located in the workspace. In accordance with strict instructions:
  - Reference fidelity is NOT claimed, as no reference document was provided to be faithful to.
  - No speculative or invented maneuver rules are introduced.
  - All mathematical equations and physical formulations are derived directly from validated,
    peer-reviewed astrodynamics textbooks:
    - **Vallado, D. A. (2013)**: *Fundamentals of Astrodynamics and Applications* (4th ed.), Microcosm Press / Springer, Section 6.5 (Orbital Maneuvers) & Section 9.5 (Conjunction Assessment).
    - **Alfriend, K. T., et al. (2010)**: *Spacecraft Formation Flying: Dynamics, control and navigation*, Elsevier, Chapter 4 (Gauss's Variational Equations).
    - **Battin, R. H. (1999)**: *An Introduction to the Mathematics and Methods of Astrodynamics*, AIAA Education Series.
    - **Clohessy, W. H., & Wiltshire, R. S. (1960)**: *Terminal Guidance System for Satellite Rendezvous*, Journal of the Aerospace Sciences.

---

## 3. Data Gaps and Scientific Limitations Documented

In compliance with the mandate to document data gaps rather than fabricating results:

### 3.1. Spacecraft Geometry & Attitude Constraints Gap
- **Finding:** Two-Line Element (TLE) sets and CelesTrak JSON catalogs (`active_satellites.json`, `stations.json`) provide mean orbital elements and a scalar drag coefficient ($B^*$). They contain **zero** spacecraft CAD geometry, cross-sectional area profiles, moments of inertia, thruster orientations, reaction wheel torque limits, or slew rate boundaries.
- **Handling:** When geometry parameters are missing, `evaluate_attitude_reorientation()` returns:
  - `feasible: false`
  - `status: "UNAVAILABLE_NO_GEOMETRY"`
  - `gap_documentation: "Spacecraft 3D geometry, cross-sectional area profile, and attitude actuator constraints are absent in the TLE dataset. Attitude reorientation cannot be evaluated without inventing unvalidated spacecraft specifications."`
- **When Geometry Is Supplied:** The engine accepts `spacecraft_geometry: {"area_min_m2", "area_max_m2", "mass_kg", "drag_coeff"}` and computes differential aerodynamic drag quantitatively.

### 3.2. Ephemeris Accuracy vs. Maneuver Resolution
- **Finding:** SGP4 TLE position errors range from ~100 m (fresh TLE < 1 day) to >10 km (stale TLE 7–14 days). Small impulsive maneuvers ($\Delta v \sim 0.01$ to $0.5$ m/s) produce displacement deltas that are physical within the mathematical model, but in reality sit below or near the TLE ephemeris uncertainty floor.
- **Handling:** An explicit `accuracy_disclaimer` is included in every API response and plan result:
  > *"Phase 5 collision avoidance plans are propagated mathematical estimates derived from SGP4 mean orbital elements and Gauss's Variational Equations. They do NOT constitute operational spacecraft flight commands. TLE ephemeris errors, lack of spacecraft propulsion telemetry, and absence of covariance data make these results unsuitable for real-time mission execution without independent verification."*

---

## 4. Astrodynamic Formulations (As Implemented)

### 4.1. Local Orbital Frame (RTN / RSW)
At maneuver time $t_{\text{man}}$, the local orbital frame is defined by the primary spacecraft state $(\mathbf{r}, \mathbf{v})$ in TEME coordinates:
$$\hat{\mathbf{R}} = \frac{\mathbf{r}}{\|\mathbf{r}\|}, \quad \hat{\mathbf{W}} = \frac{\mathbf{r} \times \mathbf{v}}{\|\mathbf{r} \times \mathbf{v}\|}, \quad \hat{\mathbf{T}} = \hat{\mathbf{W}} \times \hat{\mathbf{R}}$$
An impulsive maneuver is specified as $\Delta \mathbf{v} = \Delta v_R \hat{\mathbf{R}} + \Delta v_T \hat{\mathbf{T}} + \Delta v_W \hat{\mathbf{W}}$.

### 4.2. Gauss's Variational Equations (GVE)
From Vallado (2013) §6.5 and Alfriend (2010) §4, the variations in mean Keplerian elements are:

1. **Semi-major axis change ($\Delta a$):**
   $$\Delta a = \frac{2 a^2 v}{\mu} \Delta v_T$$
2. **Mean motion change ($\Delta n$):**
   $$\Delta n = -\frac{3}{2} \frac{n}{a} \Delta a$$
3. **Eccentricity change ($\Delta e$):**
   $$\Delta e = \frac{1}{v} \left( \sin\nu \Delta v_R + 2 \cos\nu \Delta v_T \right)$$
4. **Inclination change ($\Delta i$):**
   $$\Delta i = \frac{\cos u}{v} \Delta v_W$$
5. **RAAN change ($\Delta \Omega$):**
   $$\Delta \Omega = \frac{\sin u}{v \sin i} \Delta v_W \quad (\text{for } \sin i > 10^{-4})$$
6. **Argument of perigee change ($\Delta \omega$):**
   $$\Delta \omega = \frac{1}{e v} \left( -\cos\nu \Delta v_R + 2 \sin\nu \Delta v_T \right) - \Delta \Omega \cos i$$
7. **Orbital period change ($\Delta P$):**
   $$\Delta P = 2\pi \sqrt{\frac{(a + \Delta a)^3}{\mu}} - 2\pi \sqrt{\frac{a^3}{\mu}}$$

### 4.3. Aerodynamic Differential Drag Formulation
When spacecraft geometry is provided, differential drag acceleration between maximum and minimum projected area configurations is:
$$\Delta a_{\text{drag}} = \frac{1}{2} \rho v^2 \frac{C_D \Delta A}{m}$$
Over lead time $\tau_{\text{lead}} = t_{\text{TCA}} - t_{\text{man}}$, the secular along-track displacement is:
$$\Delta s_{\text{drag}} \approx \frac{3}{4} \Delta a_{\text{drag}} \tau_{\text{lead}}^2$$


---

## 5. Architecture and Engine Implementation

### 5.1. Files Created and Modified
| File | Action | Purpose |
|---|---|---|
| `backend/collision_avoidance.py` | NEW | Core Phase 5 collision avoidance engine, GVE solvers, ManeuveredSatrec wrapper, candidate evaluator, and plan generator. |
| `backend/app.py` | MODIFIED | Added `/api/avoidance/plan` and `/api/avoidance/evaluate` endpoints. |
| `backend/tests/test_collision_avoidance.py` | NEW | 14 focused tests verifying maneuvers, attitude gaps, feasibility, plan ranking, and APIs. |
| `docs/phase5_hybrid_avoidance_report.md` | NEW | Comprehensive technical, scientific, and verification report. |

### 5.2. ManeuveredSatrec Composite Object
To seamlessly integrate with existing SGP4 propagators without rewriting core functions:
- `ManeuveredSatrec` wraps `sat_orig` and `sat_man`.
- For $t < t_{\text{man}}$: delegates calls directly to `sat_orig.sgp4(jd, jdf)`.
- For $t \ge t_{\text{man}}$: delegates calls directly to `sat_man.sgp4(jd, jdf)`.
- This duck-typed class implements the identical `.sgp4(jd, jdf)` interface, enabling direct re-propagation through `_propagate_teme()`, `screen_satellites()`, and `refine_tca()`.

### 5.3. Re-propagation & Re-screening Pipeline
For every candidate maneuver:
1. The burn is applied at $t_{\text{man}} = t_{\text{TCA}} - \tau_{\text{lead}}$.
2. Post-maneuver orbital elements are computed and used to construct a `ManeuveredSatrec`.
3. The encounter window $[t_{\text{TCA}} - \text{hw}, t_{\text{TCA}} + \text{hw}]$ is re-screened using Phase 4 `refine_tca()` to determine the exact post-maneuver minimum separation and refined encounter time.
4. If a satellite catalog is supplied, `screen_satellites()` is executed on the maneuvered object against all other catalog satellites to detect any collateral conjunctions induced by the burn.

---

## 6. Feasibility Criteria and Constraints

A candidate maneuver is marked `is_feasible = True` if and only if all of the following conditions are satisfied:
1. **Delta-v Budget Constraint:** $\Delta v \le \Delta v_{\text{max}}$ (default 5.0 m/s).
2. **Operational Lead Time Constraint:** $\tau_{\text{lead}} \ge \tau_{\text{min\_notice}}$ (default 15.0 minutes).
3. **Target Clearance Constraint:** $d_{\text{after}} \ge d_{\text{target}}$ (default 15.0 km).
4. **Perigee Re-entry Safety Boundary:** $r_p - R_E \ge 120.0$ km (ensures burn does not lower perigee into dense atmosphere).
5. **Collateral Conjunction Awareness:** Flags any newly introduced close approaches with catalog objects.

---

## 7. New API Endpoints

### 7.1. GET /api/avoidance/plan
Compares multiple maneuver strategies and returns a recommended hybrid avoidance plan.

**Parameters:**
- `norad_a` (required, int): Primary satellite catalog ID.
- `norad_b` (required, int): Secondary satellite catalog ID.
- `delta_v_m_s` (optional, float, default 0.5): Nominal burn magnitude.
- `target_clearance_km` (optional, float, default 15.0): Desired safety threshold.
- `max_delta_v_m_s` (optional, float, default 5.0): Maximum allowable $\Delta v$ budget.
- `min_lead_time_minutes` (optional, float, default 15.0): Minimum operational reaction time.
- `lead_time_minutes` (optional, float): Explicit burn lead time before TCA.
- `spacecraft_area_min_m2`, `spacecraft_area_max_m2`, `spacecraft_mass_kg`, `spacecraft_drag_coeff` (optional): Spacecraft geometry parameters for attitude reorientation assessment.

**Response Schema:**
```json
{
  "event_id": "EVT_25544_99999_...",
  "primary_norad": 25544,
  "primary_name": "ISS",
  "secondary_norad": 99999,
  "secondary_name": "DEBRIS",
  "screening_epoch": "2025-01-01T00:00:00.000000",
  "baseline_miss_distance_km": 11.845,
  "baseline_tca_minutes": 161.736,
  "attitude_assessment": {
    "feasible": false,
    "status": "UNAVAILABLE_NO_GEOMETRY",
    "gap_documentation": "Spacecraft 3D geometry... absent in TLE dataset."
  },
  "candidates": [
    {
      "strategy_name": "Prograde Along Track (0.50 m/s)",
      "delta_v_m_s": 0.5,
      "after_miss_distance_km": 8253.01,
      "miss_distance_improvement_km": 8241.17,
      "clears_threshold": true,
      "is_feasible": true
    }
  ],
  "recommended_strategy": "prograde_along_track",
  "recommendation_rationale": "Selected Prograde Along Track (0.50 m/s)...",
  "unresolved_limitations": [...],
  "accuracy_disclaimer": "..."
}
```

### 7.2. GET /api/avoidance/evaluate
Evaluates a single specific maneuver direction (e.g. `prograde_along_track`, `retrograde_along_track`, `positive_cross_track`, `negative_cross_track`, `positive_radial`, `negative_radial`, or `custom`).

---

## 8. Full Test Suite Results (Executed 2026-10-10)

| Suite | File | Tests | Passed | Failed |
|---|---|---|---|---|
| Phase 3 Screening | `backend/tests/test_collision_screening.py` | 36 | 36 | 0 |
| Phase 4 Grid Analysis | `backend/tests/test_grid_analysis.py` | 30 | 30 | 0 |
| Phase 5 Collision Avoidance | `backend/tests/test_collision_avoidance.py` | 14 | 14 | 0 |
| **Total Project Suite** | | **80** | **80** | **0** |

**Execution time:** 0.63 seconds
**Success rate:** 100%

### Phase 5 Specific Tests Added:
1. `TestImpulsiveManeuvers`:
   - `test_prograde_along_track_improves_separation`: PASS
   - `test_retrograde_along_track_alters_separation`: PASS
   - `test_cross_track_maneuver_generates_out_of_plane_separation`: PASS
   - `test_radial_maneuver_modifies_eccentricity`: PASS
2. `TestAttitudeReorientation`:
   - `test_attitude_reorientation_gap_when_geometry_absent`: PASS
   - `test_attitude_reorientation_evaluates_differential_drag_when_geometry_provided`: PASS
3. `TestAvoidanceConstraintsAndFeasibility`:
   - `test_budget_exceeded_fails_feasibility`: PASS
   - `test_insufficient_lead_time_fails_feasibility`: PASS
4. `TestHybridAvoidancePlanning`:
   - `test_hybrid_plan_recommends_best_strategy`: PASS
   - `test_hybrid_plan_contains_required_fields_and_disclaimer`: PASS
5. `TestAvoidanceAPIEndpoints`:
   - `test_avoidance_plan_endpoint_schema`: PASS
   - `test_avoidance_plan_endpoint_safe_pair`: PASS
   - `test_avoidance_evaluate_endpoint`: PASS
   - `test_avoidance_endpoints_bad_params_return_400_or_404`: PASS

---

## 9. Assumptions and Remaining Unresolved Limitations

1. **No External Reference Document Provided:** All methods are derived from standard textbooks (Vallado 2013, Alfriend 2010); if a proprietary specification is subsequently supplied, the module must be reviewed against it.
2. **Impulsive Burn Idealization:** Burns are modeled as instantaneous velocity increments via Gauss's Variational Equations. Low-thrust continuous burns or finite-duration thrust arcs are not modeled.
3. **TLE SGP4 Error Floor:** Ephemeris errors (~100 m to 10+ km) dwarf fine trajectory perturbations. Results provide mathematical planning guidance, not confirmed operational safety margins.
4. **Covariance & Collision Probability ($P_c$):** Maneuver optimization maximizes geometric miss distance. Incorporating $P_c$ reduction requires full error covariance matrices.
5. **Attitude Reorientation Dependence:** Without explicit vehicle CAD geometry, attitude reorientation cannot be computed and is marked unavailable.

---

## 10. Conclusion

Phase 5 has successfully implemented **Hybrid Collision Avoidance** in Akash Setu with scientific integrity, rigorous astrodynamic formulation, explicit data gap documentation, zero regressions, and 100% passing test coverage (80/80 tests).
