"""
test_collision_avoidance.py  --  Akash Setu Phase 5: Hybrid Collision Avoidance Tests
=====================================================================================

Tests covering:
  1. Impulsive Delta-v maneuvers (prograde, retrograde, cross-track, radial).
  2. Attitude reorientation assessment (data gap reporting vs. differential drag).
  3. Feasibility constraints (delta-v budget, lead time notice, clearance threshold, perigee boundary).
  4. Hybrid avoidance plan generation and strategy recommendation.
  5. API endpoints: /api/avoidance/plan and /api/avoidance/evaluate.
"""

from __future__ import annotations

import math
import sys
from datetime import datetime, timezone
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

import pytest
from collision_screening import (
    ScreeningConfig, screen_satellites,
    _epoch_to_jd, _jd_add_minutes, _propagate_teme,
    _sgp4_epoch_days, _vec_norm, DEG2RAD, REVDAY2RADMIN,
)
from collision_avoidance import (
    AvoidanceConfig, AttitudeReorientationResult, ManeuverCandidateResult,
    HybridAvoidancePlan, ManeuveredSatrec,
    calculate_maneuvered_elements, evaluate_attitude_reorientation,
    evaluate_avoidance_candidate, generate_hybrid_avoidance_plan,
    serialise_avoidance_plan,
    STRATEGY_PROGRADE_ALONG_TRACK, STRATEGY_RETROGRADE_ALONG_TRACK,
    STRATEGY_POS_CROSS_TRACK, STRATEGY_NEG_CROSS_TRACK,
    STRATEGY_POS_RADIAL, STRATEGY_NEG_RADIAL, STRATEGY_ATTITUDE,
    WGS72_MU, WGS72_EARTH_RADIUS_KM,
)
from sgp4.api import Satrec, WGS72

EPOCH_STR = "2025-01-01T00:00:00.000000"
SCREENING_DT = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


def _internal_rec(norad: int, name: str, ma_deg: float = 0.0) -> dict:
    """Build an internal satellite record with pre-initialized Satrec."""
    sat = Satrec()
    sat.sgp4init(
        WGS72, "i", norad, _sgp4_epoch_days(EPOCH_STR),
        1e-5, 0.0, 0.0, 0.0001, 0.0, 51.6 * DEG2RAD,
        ma_deg * DEG2RAD, 15.5 * REVDAY2RADMIN, 0.0
    )
    return {
        "norad_id": norad, "name": name, "epoch": EPOCH_STR,
        "epoch_age_days": 1.0, "mean_motion": 15.5, "eccentricity": 0.0001,
        "inclination": 51.6, "period_min": 92.9, "source": "active_satellites",
        "priority": 1, "satrec": sat, "init_error": None, "stale": False,
        "intl_desig": "",
    }


class TestImpulsiveManeuvers:
    """Test 1: Impulsive Delta-v maneuvers and astrodynamics."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        recs = [_internal_rec(25544, "ISS", 0.0), _internal_rec(99999, "DEBRIS", 0.1)]
        p3_cfg = ScreeningConfig(
            horizon_minutes=170, coarse_step_minutes=0.5,
            screening_threshold_km=15.0, broad_phase_margin_km=200.0,
            refinement_steps=30,
        )
        p3_run = screen_satellites(recs, SCREENING_DT, p3_cfg)
        assert len(p3_run.alerts) >= 1, "Must generate encounter alert"
        self.alert = p3_run.alerts[0]
        self.recs = recs

    def test_prograde_along_track_improves_separation(self):
        """Prograde along-track burn alters semi-major axis and increases miss distance."""
        cfg = AvoidanceConfig(delta_v_m_s=0.5, target_clearance_km=15.0)
        cand = evaluate_avoidance_candidate(
            self.alert, self.recs[0], self.recs[1], SCREENING_DT,
            cfg, STRATEGY_PROGRADE_ALONG_TRACK, 0.5, 60.0
        )
        assert cand.semi_major_axis_change_m > 500.0, "0.5 m/s burn should increase sma by >500 m"
        assert cand.after_miss_distance_km > cand.before_miss_distance_km, "Miss distance must improve"
        assert cand.clears_threshold is True, "Maneuver should clear 15 km threshold"
        assert cand.is_feasible is True, "Candidate should be feasible"

    def test_retrograde_along_track_alters_separation(self):
        """Retrograde along-track burn depresses semi-major axis (da < 0) and alters separation."""
        cfg = AvoidanceConfig(delta_v_m_s=0.5, target_clearance_km=15.0)
        cand = evaluate_avoidance_candidate(
            self.alert, self.recs[0], self.recs[1], SCREENING_DT,
            cfg, STRATEGY_RETROGRADE_ALONG_TRACK, 0.5, 60.0
        )
        assert cand.semi_major_axis_change_m < -500.0, "Retrograde burn must decrease sma"
        assert cand.after_miss_distance_km > cand.before_miss_distance_km
        assert cand.is_feasible is True

    def test_cross_track_maneuver_generates_out_of_plane_separation(self):
        """Cross-track burn modifies inclination and RAAN without changing semi-major axis."""
        epoch_days = _sgp4_epoch_days(EPOCH_STR)
        sat_man, metrics = calculate_maneuvered_elements(
            self.recs[0]["satrec"], epoch_days, 60.0, 0.0, 0.0, 0.0005
        )
        assert abs(metrics["da_m"]) < 1e-6, "Cross-track burn does not alter semi-major axis"
        assert abs(metrics["di_deg"]) > 0.001 or abs(metrics["draan_deg"]) > 0.001

    def test_radial_maneuver_modifies_eccentricity(self):
        """Radial burn modifies eccentricity without altering semi-major axis."""
        epoch_days = _sgp4_epoch_days(EPOCH_STR)
        sat_man, metrics = calculate_maneuvered_elements(
            self.recs[0]["satrec"], epoch_days, 60.0, 0.0005, 0.0, 0.0
        )
        assert abs(metrics["da_m"]) < 1e-6, "Radial burn does not alter semi-major axis"
        assert abs(metrics["de"]) > 1e-6, "Radial burn alters eccentricity"


class TestAttitudeReorientation:
    """Test 2: Attitude reorientation and data gap documentation."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.primary = _internal_rec(25544, "ISS", 0.0)

    def test_attitude_reorientation_gap_when_geometry_absent(self):
        """When spacecraft geometry is absent, reports UNAVAILABLE_NO_GEOMETRY and gap notice."""
        res = evaluate_attitude_reorientation(self.primary, 100.0, 60.0, 15.0, geometry=None)
        assert res.feasible is False
        assert res.status == "UNAVAILABLE_NO_GEOMETRY"
        assert "Spacecraft 3D geometry" in res.gap_documentation
        assert "absent in the TLE dataset" in res.gap_documentation

    def test_attitude_reorientation_evaluates_differential_drag_when_geometry_provided(self):
        """When geometry is supplied, computes drag acceleration and along-track shift."""
        geom = {"area_min_m2": 5.0, "area_max_m2": 25.0, "mass_kg": 500.0, "drag_coeff": 2.2}
        res = evaluate_attitude_reorientation(self.primary, 100.0, 120.0, 0.05, geometry=geom)
        assert res.delta_area_m2 == 20.0
        assert res.delta_drag_acceleration_m_s2 > 0.0
        assert res.estimated_along_track_shift_km > 0.0
        assert res.status in ("EVALUATED_FEASIBLE", "EVALUATED_INFEASIBLE")


class TestAvoidanceConstraintsAndFeasibility:
    """Test 3: Avoidance feasibility constraints."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        recs = [_internal_rec(25544, "ISS", 0.0), _internal_rec(99999, "DEBRIS", 0.1)]
        p3_cfg = ScreeningConfig(
            horizon_minutes=170, coarse_step_minutes=0.5,
            screening_threshold_km=15.0, broad_phase_margin_km=200.0,
            refinement_steps=30,
        )
        p3_run = screen_satellites(recs, SCREENING_DT, p3_cfg)
        self.alert = p3_run.alerts[0]
        self.recs = recs

    def test_budget_exceeded_fails_feasibility(self):
        """Maneuver exceeding max_delta_v_m_s must fail feasibility."""
        cfg = AvoidanceConfig(delta_v_m_s=6.0, max_delta_v_m_s=5.0)
        cand = evaluate_avoidance_candidate(
            self.alert, self.recs[0], self.recs[1], SCREENING_DT,
            cfg, STRATEGY_PROGRADE_ALONG_TRACK, 6.0, 60.0
        )
        assert cand.is_feasible is False
        assert any("exceeds maximum budget" in r for r in cand.feasibility_reasons)

    def test_insufficient_lead_time_fails_feasibility(self):
        """Lead time below operational notice limit fails feasibility."""
        cfg = AvoidanceConfig(delta_v_m_s=0.5, min_lead_time_minutes=30.0)
        cand = evaluate_avoidance_candidate(
            self.alert, self.recs[0], self.recs[1], SCREENING_DT,
            cfg, STRATEGY_PROGRADE_ALONG_TRACK, 0.5, 10.0
        )
        assert cand.is_feasible is False
        assert any("below operational minimum notice" in r for r in cand.feasibility_reasons)


class TestHybridAvoidancePlanning:
    """Test 4: Hybrid plan generation and strategy ranking."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        recs = [_internal_rec(25544, "ISS", 0.0), _internal_rec(99999, "DEBRIS", 0.1)]
        p3_cfg = ScreeningConfig(
            horizon_minutes=170, coarse_step_minutes=0.5,
            screening_threshold_km=15.0, broad_phase_margin_km=200.0,
            refinement_steps=30,
        )
        p3_run = screen_satellites(recs, SCREENING_DT, p3_cfg)
        self.alert = p3_run.alerts[0]
        self.recs = recs

    def test_hybrid_plan_recommends_best_strategy(self):
        """Plan evaluates multiple candidates and recommends an optimal strategy."""
        plan = generate_hybrid_avoidance_plan(
            self.alert, self.recs[0], self.recs[1], SCREENING_DT
        )
        assert plan.recommended_strategy in (STRATEGY_PROGRADE_ALONG_TRACK, STRATEGY_RETROGRADE_ALONG_TRACK)
        assert plan.recommended_candidate is not None
        assert len(plan.candidates) >= 4
        assert plan.baseline_miss_distance_km == round(self.alert.miss_distance_km, 6)

    def test_hybrid_plan_contains_required_fields_and_disclaimer(self):
        """Plan includes limitations, attitude gap documentation, and disclaimer."""
        plan = generate_hybrid_avoidance_plan(
            self.alert, self.recs[0], self.recs[1], SCREENING_DT
        )
        assert len(plan.unresolved_limitations) >= 4
        assert "NOT constitute operational spacecraft flight commands" in plan.accuracy_disclaimer
        assert plan.attitude_assessment.status == "UNAVAILABLE_NO_GEOMETRY"

        # Check serialization
        data = serialise_avoidance_plan(plan)
        assert "candidates" in data
        assert "recommended_strategy" in data
        assert "accuracy_disclaimer" in data


class TestAvoidanceAPIEndpoints:
    """Test 5: Live Flask API integration for Phase 5."""

    @pytest.fixture(autouse=True)
    def setup_app(self):
        import app as fa
        fa.SATELLITES = {
            80: _internal_rec(80, "REG-A", ma_deg=0.0) | {"source": "stations", "priority": 10, "intl_desig": "", "period_min": 92.9, "inclination": 51.6},
            81: _internal_rec(81, "REG-B", ma_deg=0.1) | {"source": "active_satellites", "priority": 1, "intl_desig": "", "period_min": 92.9, "inclination": 51.6},
        }
        fa.app.config["TESTING"] = True
        self.client = fa.app.test_client()
        yield

    def test_avoidance_plan_endpoint_schema(self):
        """GET /api/avoidance/plan returns 200 and schema for encounter pair."""
        res = self.client.get("/api/avoidance/plan?norad_a=80&norad_b=81&threshold_km=50.0")
        assert res.status_code == 200
        data = res.get_json()
        assert "screening_epoch" in data
        assert "recommended_strategy" in data
        assert "candidates" in data
        assert "accuracy_disclaimer" in data

    def test_avoidance_plan_endpoint_safe_pair(self):
        """GET /api/avoidance/plan returns SAFE status when threshold is very small."""
        res = self.client.get("/api/avoidance/plan?norad_a=80&norad_b=81&threshold_km=0.001")
        assert res.status_code == 200
        data = res.get_json()
        assert data.get("status") == "SAFE"

    def test_avoidance_evaluate_endpoint(self):
        """GET /api/avoidance/evaluate returns candidate analysis."""
        res = self.client.get(
            "/api/avoidance/evaluate?norad_a=80&norad_b=81&direction=prograde_along_track&delta_v_m_s=0.5&threshold_km=50.0"
        )
        assert res.status_code == 200
        data = res.get_json()
        assert data["maneuver_direction"] == "prograde_along_track"
        assert "after_miss_distance_km" in data
        assert "is_feasible" in data

    def test_avoidance_endpoints_bad_params_return_400_or_404(self):
        """Invalid or missing parameters return 400 or 404."""
        # Missing NORAD
        res = self.client.get("/api/avoidance/plan?norad_a=99999999&norad_b=80")
        assert res.status_code == 404

        # Non-numeric NORAD
        res = self.client.get("/api/avoidance/plan?norad_a=abc&norad_b=80")
        assert res.status_code == 400

        # Missing direction on evaluate
        res = self.client.get("/api/avoidance/evaluate?norad_a=80&norad_b=81")
        assert res.status_code == 400

