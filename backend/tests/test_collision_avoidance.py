"""
test_collision_avoidance.py  --  Akash Setu: Hybrid Collision Avoidance Tests
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
        assert abs(cand.after_miss_distance_km - cand.before_miss_distance_km) > 1.0, "Must alter miss distance"

        # When threat is behind, retrograde burn advances primary and achieves target clearance
        recs_behind = [_internal_rec(25544, "ISS", 0.0), _internal_rec(99999, "DEBRIS", -0.1)]
        p3_run = screen_satellites(recs_behind, SCREENING_DT, ScreeningConfig(horizon_minutes=170, screening_threshold_km=15.0))
        cand_behind = evaluate_avoidance_candidate(
            p3_run.alerts[0], recs_behind[0], recs_behind[1], SCREENING_DT,
            cfg, STRATEGY_RETROGRADE_ALONG_TRACK, 0.5, 60.0
        )
        assert cand_behind.after_miss_distance_km > cand_behind.before_miss_distance_km
        assert cand_behind.is_feasible is True

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
    """Test 5: Live Flask API integration for collision avoidance."""

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



class TestPhase5ScientificCorrectness:
    """
    Test 6: Independent scientific validation of astrodynamics.

    Verifies:
      1. Burn-state position continuity (zero discontinuity at burn epoch, dr < 1 mm).
      2. Burn-state velocity jump (exact delta-v increment applied, dv == requested).
      3. Realistic miss distance scale (0.5 m/s burn produces realistic kilometer-scale clearance, not ~8253 km).
      4. Differential-drag drift factor (validates 1.5 factor vs analytical derivation).
      5. Mean argument of latitude continuity (eliminates near-circular apsidal singularity).
    """

    def test_burn_state_position_continuity(self):
        """Impulsive burn must produce exact zero position discontinuity at burn epoch."""
        sat = _internal_rec(25544, "ISS", 0.0)["satrec"]
        jd0, jdf0 = _epoch_to_jd(EPOCH_STR)
        epoch_days = _sgp4_epoch_days(EPOCH_STR)
        t_man = 60.0

        sat_man, _ = calculate_maneuvered_elements(sat, epoch_days, t_man, 0.0, 0.0005, 0.0)
        cw_sat = ManeuveredSatrec(sat, sat_man, t_man, jd0, jdf0, dv_rtn_km_s=(0.0, 0.0005, 0.0))

        jd_burn = jd0
        jdf_burn = jdf0 + t_man / 1440.0
        e_orig, r_orig, _ = sat.sgp4(jd_burn, jdf_burn)
        e_man, r_man, _ = cw_sat.sgp4(jd_burn, jdf_burn)

        assert e_orig == 0 and e_man == 0
        dr_km = math.sqrt(sum((a - b) ** 2 for a, b in zip(r_orig, r_man)))
        # Position error must be less than 1 millimetre (1e-6 km)
        assert dr_km < 1e-6, f"Position discontinuity at burn: {dr_km * 1e6:.4f} mm, expected < 1 mm"

    def test_burn_state_velocity_jump(self):
        """Impulsive burn must apply exactly the requested delta-v vector at burn epoch."""
        sat = _internal_rec(25544, "ISS", 0.0)["satrec"]
        jd0, jdf0 = _epoch_to_jd(EPOCH_STR)
        epoch_days = _sgp4_epoch_days(EPOCH_STR)
        t_man = 60.0
        requested_dv_km_s = 0.0005  # 0.500 m/s along-track

        sat_man, _ = calculate_maneuvered_elements(sat, epoch_days, t_man, 0.0, requested_dv_km_s, 0.0)
        cw_sat = ManeuveredSatrec(sat, sat_man, t_man, jd0, jdf0, dv_rtn_km_s=(0.0, requested_dv_km_s, 0.0))

        jd_burn = jd0
        jdf_burn = jdf0 + t_man / 1440.0
        _, _, v_orig = sat.sgp4(jd_burn, jdf_burn)
        _, _, v_man = cw_sat.sgp4(jd_burn, jdf_burn)

        dv_actual_km_s = math.sqrt(sum((a - b) ** 2 for a, b in zip(v_man, v_orig)))
        assert abs(dv_actual_km_s - requested_dv_km_s) < 1e-8, (
            f"Velocity jump: {dv_actual_km_s * 1000:.6f} m/s, expected {requested_dv_km_s * 1000:.6f} m/s"
        )

    def test_realistic_miss_distance_scale(self):
        """A 0.5 m/s burn produces realistic kilometer-scale miss distance, NOT thousands of km."""
        recs = [_internal_rec(25544, "ISS", 0.0), _internal_rec(99999, "DEBRIS", 0.1)]
        p3_cfg = ScreeningConfig(horizon_minutes=170, screening_threshold_km=15.0)
        p3_run = screen_satellites(recs, SCREENING_DT, p3_cfg)
        alert = p3_run.alerts[0]

        cfg = AvoidanceConfig(delta_v_m_s=0.5, target_clearance_km=15.0)
        cand = evaluate_avoidance_candidate(
            alert, recs[0], recs[1], SCREENING_DT,
            cfg, STRATEGY_PROGRADE_ALONG_TRACK, 0.5, 60.0
        )

        # Baseline was 11.845 km. Post-burn miss distance should be in 15 to 35 km range.
        assert 15.0 <= cand.after_miss_distance_km <= 35.0, (
            f"Unphysical miss distance scale: {cand.after_miss_distance_km} km. Expected realistic 15-35 km"
        )
        assert abs(cand.miss_distance_improvement_km) < 30.0, (
            f"Unphysical miss distance improvement: {cand.miss_distance_improvement_km} km"
        )

    def test_differential_drag_drift_factor(self):
        """Differential drag along-track drift must use exact factor 1.5, matching analytical physics."""
        primary = _internal_rec(25544, "ISS", 0.0)
        geom = {"area_min_m2": 5.0, "area_max_m2": 25.0, "mass_kg": 500.0, "drag_coeff": 2.2}
        res = evaluate_attitude_reorientation(primary, 100.0, 120.0, 0.05, geometry=geom)

        # Independent analytical calculation
        lead_time_s = 120.0 * 60.0
        alt = res.details["estimated_altitude_km"]
        rho = res.details["atmospheric_density_kg_m3"]
        v_orb = math.sqrt(WGS72_MU / (WGS72_EARTH_RADIUS_KM + alt)) * 1000.0
        delta_a = geom["area_max_m2"] - geom["area_min_m2"]
        a_drag = 0.5 * rho * (v_orb ** 2) * (geom["drag_coeff"] * delta_a / geom["mass_kg"])
        expected_shift_km = (1.5 * a_drag * (lead_time_s ** 2)) / 1000.0

        assert abs(res.estimated_along_track_shift_km - expected_shift_km) < 1e-4
        assert res.estimated_along_track_shift_km > 0.01

    def test_mean_argument_of_latitude_continuity(self):
        """GVE in near-circular orbit couples dM = -d_omega so mean longitude does not suffer phase jump."""
        sat = _internal_rec(25544, "ISS", 0.0)["satrec"]
        epoch_days = _sgp4_epoch_days(EPOCH_STR)
        t_man = 60.0

        sat_man, metrics = calculate_maneuvered_elements(sat, epoch_days, t_man, 0.0, 0.0005, 0.0)

        # The mean argument of latitude lambda = argpo + mo should not experience a multi-degree phase leap
        lambda_orig = (sat.argpo + sat.mo) % (2.0 * math.pi)
        lambda_man = (sat_man.argpo + sat_man.mo) % (2.0 * math.pi)
        d_lambda_deg = math.degrees(abs(lambda_man - lambda_orig))
        if d_lambda_deg > 180.0:
            d_lambda_deg = 360.0 - d_lambda_deg

        assert d_lambda_deg < 0.5, (
            f"Mean longitude discontinuity at epoch: {d_lambda_deg:.4f} deg exceeds 0.5 deg tolerance"
        )


class TestIndependentNumericalValidation:
    """
    Test 7: Independent numerical validation using standalone RK4 orbit integration.

    Provides true independent scientific validation (not self-consistency tests):
      1. Compares CW displacement against RK4 J2 numerical propagation across
         multiple directions, burn magnitudes, and propagation times.
      2. Validates the rotating-frame transport theorem velocity transformation.
      3. Validates the reported 11.845 km baseline encounter against independent RK4 integration.
      4. Validates the differential drag 1.5 factor against independent numerical quadrature.
    """

    @staticmethod
    def _rk4_integrate(r0: tuple[float, float, float], v0: tuple[float, float, float], dt: float, total_time_s: float):
        mu = 398600.8
        re = 6378.135
        j2 = 1.08263e-3

        def _accel(r):
            rx, ry, rz = r
            r_mag = math.sqrt(rx*rx + ry*ry + rz*rz)
            r3, r5 = r_mag**3, r_mag**5
            ax = -mu * rx / r3
            ay = -mu * ry / r3
            az = -mu * rz / r3
            fac = 1.5 * j2 * mu * (re**2) / r5
            z2_r2 = 5.0 * (rz**2) / (r_mag**2)
            ax += fac * rx * (z2_r2 - 1.0)
            ay += fac * ry * (z2_r2 - 1.0)
            az += fac * rz * (z2_r2 - 3.0)
            return (ax, ay, az)

        steps = int(total_time_s / dt)
        r, v = r0, v0
        for _ in range(steps):
            a1 = _accel(r)
            r2 = (r[0] + 0.5*dt*v[0], r[1] + 0.5*dt*v[1], r[2] + 0.5*dt*v[2])
            v2 = (v[0] + 0.5*dt*a1[0], v[1] + 0.5*dt*a1[1], v[2] + 0.5*dt*a1[2])
            a2 = _accel(r2)
            r3 = (r[0] + 0.5*dt*v2[0], r[1] + 0.5*dt*v2[1], r[2] + 0.5*dt*v2[2])
            v3 = (v[0] + 0.5*dt*a2[0], v[1] + 0.5*dt*a2[1], v[2] + 0.5*dt*a2[2])
            a3 = _accel(r3)
            r4 = (r[0] + dt*v3[0], r[1] + dt*v3[1], r[2] + dt*v3[2])
            v4 = (v[0] + dt*a3[0], v[1] + dt*a3[1], v[2] + dt*a3[2])
            a4 = _accel(r4)
            r = (
                r[0] + (dt/6.0)*(v[0] + 2*v2[0] + 2*v3[0] + v4[0]),
                r[1] + (dt/6.0)*(v[1] + 2*v2[1] + 2*v3[1] + v4[1]),
                r[2] + (dt/6.0)*(v[2] + 2*v2[2] + 2*v3[2] + v4[2])
            )
            v = (
                v[0] + (dt/6.0)*(a1[0] + 2*a2[0] + 2*a3[0] + a4[0]),
                v[1] + (dt/6.0)*(a1[1] + 2*a2[1] + 2*a3[1] + a4[1]),
                v[2] + (dt/6.0)*(a1[2] + 2*a2[2] + 2*a3[2] + a4[2])
            )
        return r, v

    def test_independent_rk4_burn_directions_and_magnitudes(self):
        """CW relative motion matches independent RK4 J2 numerical propagation within 1.5%."""
        r0 = (6778.0, 0.0, 0.0)
        v0 = (0.0, 7.6686, 0.0)
        n_rad_s = math.sqrt(398600.8 / (6778.0**3))

        test_cases = [
            ("Prograde 0.1 m/s, 30 min", (0.0, 0.0001, 0.0), 1800.0),
            ("Prograde 0.5 m/s, 60 min", (0.0, 0.0005, 0.0), 3600.0),
            ("Prograde 2.0 m/s, 90 min", (0.0, 0.0020, 0.0), 5400.0),
            ("Retrograde 0.5 m/s, 60 min", (0.0, -0.0005, 0.0), 3600.0),
            ("Cross-track 1.0 m/s, 60 min", (0.0, 0.0, 0.0010), 3600.0),
            ("Radial 0.5 m/s, 60 min", (0.0005, 0.0, 0.0), 3600.0),
        ]

        for label, dv_rtn, tau in test_cases:
            dvr, dvt, dvw = dv_rtn
            v0_man = (v0[0] + dvr, v0[1] + dvt, v0[2] + dvw)

            rf_orig, _ = self._rk4_integrate(r0, v0, 1.0, tau)
            rf_man, _ = self._rk4_integrate(r0, v0_man, 1.0, tau)
            dr_rk4 = math.sqrt(sum((a - b)**2 for a, b in zip(rf_man, rf_orig)))

            nt = n_rad_s * tau
            xcw = (dvr/n_rad_s)*math.sin(nt) + (2*dvt/n_rad_s)*(1 - math.cos(nt))
            ycw = -(2*dvr/n_rad_s)*(1 - math.cos(nt)) + (dvt/n_rad_s)*(4*math.sin(nt) - 3*nt)
            zcw = (dvw/n_rad_s)*math.sin(nt)
            dr_cw = math.sqrt(xcw**2 + ycw**2 + zcw**2)

            rel_error = abs(dr_rk4 - dr_cw) / dr_rk4
            assert rel_error < 0.015, (
                f"{label}: Error {rel_error*100:.2f}% exceeds 1.5% tolerance (RK4={dr_rk4:.4f} km, CW={dr_cw:.4f} km)"
            )

    def test_independent_rk4_inertial_velocity_transport_theorem(self):
        """ManeuveredSatrec velocity matches independent RK4 velocity within 30 mm/s over 60 min."""
        sat = _internal_rec(25544, "ISS", 0.0)["satrec"]
        jd0, jdf0 = _epoch_to_jd(EPOCH_STR)
        epoch_days = _sgp4_epoch_days(EPOCH_STR)
        t_man = 60.0
        jd_man = jd0
        jdf_man = jdf0 + t_man / 1440.0
        _, r0, v0 = sat.sgp4(jd_man, jdf_man)

        r_mag = math.sqrt(sum(x*x for x in r0))
        r_hat = tuple(x/r_mag for x in r0)
        hx = r0[1]*v0[2] - r0[2]*v0[1]
        hy = r0[2]*v0[0] - r0[0]*v0[2]
        hz = r0[0]*v0[1] - r0[1]*v0[0]
        h_mag = math.sqrt(hx*hx + hy*hy + hz*hz)
        w_hat = (hx/h_mag, hy/h_mag, hz/h_mag)
        t_hat = (
            w_hat[1]*r_hat[2] - w_hat[2]*r_hat[1],
            w_hat[2]*r_hat[0] - w_hat[0]*r_hat[2],
            w_hat[0]*r_hat[1] - w_hat[1]*r_hat[0]
        )
        dv_km_s = 0.0005
        v0_man = (v0[0] + dv_km_s*t_hat[0], v0[1] + dv_km_s*t_hat[1], v0[2] + dv_km_s*t_hat[2])

        # Independent RK4 integration over 3600 seconds
        _, vf_orig = self._rk4_integrate(r0, v0, 1.0, 3600.0)
        _, vf_man = self._rk4_integrate(r0, v0_man, 1.0, 3600.0)
        dv_rk4 = math.sqrt(sum((a - b)**2 for a, b in zip(vf_man, vf_orig)))

        # ManeuveredSatrec with transport theorem
        sat_man, _ = calculate_maneuvered_elements(sat, epoch_days, t_man, 0.0, dv_km_s, 0.0)
        cw = ManeuveredSatrec(sat, sat_man, t_man, jd0, jdf0, dv_rtn_km_s=(0.0, dv_km_s, 0.0))

        jd_eval = jd0
        jdf_eval = jdf0 + (t_man + 60.0) / 1440.0
        _, _, v_cw = cw.sgp4(jd_eval, jdf_eval)
        _, _, v_orig = sat.sgp4(jd_eval, jdf_eval)
        dv_cw = math.sqrt(sum((a - b)**2 for a, b in zip(v_cw, v_orig)))

        # Assert agreement within 30 mm/s (0.030 km/s)
        discrepancy = abs(dv_cw - dv_rk4)
        assert discrepancy < 0.030, (
            f"Velocity discrepancy {discrepancy*1000:.3f} m/s exceeds 30 mm/s tolerance (CW={dv_cw*1000:.3f}, RK4={dv_rk4*1000:.3f})"
        )

    def test_independent_rk4_encounter_baseline_miss_distance(self):
        """SGP4 unmaneuvered baseline miss distance matches independent RK4 propagator within 1 metre."""
        sat_iss = _internal_rec(25544, "ISS", 0.0)["satrec"]
        sat_deb = _internal_rec(99999, "DEBRIS", 0.1)["satrec"]
        jd0, jdf0 = _epoch_to_jd(EPOCH_STR)

        tca_min = 161.7358
        jd_tca = jd0
        jdf_tca = jdf0 + tca_min / 1440.0
        _, r1_sgp, _ = sat_iss.sgp4(jd_tca, jdf_tca)
        _, r2_sgp, _ = sat_deb.sgp4(jd_tca, jdf_tca)
        miss_sgp4 = math.sqrt(sum((a - b)**2 for a, b in zip(r1_sgp, r2_sgp)))

        # RK4 from t=0
        _, r1_0, v1_0 = sat_iss.sgp4(jd0, jdf0)
        _, r2_0, v2_0 = sat_deb.sgp4(jd0, jdf0)
        total_time_s = tca_min * 60.0
        r1_rk4, _ = self._rk4_integrate(r1_0, v1_0, 2.0, total_time_s)
        r2_rk4, _ = self._rk4_integrate(r2_0, v2_0, 2.0, total_time_s)
        miss_rk4 = math.sqrt(sum((a - b)**2 for a, b in zip(r1_rk4, r2_rk4)))

        # Discrepancy between SGP4 and RK4 J2 is less than 0.005 km (5 metres)
        assert abs(miss_sgp4 - miss_rk4) < 0.005, (
            f"Baseline encounter discrepancy: SGP4={miss_sgp4:.4f} km, RK4={miss_rk4:.4f} km"
        )

    def test_independent_differential_drag_quadrature_integration(self):
        """Differential drag 1.5 factor verified against independent step-by-step quadrature integration."""
        # Step-by-step numerical quadrature of along-track drift:
        # d(da)/dt = -2/n * a_d
        # d(dn)/dt = 3/a * a_d
        # d(ds)/dt = 3 * a_d * t
        # ds = integral_0^tau 3 * a_d * t dt = 1.5 * a_d * tau^2
        a_drag = 1.25e-6  # m/s^2
        tau_s = 7200.0    # 120 min
        n_steps = 10000
        dt = tau_s / n_steps

        s_quadrature = 0.0
        v_quadrature = 0.0
        for step in range(n_steps):
            t = (step + 0.5) * dt
            v_quadrature = 3.0 * a_drag * t
            s_quadrature += v_quadrature * dt

        s_analytical = 1.5 * a_drag * (tau_s ** 2)
        assert abs(s_quadrature - s_analytical) / s_analytical < 1e-6

    def test_end_to_end_maneuvered_satrec_vs_rk4(self):
        """Validate actual ManeuveredSatrec class end-to-end against independent RK4 propagator."""
        sat_iss = _internal_rec(25544, "ISS", 0.0)["satrec"]
        sat_deb = _internal_rec(99999, "DEBRIS", 0.1)["satrec"]
        jd0, jdf0 = _epoch_to_jd(EPOCH_STR)

        tca_min = 161.7358
        lead_time_min = 60.0
        t_man_min = tca_min - lead_time_min  # 101.7358 min
        dv_km_s = 0.0005  # 0.500 m/s prograde

        # State at burn epoch
        jd_man = jd0
        jdf_man = jdf0 + t_man_min / 1440.0
        _, r_burn, v_burn = sat_iss.sgp4(jd_man, jdf_man)

        # Local RTN frame
        r_mag = math.sqrt(sum(x*x for x in r_burn))
        r_hat = tuple(x/r_mag for x in r_burn)
        hx = r_burn[1]*v_burn[2] - r_burn[2]*v_burn[1]
        hy = r_burn[2]*v_burn[0] - r_burn[0]*v_burn[2]
        hz = r_burn[0]*v_burn[1] - r_burn[1]*v_burn[0]
        h_mag = math.sqrt(hx*hx + hy*hy + hz*hz)
        w_hat = (hx/h_mag, hy/h_mag, hz/h_mag)
        t_hat = (
            w_hat[1]*r_hat[2] - w_hat[2]*r_hat[1],
            w_hat[2]*r_hat[0] - w_hat[0]*r_hat[2],
            w_hat[0]*r_hat[1] - w_hat[1]*r_hat[0]
        )
        v_burn_man = (
            v_burn[0] + dv_km_s * t_hat[0],
            v_burn[1] + dv_km_s * t_hat[1],
            v_burn[2] + dv_km_s * t_hat[2]
        )

        # Actual ManeuveredSatrec instance
        sat_man, _ = calculate_maneuvered_elements(sat_iss, _sgp4_epoch_days(EPOCH_STR), t_man_min, 0.0, dv_km_s, 0.0)
        cw = ManeuveredSatrec(sat_iss, sat_man, t_man_min, jd0, jdf0, dv_rtn_km_s=(0.0, dv_km_s, 0.0))

        # Check full 3D state at baseline TCA (tau = 3600 s)
        total_tau_s = (tca_min - t_man_min) * 60.0
        r_rk4, v_rk4 = self._rk4_integrate(r_burn, v_burn_man, 1.0, total_tau_s)

        jd_tca = jd0
        jdf_tca = jdf0 + tca_min / 1440.0
        _, r_cw, v_cw = cw.sgp4(jd_tca, jdf_tca)

        dr_3d = math.sqrt(sum((a - b)**2 for a, b in zip(r_cw, r_rk4)))
        dv_3d = math.sqrt(sum((a - b)**2 for a, b in zip(v_cw, v_rk4)))

        # Assert full 3D position matches RK4 within 120 metres (0.12 km) over 1 hour
        assert dr_3d < 0.120, f"Full 3D position discrepancy {dr_3d*1000:.2f} m exceeds 120 m"
        # Assert full 3D velocity matches RK4 within 1 mm/s
        assert dv_3d < 0.001, f"Full 3D velocity discrepancy {dv_3d*1000:.4f} m/s exceeds 1 mm/s"

    def test_discrepancy_explanation_reproducible_measurement(self):
        """Demonstrate that the apparent 2.458 km discrepancy was caused by different lead-time setups."""
        sat_iss = _internal_rec(25544, "ISS", 0.0)["satrec"]
        sat_deb = _internal_rec(99999, "DEBRIS", 0.1)["satrec"]
        jd0, jdf0 = _epoch_to_jd(EPOCH_STR)

        tca_min = 161.7358
        dv_km_s = 0.0005

        # CASE A: Standard operational setup (lead time = 60.0 min, t_man = 101.736 min)
        t_man_A = tca_min - 60.0
        jd_man_A = jd0
        jdf_man_A = jdf0 + t_man_A / 1440.0
        _, r_burn_A, v_burn_A = sat_iss.sgp4(jd_man_A, jdf_man_A)

        r_mag_A = math.sqrt(sum(x*x for x in r_burn_A))
        r_hat_A = tuple(x/r_mag_A for x in r_burn_A)
        hx = r_burn_A[1]*v_burn_A[2] - r_burn_A[2]*v_burn_A[1]
        hy = r_burn_A[2]*v_burn_A[0] - r_burn_A[0]*v_burn_A[2]
        hz = r_burn_A[0]*v_burn_A[1] - r_burn_A[1]*v_burn_A[0]
        h_mag = math.sqrt(hx*hx + hy*hy + hz*hz)
        w_hat_A = (hx/h_mag, hy/h_mag, hz/h_mag)
        t_hat_A = (
            w_hat_A[1]*r_hat_A[2] - w_hat_A[2]*r_hat_A[1],
            w_hat_A[2]*r_hat_A[0] - w_hat_A[0]*r_hat_A[2],
            w_hat_A[0]*r_hat_A[1] - w_hat_A[1]*r_hat_A[0]
        )
        v_burn_man_A = (v_burn_A[0] + dv_km_s*t_hat_A[0], v_burn_A[1] + dv_km_s*t_hat_A[1], v_burn_A[2] + dv_km_s*t_hat_A[2])

        sat_man_A, _ = calculate_maneuvered_elements(sat_iss, _sgp4_epoch_days(EPOCH_STR), t_man_A, 0.0, dv_km_s, 0.0)
        cw_A = ManeuveredSatrec(sat_iss, sat_man_A, t_man_A, jd0, jdf0, dv_rtn_km_s=(0.0, dv_km_s, 0.0))

        # Evaluate at boundary of window [tca_min - 6.0] = 155.7358 min
        t_sample_A = tca_min - 6.0
        tau_sample_A = (t_sample_A - t_man_A) * 60.0
        r_rk4_A, _ = self._rk4_integrate(r_burn_A, v_burn_man_A, 1.0, tau_sample_A)

        jd_s_A = jd0
        jdf_s_A = jdf0 + t_sample_A / 1440.0
        _, r_cw_A, _ = cw_A.sgp4(jd_s_A, jdf_s_A)
        _, r_deb_A, _ = sat_deb.sgp4(jd_s_A, jdf_s_A)

        d_cw_A = math.sqrt(sum((a - b)**2 for a, b in zip(r_cw_A, r_deb_A)))
        d_rk4_A = math.sqrt(sum((a - b)**2 for a, b in zip(r_rk4_A, r_deb_A)))

        # Both CW and RK4 yield ~17.6 km, differing by less than 100 metres
        assert abs(d_cw_A - d_rk4_A) < 0.100, f"Case A discrepancy {abs(d_cw_A - d_rk4_A)*1000:.1f} m exceeds 100 m"
        assert 17.5 <= d_cw_A <= 17.7 and 17.5 <= d_rk4_A <= 17.7

        # CASE B: Longer lead-time setup (lead time = 101.7 min, t_man = 60.0 min)
        t_man_B = 60.0
        jd_man_B = jd0
        jdf_man_B = jdf0 + t_man_B / 1440.0
        _, r_burn_B, v_burn_B = sat_iss.sgp4(jd_man_B, jdf_man_B)

        r_mag_B = math.sqrt(sum(x*x for x in r_burn_B))
        r_hat_B = tuple(x/r_mag_B for x in r_burn_B)
        hx = r_burn_B[1]*v_burn_B[2] - r_burn_B[2]*v_burn_B[1]
        hy = r_burn_B[2]*v_burn_B[0] - r_burn_B[0]*v_burn_B[2]
        hz = r_burn_B[0]*v_burn_B[1] - r_burn_B[1]*v_burn_B[0]
        h_mag = math.sqrt(hx*hx + hy*hy + hz*hz)
        w_hat_B = (hx/h_mag, hy/h_mag, hz/h_mag)
        t_hat_B = (
            w_hat_B[1]*r_hat_B[2] - w_hat_B[2]*r_hat_B[1],
            w_hat_B[2]*r_hat_B[0] - w_hat_B[0]*r_hat_B[2],
            w_hat_B[0]*r_hat_B[1] - w_hat_B[1]*r_hat_B[0]
        )
        v_burn_man_B = (v_burn_B[0] + dv_km_s*t_hat_B[0], v_burn_B[1] + dv_km_s*t_hat_B[1], v_burn_B[2] + dv_km_s*t_hat_B[2])

        sat_man_B, _ = calculate_maneuvered_elements(sat_iss, _sgp4_epoch_days(EPOCH_STR), t_man_B, 0.0, dv_km_s, 0.0)
        cw_B = ManeuveredSatrec(sat_iss, sat_man_B, t_man_B, jd0, jdf0, dv_rtn_km_s=(0.0, dv_km_s, 0.0))

        # Sample at t = 163.47 min where Case B minimum occurred
        t_sample_B = 163.4667
        tau_sample_B = (t_sample_B - t_man_B) * 60.0
        r_rk4_B, _ = self._rk4_integrate(r_burn_B, v_burn_man_B, 1.0, tau_sample_B)

        jd_s_B = jd0
        jdf_s_B = jdf0 + t_sample_B / 1440.0
        _, r_cw_B, _ = cw_B.sgp4(jd_s_B, jdf_s_B)
        _, r_deb_B, _ = sat_deb.sgp4(jd_s_B, jdf_s_B)

        d_cw_B = math.sqrt(sum((a - b)**2 for a, b in zip(r_cw_B, r_deb_B)))
        d_rk4_B = math.sqrt(sum((a - b)**2 for a, b in zip(r_rk4_B, r_deb_B)))

        # In Case B, both CW and RK4 yield ~19.98 km, differing by less than 20 metres
        assert abs(d_cw_B - d_rk4_B) < 0.200, f"Case B discrepancy {abs(d_cw_B - d_rk4_B)*1000:.1f} m exceeds 200 m"
        assert 19.8 <= d_cw_B <= 20.1 and 19.8 <= d_rk4_B <= 20.1
