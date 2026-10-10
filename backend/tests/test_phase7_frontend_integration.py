"""
Phase 7 Frontend 3D Conjunction & Avoidance Visualization Integration Tests
===========================================================================
Tests API endpoints, static assets serving, and end-to-end multi-phase workflows
linking Phase 3 screening, Phase 4 grid refinement, Phase 5 maneuver planning,
Phase 6 ML risk prediction, and Phase 7 3D maneuver trajectory generation.
"""

import math
import sys
from pathlib import Path
import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import app as fa


@pytest.fixture(scope="module")
def client():
    # Warm up catalogue
    fa.SATELLITES = fa._load_satellites()
    fa.app.config["TESTING"] = True
    with fa.app.test_client() as c:
        yield c


class TestPhase7TrajectoryEndpoint:
    """Tests for GET /api/avoidance/trajectory."""

    def test_trajectory_default_params(self, client):
        resp = client.get("/api/avoidance/trajectory?norad_id=25544")
        assert resp.status_code == 200
        data = resp.get_json()

        assert data["norad_id"] == 25544
        assert data["direction"] == "prograde"
        assert data["delta_v_m_s"] == 0.5
        assert "orbit_path" in data
        assert len(data["orbit_path"]) >= 100

        # Verify 3D ECEF coordinate validity (ISS radius ~ 6000-8000 km)
        for pt in data["orbit_path"][:10]:
            assert len(pt) == 3
            radius = math.sqrt(pt[0]**2 + pt[1]**2 + pt[2]**2)
            assert 6000.0 < radius < 8000.0, f"Unphysical orbit radius: {radius}"

        assert "orbital_period_change_s" in data
        assert "semi_major_axis_change_m" in data
        assert "disclaimer" in data
        assert "ESTIMATE ONLY" in data["disclaimer"]

    @pytest.mark.parametrize("direction", [
        "prograde", "retrograde", "radial_out", "radial_in", "cross_track_north", "cross_track_south"
    ])
    def test_trajectory_all_directions(self, client, direction):
        resp = client.get(f"/api/avoidance/trajectory?norad_id=25544&direction={direction}&delta_v_m_s=1.0&steps=60")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["direction"] == direction
        assert len(data["orbit_path"]) == 60

    def test_trajectory_invalid_norad(self, client):
        resp = client.get("/api/avoidance/trajectory?norad_id=999999")
        assert resp.status_code == 404
        data = resp.get_json()
        error_msg = (data.get("detail") or data.get("error") or "").lower()
        assert "not found" in error_msg

    def test_trajectory_invalid_direction(self, client):
        resp = client.get("/api/avoidance/trajectory?norad_id=25544&direction=diagonal")
        assert resp.status_code == 400
        data = resp.get_json()
        error_msg = (data.get("detail") or data.get("error") or "").lower()
        assert "direction" in error_msg

    def test_trajectory_invalid_deltav(self, client):
        resp = client.get("/api/avoidance/trajectory?norad_id=25544&delta_v_m_s=-5.0")
        assert resp.status_code == 400


class TestStaticAssetsServing:
    """Verifies that frontend static files are properly served by Flask."""

    def test_serve_index_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        html = resp.get_data(as_text=True)
        assert "<title>Akash Setu" in html
        assert 'id="globe-canvas"' in html
        assert 'id="tab-tracker"' in html
        assert 'id="tab-screening"' in html
        assert 'id="tab-avoidance"' in html
        assert 'id="tab-ml-risk"' in html
        assert 'id="detail-panel"' in html

    def test_serve_css(self, client):
        resp = client.get("/css/style.css")
        assert resp.status_code == 200
        assert "text/css" in resp.headers.get("Content-Type", "")
        css = resp.get_data(as_text=True)
        assert ".mode-nav" in css
        assert ".candidate-card" in css
        assert ".disclaimer-box" in css

    def test_serve_app_js(self, client):
        resp = client.get("/js/app.js")
        assert resp.status_code == 200
        assert "javascript" in resp.headers.get("Content-Type", "")
        js = resp.get_data(as_text=True)
        assert "renderManeuverTrajectory" in js
        assert "planAvoidanceManeuver" in js
        assert "runMLPrediction" in js
        assert "runGridRefinement" in js


class TestMultiphaseEndToEndWorkflow:
    """Verifies seamless multi-phase integration across Phases 3, 4, 5, 6, and 7."""

    def test_complete_encounter_workflow(self, client):
        # 1. Phase 3: Run screening
        screen_resp = client.get("/api/screen?threshold_km=150.0&horizon_hours=2.0")
        assert screen_resp.status_code == 200
        screen_data = screen_resp.get_json()
        assert "alerts" in screen_data

        # Pick two satellites from catalogue that exist
        sats = list(fa.SATELLITES.keys())
        assert len(sats) >= 2
        norad_a = sats[0]
        norad_b = sats[1]

        # 2. Phase 4: Run nested grid refinement on this pair
        refine_resp = client.get(f"/api/analyse/pair?norad_a={norad_a}&norad_b={norad_b}&window_minutes=15.0&initial_step_s=10.0")
        assert refine_resp.status_code == 200
        refine_data = refine_resp.get_json()
        assert "grid_analyses" in refine_data
        assert "accuracy_disclaimer" in refine_data or "stop_reasons_summary" in refine_data

        # 3. Phase 5: Plan avoidance maneuvers
        plan_resp = client.get(f"/api/avoidance/plan?norad_a={norad_a}&norad_b={norad_b}&miss_threshold_km=15.0&dv_mag=0.5")
        assert plan_resp.status_code == 200
        plan_data = plan_resp.get_json()
        candidates = plan_data.get("candidates") or plan_data.get("plan", {}).get("candidates", [])
        assert len(candidates) >= 4

        # 4. Phase 7: Fetch 3D trajectory for top candidate
        top_cand = candidates[0]
        cand_dir = top_cand.get("maneuver_direction") or top_cand.get("direction", "prograde")
        cand_dv = top_cand.get("delta_v_m_s", 0.5)
        traj_resp = client.get(
            f"/api/avoidance/trajectory?norad_id={norad_a}&direction={cand_dir}&delta_v_m_s={cand_dv}&steps=90"
        )
        assert traj_resp.status_code == 200
        traj_data = traj_resp.get_json()
        assert len(traj_data["orbit_path"]) == 90

        # 5. Phase 6: Predict ML risk using encounter features
        ml_resp = client.post("/api/v1/ml/predict_risk", json={
            "features": {
                "time_to_tca_hours": 1.5,
                "miss_distance_m": 420.0,
                "relative_speed_m_s": 14200.0,
                "secondary_object_type": "DEBRIS",
                "mahalanobis_distance": 2.1
            }
        })
        assert ml_resp.status_code == 200
        ml_data = ml_resp.get_json()
        assert "predicted_log10_risk" in ml_data or "prediction" in ml_data
        assert "is_high_risk" in ml_data or (ml_data.get("prediction") and "is_high_risk" in ml_data["prediction"])
        assert "scientific_caveat" in ml_data or "scientific_disclaimer" in ml_data

    def test_research_disclaimers_present_on_all_critical_outputs(self, client):
        # Trajectory endpoint
        r1 = client.get("/api/avoidance/trajectory?norad_id=25544").get_json()
        assert "disclaimer" in r1

        sats = list(fa.SATELLITES.keys())
        norad_a = sats[0]
        norad_b = sats[1]

        # Avoidance plan endpoint
        r2 = client.get(f"/api/avoidance/plan?norad_a={norad_a}&norad_b={norad_b}").get_json()
        assert "accuracy_disclaimer" in r2 or "disclaimer" in r2

        # ML Risk prediction endpoint
        r3 = client.post("/api/v1/ml/predict_risk", json={
            "features": {"miss_distance_m": 500.0}
        }).get_json()
        assert "scientific_caveat" in r3 or "scientific_disclaimer" in r3
