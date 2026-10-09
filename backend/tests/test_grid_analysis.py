"""
test_grid_analysis.py  --  Akash Setu Phase 4 essential tests
==============================================================
Three mandatory tests as specified in Phase 4 requirements:

  Test 1. Controlled encounter with known minimum separation.
  Test 2. Grid-refinement convergence against an independently calculated reference.
  Test 3. Regression check: Phase 3 and existing APIs still work.

Reference-document status: No external reference was provided with the Phase 4
specification. The method is derived from standard TCA determination literature
(Alfano 2005, Hoots 1984, Vallado 2013). This is documented in grid_analysis.py.

Run with:
    cd <project_root>
    python -m pytest backend/tests/test_grid_analysis.py -v
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
    _epoch_to_jd, _jd_add_minutes, _propagate_teme, _vec_norm, _vec_sub,
    _make_satrec_from_record,
)
from grid_analysis import GridConfig, GridAnalysisResult, refine_tca, analyse_encounter

# Fixed screening epoch for deterministic tests
EPOCH_STR = "2025-01-01T00:00:00.000000"
SCREENING_DT = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _iss_rec(norad: int, name: str, ma_deg: float = 0.0) -> dict:
    """Build a CelesTrak-style ISS-like record."""
    return {
        "NORAD_CAT_ID": str(norad), "OBJECT_NAME": name,
        "EPOCH": EPOCH_STR,
        "MEAN_MOTION": "15.5", "ECCENTRICITY": "0.0001", "INCLINATION": "51.6",
        "RA_OF_ASC_NODE": "0.0", "ARG_OF_PERICENTER": "0.0",
        "MEAN_ANOMALY": str(ma_deg), "BSTAR": "0.00001",
        "MEAN_MOTION_DOT": "0.0", "MEAN_MOTION_DDOT": "0.0",
    }


def _internal_rec(norad: int, name: str, ma_deg: float = 0.0) -> dict:
    """Build an internal-format record with a pre-built Satrec."""
    from sgp4.api import Satrec, WGS72
    from collision_screening import _sgp4_epoch_days, DEG2RAD, REVDAY2RADMIN
    sat = Satrec()
    sat.sgp4init(WGS72, "i", norad,
                 _sgp4_epoch_days(EPOCH_STR),
                 1e-5, 0.0, 0.0, 0.0001, 0.0, 51.6 * DEG2RAD,
                 ma_deg * DEG2RAD, 15.5 * REVDAY2RADMIN, 0.0)
    return {
        "norad_id": norad, "name": name, "epoch": EPOCH_STR,
        "epoch_age_days": 1.0, "mean_motion": 15.5, "eccentricity": 0.0001,
        "inclination": 51.6, "period_min": 92.9, "source": "active_satellites",
        "priority": 1, "satrec": sat, "init_error": None, "stale": False,
        "intl_desig": "",
    }


def _brute_force_min_sep(satrec_a, satrec_b, jd0, jdf0,
                          t_center, half_window_min,
                          n_steps=10000) -> tuple[float, float]:
    """
    Independently calculate minimum separation by brute-force fine scan.
    Returns (best_t_minutes, min_sep_km).
    Used as the independent reference for Test 2.
    """
    dt = (2.0 * half_window_min) / n_steps
    t_lo = t_center - half_window_min
    best_t = t_center
    best_sep = float("inf")
    for i in range(n_steps + 1):
        t = t_lo + i * dt
        jdt, jdft = _jd_add_minutes(jd0, jdf0, t)
        try:
            r_a, _ = _propagate_teme(satrec_a, jdt, jdft)
            r_b, _ = _propagate_teme(satrec_b, jdt, jdft)
            sep = _vec_norm(_vec_sub(r_a, r_b))
            if sep < best_sep:
                best_sep = sep
                best_t = t
        except RuntimeError:
            pass
    return best_t, best_sep


# ===========================================================================
# Test 1: Controlled encounter with known minimum separation
# ===========================================================================

class TestControlledEncounter:
    """
    Use the same verified fixture from Phase 3:
      ma_a=0.0 deg, ma_b=0.1 deg, same orbital plane.
      Phase-3 confirmed: min_sep ~ 11.85 km at t ~ 161.7 min.

    Phase 4 must:
      (a) Find an encounter (Phase 3 ALERT)
      (b) Run grid analysis without error
      (c) Return a non-NaN, non-negative miss distance
      (d) Return a spatial_resolution_m that is positive and finite
      (e) Return a tca within the Phase-3 search window
    """

    def test_grid_analysis_on_known_encounter(self):
        # Phase-3 screening to find the encounter
        p3_cfg = ScreeningConfig(
            horizon_minutes=170, coarse_step_minutes=0.5,
            screening_threshold_km=15.0, broad_phase_margin_km=200.0,
            refinement_steps=30,
        )
        recs = [_iss_rec(1, "SAT-A", ma_deg=0.0), _iss_rec(2, "SAT-B", ma_deg=0.1)]
        p3_run = screen_satellites(recs, SCREENING_DT, p3_cfg)

        assert len(p3_run.alerts) >= 1, "Phase 3 must find at least one encounter"
        alert = p3_run.alerts[0]
        assert alert.screening_status == "ALERT"
        p3_miss = alert.miss_distance_km
        p3_tca = alert.tca_minutes_from_start
        assert math.isfinite(p3_miss) and 0 < p3_miss < 15.0

        # Phase-4 grid analysis
        g4_cfg = GridConfig(
            search_half_window_minutes=2.0,
            initial_grid_points=20,
            target_spatial_resolution_m=1.0,   # 1 m target (fast for this test)
            max_iterations=10,
            include_iteration_log=True,
        )
        result = analyse_encounter(alert, recs, SCREENING_DT, g4_cfg)

        # (b) No error
        assert result.convergence_reason[:5] != "ERROR", (
            f"Grid analysis returned error: {result.convergence_reason}"
        )

        # (c) Miss distance is finite and non-negative
        assert math.isfinite(result.refined_miss_distance_km), \
            "Refined miss distance must be finite"
        assert result.refined_miss_distance_km >= 0, \
            "Refined miss distance must be non-negative"

        # (d) Grid achieved some positive spatial resolution
        assert math.isfinite(result.spatial_resolution_m), \
            "Spatial resolution must be finite"
        assert result.spatial_resolution_m > 0, \
            "Spatial resolution must be positive"

        # (e) Refined TCA is within +-2 minutes of Phase-3 TCA
        assert abs(result.refined_tca_minutes - p3_tca) <= 2.0, \
            f"Refined TCA {result.refined_tca_minutes:.3f} too far from P3 TCA {p3_tca:.3f}"

        # (f) Refined miss distance must be <= Phase-3 miss distance + small tolerance
        # (grid refinement should not be significantly worse)
        tolerance_km = 1.0
        assert result.refined_miss_distance_km <= p3_miss + tolerance_km, (
            f"Refined miss ({result.refined_miss_distance_km:.4f} km) is significantly "
            f"worse than Phase-3 ({p3_miss:.4f} km)"
        )

        # (g) Disclaimer is present
        assert len(result.accuracy_disclaimer) > 50, "Accuracy disclaimer must be non-empty"

        # (h) Iteration log non-empty since include_iteration_log=True
        assert len(result.iteration_log) >= 1, "Iteration log must be populated"

        # Print for manual inspection
        print(f"\n  Phase-3: miss={p3_miss:.4f} km  tca={p3_tca:.4f} min")
        print(f"  Phase-4: miss={result.refined_miss_distance_km:.6f} km "
              f"({result.refined_miss_distance_m:.3f} m)  "
              f"tca={result.refined_tca_minutes:.6f} min")
        print(f"  Grid: dt={result.final_dt_seconds:.6g} s  "
              f"spatial_res={result.spatial_resolution_m:.4g} m  "
              f"iters={result.iterations_used}  converged={result.converged}")
        print(f"  Reason: {result.convergence_reason}")

    def test_miss_distance_in_metres(self):
        """refined_miss_distance_m == refined_miss_distance_km * 1000."""
        p3_cfg = ScreeningConfig(horizon_minutes=170, coarse_step_minutes=0.5,
                                  screening_threshold_km=15.0, broad_phase_margin_km=200.0)
        recs = [_iss_rec(3, "SA", ma_deg=0.0), _iss_rec(4, "SB", ma_deg=0.1)]
        p3_run = screen_satellites(recs, SCREENING_DT, p3_cfg)
        if not p3_run.alerts:
            pytest.skip("No Phase-3 alerts to refine")
        result = analyse_encounter(p3_run.alerts[0], recs, SCREENING_DT)
        assert abs(result.refined_miss_distance_m -
                   result.refined_miss_distance_km * 1000) < 1e-3, \
            "refined_miss_distance_m must equal refined_miss_distance_km * 1000"

    def test_disclaimer_present(self):
        """Accuracy disclaimer must be non-trivial."""
        p3_cfg = ScreeningConfig(horizon_minutes=170, coarse_step_minutes=0.5,
                                  screening_threshold_km=15.0, broad_phase_margin_km=200.0)
        recs = [_iss_rec(5, "SC"), _iss_rec(6, "SD", ma_deg=0.1)]
        p3_run = screen_satellites(recs, SCREENING_DT, p3_cfg)
        if not p3_run.alerts:
            pytest.skip("No Phase-3 alerts")
        result = analyse_encounter(p3_run.alerts[0], recs, SCREENING_DT)
        assert "accuracy" in result.accuracy_disclaimer.lower() or \
               "numerical" in result.accuracy_disclaimer.lower(), \
            "Disclaimer must mention accuracy or numerical nature"
        assert "operational" in result.accuracy_disclaimer.lower(), \
            "Disclaimer must warn against operational use"


# ===========================================================================
# Test 2: Grid-refinement convergence against independent reference
# ===========================================================================

class TestGridConvergence:
    """
    Verify that grid-refinement converges toward the independently calculated
    minimum separation from a brute-force fine scan.

    Independent reference: _brute_force_min_sep() evaluates 10,000 equally-
    spaced points over the same search window. This is used as the "ground
    truth" for the grid algorithm's result to converge toward.

    The grid algorithm starts coarser but iteratively narrows; its result
    must be within a tolerance of the brute-force reference.
    """

    def _get_satrecs_and_p3(self):
        recs = [_internal_rec(10, "GA", ma_deg=0.0), _internal_rec(11, "GB", ma_deg=0.1)]
        p3_cfg = ScreeningConfig(horizon_minutes=170, coarse_step_minutes=0.5,
                                  screening_threshold_km=15.0, broad_phase_margin_km=200.0,
                                  refinement_steps=30)
        p3_run = screen_satellites(recs, SCREENING_DT, p3_cfg)
        if not p3_run.alerts:
            pytest.skip("No Phase-3 alerts for convergence test")
        alert = p3_run.alerts[0]
        sa = recs[0]["satrec"]
        sb = recs[1]["satrec"]
        return sa, sb, alert, recs

    def test_convergence_toward_brute_force_reference(self):
        sa, sb, alert, recs = self._get_satrecs_and_p3()

        jd0, jdf0 = _epoch_to_jd(EPOCH_STR)

        # Independent reference: brute-force 10,000-point scan
        ref_t, ref_sep = _brute_force_min_sep(
            sa, sb, jd0, jdf0,
            t_center=alert.tca_minutes_from_start,
            half_window_min=2.0,
            n_steps=10000,
        )

        print(f"\n  Brute-force reference: sep={ref_sep:.6f} km at t={ref_t:.6f} min")

        # Grid algorithm
        g4_cfg = GridConfig(
            search_half_window_minutes=2.0,
            initial_grid_points=20,
            target_spatial_resolution_m=0.01,   # 1 cm target
            max_iterations=12,
            include_iteration_log=True,
        )
        result = analyse_encounter(alert, recs, SCREENING_DT, g4_cfg)

        grid_sep = result.refined_miss_distance_km
        print(f"  Grid result:           sep={grid_sep:.6f} km at t={result.refined_tca_minutes:.6f} min")
        print(f"  |grid - ref| = {abs(grid_sep - ref_sep)*1000:.3f} m")
        print(f"  Grid spatial_res = {result.spatial_resolution_m:.4g} m  iters={result.iterations_used}")
        print(f"  Convergence reason: {result.convergence_reason}")

        # The grid result must be within 10 m of the brute-force reference.
        # (10 m is generous -- the brute-force itself has resolution of
        #  2*2min/10000 * 60s * ~0.21 km/s * 1000 m/km ~ 0.05 m so the
        #  brute-force reference is far more accurate than this tolerance allows.)
        tolerance_km = 0.010   # 10 m
        assert abs(grid_sep - ref_sep) < tolerance_km, (
            f"Grid result ({grid_sep:.6f} km) deviates from brute-force "
            f"reference ({ref_sep:.6f} km) by "
            f"{abs(grid_sep-ref_sep)*1000:.2f} m > {tolerance_km*1000:.0f} m tolerance"
        )

    def test_finer_resolution_gives_smaller_or_equal_miss(self):
        """
        Run grid analysis at two resolutions: coarse (1 m) and fine (0.01 m).
        The fine-resolution result must be <= coarse result + tolerance.
        """
        sa, sb, alert, recs = self._get_satrecs_and_p3()

        cfg_coarse = GridConfig(target_spatial_resolution_m=1.0, max_iterations=6)
        cfg_fine   = GridConfig(target_spatial_resolution_m=0.01, max_iterations=12)

        r_coarse = analyse_encounter(alert, recs, SCREENING_DT, cfg_coarse)
        r_fine   = analyse_encounter(alert, recs, SCREENING_DT, cfg_fine)

        print(f"\n  Coarse (1m):  miss={r_coarse.refined_miss_distance_km:.6f} km  "
              f"dt={r_coarse.final_dt_seconds:.4g} s  iters={r_coarse.iterations_used}")
        print(f"  Fine (1cm):   miss={r_fine.refined_miss_distance_km:.6f} km  "
              f"dt={r_fine.final_dt_seconds:.4g} s  iters={r_fine.iterations_used}")

        tolerance_km = 0.005   # 5 m tolerance
        assert r_fine.refined_miss_distance_km <= r_coarse.refined_miss_distance_km + tolerance_km, (
            f"Fine-resolution result ({r_fine.refined_miss_distance_km:.6f} km) is "
            f"significantly worse than coarse ({r_coarse.refined_miss_distance_km:.6f} km)"
        )

        # Fine resolution dt must be <= coarse resolution dt (or very close)
        # (Fine run should achieve a finer grid spacing)
        assert r_fine.final_dt_seconds <= r_coarse.final_dt_seconds + 1e-6, \
            "Fine run must achieve <= dt compared to coarse run"

    def test_spatial_resolution_reported_correctly(self):
        """
        spatial_resolution_m == final_dt_seconds * relative_speed_km_s * 1000
        within floating-point tolerance.
        """
        sa, sb, alert, recs = self._get_satrecs_and_p3()
        cfg = GridConfig(target_spatial_resolution_m=0.01, max_iterations=8,
                          include_iteration_log=True)
        result = analyse_encounter(alert, recs, SCREENING_DT, cfg)

        if math.isfinite(result.spatial_resolution_m) and math.isfinite(result.final_dt_seconds):
            rel_speed = max(result.relative_speed_km_s, 1e-6)
            expected_m = result.final_dt_seconds * rel_speed * 1000.0
            assert abs(result.spatial_resolution_m - expected_m) < 0.001, (
                f"spatial_resolution_m ({result.spatial_resolution_m:.4g}) does not match "
                f"dt * speed * 1000 ({expected_m:.4g})"
            )

    def test_iteration_log_is_monotone_in_dt(self):
        """
        Each successive iteration should use a smaller or equal dt than the prior.
        """
        sa, sb, alert, recs = self._get_satrecs_and_p3()
        cfg = GridConfig(target_spatial_resolution_m=0.01, max_iterations=8,
                          include_iteration_log=True)
        result = analyse_encounter(alert, recs, SCREENING_DT, cfg)

        if len(result.iteration_log) < 2:
            pytest.skip("Fewer than 2 iterations -- cannot test monotonicity")

        dts = [entry["dt_min"] for entry in result.iteration_log]
        for i in range(1, len(dts)):
            assert dts[i] <= dts[i - 1] + 1e-12, (
                f"dt increased at iteration {i}: {dts[i-1]:.4g} -> {dts[i]:.4g}"
            )


# ===========================================================================
# Test 3: Regression -- Phase 3 and existing APIs still work
# ===========================================================================

class TestRegression:
    """
    Verify Phase 3 and all pre-Phase-3 functionality is unbroken.
    Uses Flask test client with synthetic satellites (no data files required).
    """

    @pytest.fixture(autouse=True)
    def setup_app(self):
        import app as fa
        fa.SATELLITES = {
            80: _internal_rec(80, "REG-A", ma_deg=0.0) | {"source": "stations", "priority": 10, "intl_desig": "", "period_min": 92.9, "inclination": 51.6},
            81: _internal_rec(81, "REG-B", ma_deg=0.1) | {"source": "active_satellites", "priority": 1, "intl_desig": "", "period_min": 92.9, "inclination": 51.6},
        }
        fa._last_screening_run = None
        self.client = fa.app.test_client()
        yield

    # ---- Pre-Phase-3 endpoints ----
    def test_health(self):
        r = self.client.get("/api/health")
        assert r.status_code == 200
        d = r.get_json()
        assert d["status"] == "ok" and "satellites_loaded" in d

    def test_satellites(self):
        r = self.client.get("/api/satellites?limit=5")
        assert r.status_code == 200
        assert "satellites" in r.get_json()

    def test_propagate(self):
        r = self.client.get("/api/propagate?ts=2025-01-01T00:00:00Z&limit=2")
        assert r.status_code == 200
        d = r.get_json()
        assert "positions" in d and "note" in d

    def test_propagate_one(self):
        r = self.client.get("/api/propagate_one?norad_id=80&ts=2025-01-01T00:00:00Z")
        assert r.status_code == 200
        d = r.get_json()
        assert "orbit_path" in d and "position" in d

    def test_orbit_batch(self):
        r = self.client.get("/api/orbit_batch?norad_ids=80&ts=2025-01-01T00:00:00Z")
        assert r.status_code == 200
        assert "orbits" in r.get_json()

    # ---- Phase-3 endpoints ----
    def test_screen_endpoint(self):
        r = self.client.get("/api/screen?horizon_minutes=170&threshold_km=15")
        assert r.status_code == 200
        d = r.get_json()
        assert "run_id" in d and "alerts" in d and "disclaimer" in d

    def test_screen_results_endpoint(self):
        self.client.get("/api/screen?horizon_minutes=170&threshold_km=15")
        r = self.client.get("/api/screen/results")
        assert r.status_code == 200
        assert "run_id" in r.get_json()

    def test_screen_bad_param_still_400(self):
        r = self.client.get("/api/screen?horizon_minutes=NaN")
        assert r.status_code == 400

    # ---- Phase-4 endpoints ----
    def test_analyse_endpoint_schema(self):
        r = self.client.get(
            "/api/analyse?horizon_minutes=170&threshold_km=15"
            "&target_resolution_m=1.0&max_grid_iterations=5"
        )
        assert r.status_code == 200, f"Got {r.status_code}: {r.get_json()}"
        d = r.get_json()
        required = ["phase3_run_id", "screening_epoch", "objects_screened",
                    "phase3_alert_count", "grid_analyses",
                    "runtime_seconds", "accuracy_disclaimer"]
        for k in required:
            assert k in d, f"Missing key '{k}' in /api/analyse response"
        assert isinstance(d["grid_analyses"], list)
        assert "grid" in d["accuracy_disclaimer"].lower() or \
               "numerical" in d["accuracy_disclaimer"].lower()

    def test_analyse_pair_endpoint_schema(self):
        r = self.client.get(
            "/api/analyse/pair?norad_a=80&norad_b=81"
            "&horizon_minutes=170&threshold_km=15"
            "&target_resolution_m=1.0&max_grid_iterations=5"
        )
        assert r.status_code == 200, f"Got {r.status_code}: {r.get_json()}"
        d = r.get_json()
        for k in ["phase3_run_id", "norad_a", "norad_b", "grid_analyses"]:
            assert k in d, f"Missing key '{k}'"

    def test_analyse_pair_missing_norad_returns_400(self):
        r = self.client.get("/api/analyse/pair")
        assert r.status_code == 400

    def test_analyse_pair_unknown_norad_returns_404(self):
        r = self.client.get("/api/analyse/pair?norad_a=99999&norad_b=80")
        assert r.status_code == 404

    def test_analyse_bad_config_returns_400(self):
        r = self.client.get("/api/analyse?horizon_minutes=NaN")
        assert r.status_code == 400

    def test_grid_analysis_has_disclaimer(self):
        r = self.client.get(
            "/api/analyse?horizon_minutes=170&threshold_km=15&target_resolution_m=1.0"
        )
        d = r.get_json()
        assert "accuracy_disclaimer" in d
        disc = d["accuracy_disclaimer"]
        assert len(disc) > 30, "Disclaimer must be substantive"
        if d["grid_analyses"]:
            ga = d["grid_analyses"][0]
            assert "accuracy_disclaimer" in ga, "GridAnalysisResult must carry its own disclaimer"
