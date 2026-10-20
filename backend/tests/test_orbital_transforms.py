"""
Orbital Coordinate Transform Tests
===================================
Unit-level tests for:
  - _greenwich_sidereal_angle   (GMST calculation)
  - _teme_to_ecef               (rotation correctness, identity, reversibility)
  - _compute_orbit_path         (continuity, ordering, valid radius, no artificial closure,
                                 invalid-sample skipping)

Run with: python -m pytest backend/tests/test_orbital_transforms.py -v
"""

import math
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import app as fa

# Convenience aliases
_gmst      = fa._greenwich_sidereal_angle
_teme_ecef = fa._teme_to_ecef
_orbit     = fa._compute_orbit_path

# Known J2000.0 epoch: 2000-01-01 12:00:00 TT -> JD = 2451545.0
J2000_JD       = 2451545.0
J2000_JD_FRAC  = 0.0

# Published GMST at J2000.0 (IAU 1982): ~280.46 deg = 4.8950 rad
GMST_J2000_RAD = 4.894961213


# ===========================================================================
# GMST Tests
# ===========================================================================

class TestGreenwichSiderealAngle:
    def test_j2000_epoch_approx(self):
        """GMST at J2000.0 should be ~4.895 rad (approx 280.46 deg)."""
        gmst = _gmst(J2000_JD, J2000_JD_FRAC)
        assert abs(gmst - GMST_J2000_RAD) < 0.01, (
            f"GMST at J2000.0 = {math.degrees(gmst):.4f} deg, expected approx 280.46 deg"
        )

    def test_range_zero_to_two_pi(self):
        """GMST should always be in [0, 2pi)."""
        for offset_days in [0, 1, 7, 30, 365, 3652]:
            jd = J2000_JD + offset_days
            gmst = _gmst(jd, J2000_JD_FRAC)
            assert 0.0 <= gmst < 2 * math.pi, (
                f"GMST out of [0, 2pi) at JD+{offset_days}d: {gmst}"
            )

    def test_half_sidereal_day_advances_pi(self):
        """Half a sidereal day later, GMST should advance by ~pi rad."""
        gmst0    = _gmst(J2000_JD, 0.0)
        gmst_half = _gmst(J2000_JD, 0.4986)   # ~0.5 sidereal days
        diff = (gmst_half - gmst0) % (2 * math.pi)
        assert math.isclose(diff, math.pi, abs_tol=0.05), (
            f"Half sidereal day GMST advance = {diff:.4f} rad, expected ~pi"
        )


# ===========================================================================
# TEME->ECEF rotation tests
# ===========================================================================

class TestTemeToEcef:
    def test_gmst_zero_identity(self):
        """At GMST=0 the rotation is the identity: ECEF == TEME."""
        xe, ye, ze = _teme_ecef(7000.0, 0.0, 0.0, gmst=0.0)
        assert math.isclose(xe, 7000.0, abs_tol=1e-6)
        assert math.isclose(ye, 0.0,    abs_tol=1e-6)
        assert math.isclose(ze, 0.0,    abs_tol=1e-6)

    def test_gmst_pi_half_rotates_y_to_x(self):
        """At GMST=pi/2, a point on TEME +Y maps to ECEF +X."""
        xe, ye, ze = _teme_ecef(0.0, 7000.0, 0.0, gmst=math.pi / 2)
        assert math.isclose(xe,  7000.0, abs_tol=1e-4)
        assert math.isclose(ye,  0.0,    abs_tol=1e-4)
        assert math.isclose(ze,  0.0,    abs_tol=1e-4)

    def test_z_component_unchanged(self):
        """The Z (polar) component must not be affected by GMST rotation."""
        for z_val in [0.0, 6356.0, -6356.0]:
            _, _, ze = _teme_ecef(1000.0, 2000.0, z_val, gmst=1.234)
            assert math.isclose(ze, z_val, abs_tol=1e-9)

    def test_radius_preserved(self):
        """Rotation must be norm-preserving (isometric)."""
        r_in = 7000.0
        for gmst in [0.0, math.pi / 4, math.pi / 2, math.pi, 3 * math.pi / 2]:
            xe, ye, ze = _teme_ecef(r_in, 0.0, 0.0, gmst)
            r_out = math.sqrt(xe**2 + ye**2 + ze**2)
            assert math.isclose(r_out, r_in, rel_tol=1e-9), (
                f"Radius changed at GMST={gmst:.3f}: in={r_in}, out={r_out}"
            )

    def test_determinant_plus_one(self):
        """Transform must have determinant +1 (proper rotation, no reflection)."""
        g = 0.7
        e1 = _teme_ecef(1.0, 0.0, 0.0, g)
        e2 = _teme_ecef(0.0, 1.0, 0.0, g)
        e3 = _teme_ecef(0.0, 0.0, 1.0, g)
        # scalar triple product = det
        cx = e2[1]*e3[2] - e2[2]*e3[1]
        cy = e2[2]*e3[0] - e2[0]*e3[2]
        cz = e2[0]*e3[1] - e2[1]*e3[0]
        det = e1[0]*cx + e1[1]*cy + e1[2]*cz
        assert math.isclose(det, 1.0, abs_tol=1e-9), (
            f"TEME->ECEF determinant = {det}, expected +1"
        )


# ===========================================================================
# _compute_orbit_path tests
# ===========================================================================

@pytest.fixture(scope="module")
def iss_sat_data():
    fa.SATELLITES = fa._load_satellites()
    sat = fa.SATELLITES.get(25544)
    if sat is None:
        pytest.skip("ISS (NORAD 25544) not found in catalogue")
    return sat


class TestComputeOrbitPath:
    def test_returns_list(self, iss_sat_data):
        pts = _orbit(iss_sat_data, J2000_JD + 9000, J2000_JD_FRAC, steps=30)
        assert isinstance(pts, list)

    def test_returns_enough_points(self, iss_sat_data):
        """With steps=60, expect between 55 and 61 points (a few may skip on SGP4 error)."""
        pts = _orbit(iss_sat_data, J2000_JD + 9000, J2000_JD_FRAC, steps=60)
        assert 55 <= len(pts) <= 61, f"Unexpected point count: {len(pts)}"

    def test_points_have_xyz_keys(self, iss_sat_data):
        pts = _orbit(iss_sat_data, J2000_JD + 9000, J2000_JD_FRAC, steps=30)
        for pt in pts:
            assert "x" in pt and "y" in pt and "z" in pt

    def test_all_radii_in_leo_range(self, iss_sat_data):
        """Every point must be between 6400 km and 9000 km from Earth centre."""
        pts = _orbit(iss_sat_data, J2000_JD + 9000, J2000_JD_FRAC, steps=60)
        for pt in pts:
            r = math.sqrt(pt["x"]**2 + pt["y"]**2 + pt["z"]**2)
            assert 6400.0 < r < 9000.0, f"Point outside LEO band: r={r:.1f} km"

    def test_no_nan_or_inf(self, iss_sat_data):
        pts = _orbit(iss_sat_data, J2000_JD + 9000, J2000_JD_FRAC, steps=60)
        for pt in pts:
            assert math.isfinite(pt["x"]) and math.isfinite(pt["y"]) and math.isfinite(pt["z"])

    def test_path_is_not_artificially_closed(self, iss_sat_data):
        """
        First and last points must NOT be the same.
        For LEO, the ECEF ground track shifts ~22 deg west per orbit,
        so the gap between start and end should be well over 100 km.
        """
        pts = _orbit(iss_sat_data, J2000_JD + 9000, J2000_JD_FRAC, steps=60)
        assert len(pts) >= 2
        first, last = pts[0], pts[-1]
        dist = math.sqrt((first["x"]-last["x"])**2 +
                         (first["y"]-last["y"])**2 +
                         (first["z"]-last["z"])**2)
        assert dist > 100.0, (
            f"First/last points only {dist:.1f} km apart -- path may be artificially closed."
        )

    def test_consecutive_separation_physical(self, iss_sat_data):
        """
        Consecutive ECEF points must be < 2000 km apart.
        A spurious closure chord (ISS gap ~2700 km) would fail this test.
        """
        pts = _orbit(iss_sat_data, J2000_JD + 9000, J2000_JD_FRAC, steps=60)
        for i in range(len(pts) - 1):
            p, q = pts[i], pts[i + 1]
            d = math.sqrt((p["x"]-q["x"])**2 + (p["y"]-q["y"])**2 + (p["z"]-q["z"])**2)
            assert d < 2000.0, (
                f"Unphysical gap {d:.1f} km at index {i} -- "
                "may be an artificial closure chord or mis-ordered timestamps."
            )

    def test_invalid_satrec_returns_empty(self):
        """If satrec is None (init failure), _compute_orbit_path must return []."""
        bad = {"satrec": None, "init_error": "unit-test", "period_min": 90.0}
        pts = _orbit(bad, J2000_JD, J2000_JD_FRAC, steps=10)
        assert pts == []
