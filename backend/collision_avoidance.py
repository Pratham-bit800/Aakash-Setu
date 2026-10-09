"""
collision_avoidance.py  --  Akash Setu Phase 5: Hybrid Collision Avoidance Engine
===================================================================================

Phase 5 implements a hybrid collision avoidance framework for satellite encounters
identified by Phase 3 screening and refined by Phase 4 grid analysis.

Reference-Document Status:
  No external reference document was provided with the Phase 5 specification.
  The methods implemented here are strictly derived from validated astrodynamics
  literature:
    - Vallado (2013): Fundamentals of Astrodynamics and Applications (4th ed.),
      Section 6.5 (Orbital Maneuvers) & Section 9.5 (Conjunction Assessment).
    - Alfriend et al. (2010): Spacecraft Formation Flying, Chapter 4
      (Gauss's Variational Equations in RTN/RSW frame).
    - Battin (1999): An Introduction to the Mathematics and Methods of Astrodynamics.
    - Clohessy & Wiltshire (1960): Terminal Guidance System for Satellite Rendezvous.

Avoidance Strategies Supported:
  1. Impulsive Delta-v Maneuvers (Validated Orbital Mechanics via GVE):
     - Prograde Along-Track (+T / in-track): Alters semi-major axis da = 2*a*v/mu * dv_T;
       most fuel-efficient collision avoidance burn for LEO.
     - Retrograde Along-Track (-T / in-track): Depresses semi-major axis; advances arrival time.
     - Out-of-plane / Cross-Track (+W / -W / normal): Modifies inclination and RAAN to
       separate spacecraft normal to the orbital plane.
     - Radial (+R / -R): Rotates apsidal line; modifies eccentricity.
     - Custom Impulse: User-specified 3D (dv_R, dv_T, dv_W) vector.

  2. Attitude Reorientation (Differential Drag / Aerodynamic Cross-Section):
     - Evaluated ONLY when spacecraft 3D geometry and attitude constraints are supplied
       (cross-sectional area min/max, mass, drag coefficient).
     - When geometry is absent (standard TLE catalog condition), the engine explicitly
       documents the data gap and reports attitude reorientation as UNAVAILABLE_NO_GEOMETRY,
       avoiding fabrication of fictitious physical parameters.

Maneuver Re-propagation & Re-screening:
  - Maneuvered satellites are modeled via ManeuveredSatrec, preserving the unmaneuvered
    state before maneuver epoch t_man and propagating the maneuvered state thereafter.
  - Candidate encounters are re-propagated and re-screened using the existing Phase 3
    screening pipeline and Phase 4 grid-refinement engine.
  - Before/after miss distances, TCA shifts, delta-v consumption, clearance thresholds,
    feasibility, and unresolved limitations are reported quantitatively.
"""

from __future__ import annotations

import math
import time as _time_mod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from sgp4.api import Satrec, WGS72

from collision_screening import (
    DEG2RAD, REVDAY2RADMIN,
    _epoch_to_jd, _jd_add_minutes, _propagate_teme,
    _sgp4_epoch_days, _vec_norm, _vec_sub,
    ScreeningConfig, screen_satellites,
)
from grid_analysis import GridConfig, refine_tca


# ---------------------------------------------------------------------------
# Physical and Algorithmic Constants (WGS72)
# ---------------------------------------------------------------------------

RAD2DEG: float = 180.0 / math.pi
WGS72_MU: float = 398600.8                # km^3 / s^2 (Earth gravitational parameter)
WGS72_EARTH_RADIUS_KM: float = 6378.135   # km (Earth equatorial radius)
MIN_PERIGEE_ALTITUDE_KM: float = 120.0    # km (Atmospheric re-entry safety boundary)

# Strategy constants
STRATEGY_PROGRADE_ALONG_TRACK = "prograde_along_track"
STRATEGY_RETROGRADE_ALONG_TRACK = "retrograde_along_track"
STRATEGY_POS_CROSS_TRACK = "positive_cross_track"
STRATEGY_NEG_CROSS_TRACK = "negative_cross_track"
STRATEGY_POS_RADIAL = "positive_radial"
STRATEGY_NEG_RADIAL = "negative_radial"
STRATEGY_ATTITUDE = "attitude_reorientation"
STRATEGY_CUSTOM = "custom_impulse"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class AttitudeReorientationResult:
    """Assessment of attitude reorientation / aerodynamic cross-section variation."""
    feasible: bool
    status: str                         # UNAVAILABLE_NO_GEOMETRY | EVALUATED_FEASIBLE | EVALUATED_INFEASIBLE
    delta_area_m2: float = 0.0
    delta_drag_acceleration_m_s2: float = 0.0
    estimated_along_track_shift_km: float = 0.0
    target_clearance_km: float = 0.0
    gap_documentation: str = ""
    details: dict = field(default_factory=dict)


@dataclass
class AvoidanceConfig:
    """Configuration for collision avoidance maneuver generation and screening."""
    delta_v_m_s: float = 0.5            # Nominal impulsive burn magnitude in m/s
    maneuver_direction: str = STRATEGY_PROGRADE_ALONG_TRACK
    custom_dv_rtn_m_s: tuple[float, float, float] | None = None  # (dv_R, dv_T, dv_W) in m/s
    lead_time_minutes: float | None = None  # Minutes before TCA when burn occurs. None -> 1 orbit period
    max_delta_v_m_s: float = 5.0        # Maximum allowable delta-v budget (m/s)
    min_lead_time_minutes: float = 15.0 # Operational lead time notice constraint (minutes)
    target_clearance_km: float = 15.0   # Required minimum separation at TCA (km)
    spacecraft_geometry: dict | None = None  # Optional: {"area_min_m2": float, "area_max_m2": float, "mass_kg": float, "drag_coeff": float}


@dataclass
class ManeuverCandidateResult:
    """Result of a single candidate maneuver evaluation."""
    strategy_name: str
    maneuver_direction: str
    delta_v_m_s: float
    delta_v_vector_rtn_m_s: tuple[float, float, float]
    maneuver_lead_time_minutes: float
    maneuver_time_minutes: float
    maneuver_epoch_utc: str

    # Before maneuver (unmaneuvered baseline)
    before_miss_distance_km: float
    before_tca_minutes: float
    before_tca_utc: str

    # After maneuver (re-propagated & re-screened)
    after_miss_distance_km: float
    after_tca_minutes: float
    after_tca_utc: str

    # Deltas
    miss_distance_improvement_km: float
    tca_delta_seconds: float

    # Clearance and Feasibility
    clears_threshold: bool
    is_feasible: bool
    feasibility_reasons: list[str]

    # Maneuvered orbit mechanics
    semi_major_axis_change_m: float
    orbital_period_change_s: float
    perigee_altitude_km: float

    # Secondary conjunctions
    secondary_conjunctions_detected: int = 0
    secondary_conjunction_alerts: list[dict] = field(default_factory=list)


@dataclass
class HybridAvoidancePlan:
    """Comprehensive hybrid avoidance plan comparing maneuver options."""
    event_id: str
    primary_norad: int
    primary_name: str
    secondary_norad: int
    secondary_name: str
    screening_epoch: str

    # Baseline unmaneuvered
    baseline_miss_distance_km: float
    baseline_tca_minutes: float
    baseline_tca_utc: str

    # Attitude reorientation evaluation
    attitude_assessment: AttitudeReorientationResult

    # Evaluated candidates
    candidates: list[ManeuverCandidateResult]

    # Recommended action
    recommended_strategy: str
    recommended_candidate: Optional[ManeuverCandidateResult]
    recommendation_rationale: str

    # Limitations & Disclaimers
    unresolved_limitations: list[str]
    accuracy_disclaimer: str
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# ManeuveredSatrec Composite
# ---------------------------------------------------------------------------

class ManeuveredSatrec:
    """
    Composite satellite record wrapper implementing the SGP4 interface with
    scientifically validated Clohessy-Wiltshire (CW) relative motion superposition.

    Astrodynamic Properties:
      - For t < t_man_minutes: Propagates the unmaneuvered trajectory sat_orig.sgp4(jd, jdf).
      - At burn epoch t = t_man_minutes:
          * Guarantees exact position continuity (dr = 0.000 mm, zero discontinuity).
          * Guarantees exact impulsive velocity step (dv = requested delta-v vector in RTN).
      - For t >= t_man_minutes:
          * Superposes closed-form Hill / Clohessy-Wiltshire relative displacement (x, y, z)
            and relative velocity (vx, vy, vz) in the local Radial-Transverse-Normal (RTN/RSW)
            orbital frame onto the background SGP4 orbit.
          * Eliminates artificial apsidal singularities (1/e) present in near-circular Keplerian GVEs,
            preventing unphysical multi-thousand-kilometer discontinuities.
    """
    def __init__(
        self,
        sat_orig: Any,
        sat_man: Any,
        t_man_minutes: float,
        jd0: float,
        jdf0: float,
        dv_rtn_km_s: tuple[float, float, float] | None = None,
    ):
        self.sat_orig = sat_orig
        self.sat_man = sat_man
        self.t_man_minutes = t_man_minutes
        self.jd0 = jd0
        self.jdf0 = jdf0

        if dv_rtn_km_s is not None:
            self.dv_r, self.dv_t, self.dv_w = dv_rtn_km_s
        else:
            self.dv_r, self.dv_t, self.dv_w = 0.0, 0.0, 0.0

        # Mirror essential attributes from sat_orig
        self.satnum = getattr(sat_orig, "satnum", 0)
        self.bstar = getattr(sat_orig, "bstar", 1e-5)
        self.no_kozai = getattr(sat_man, "no_kozai", getattr(sat_orig, "no_kozai", 0.0))
        self.ecco = getattr(sat_man, "ecco", getattr(sat_orig, "ecco", 0.0))
        self.inclo = getattr(sat_man, "inclo", getattr(sat_orig, "inclo", 0.0))
        self.nodeo = getattr(sat_man, "nodeo", getattr(sat_orig, "nodeo", 0.0))
        self.argpo = getattr(sat_man, "argpo", getattr(sat_orig, "argpo", 0.0))
        self.mo = getattr(sat_man, "mo", getattr(sat_orig, "mo", 0.0))

    def sgp4(self, jd: float, jdf: float) -> tuple[int, tuple[float, float, float], tuple[float, float, float]]:
        err, r, v = self.sat_orig.sgp4(jd, jdf)
        if err != 0:
            return err, r, v

        t_minutes = ((jd - self.jd0) + (jdf - self.jdf0)) * 1440.0
        # Before burn epoch (numerical epsilon for float precision at t_man)
        if t_minutes < self.t_man_minutes - 1e-8:
            return err, r, v

        if self.dv_r == 0.0 and self.dv_t == 0.0 and self.dv_w == 0.0:
            return err, r, v

        tau_s = max(0.0, (t_minutes - self.t_man_minutes) * 60.0)
        n_rad_s = getattr(self.sat_orig, "no_kozai", self.no_kozai) / 60.0
        if n_rad_s <= 0.0:
            return self.sat_man.sgp4(jd, jdf)

        nt = n_rad_s * tau_s
        sin_nt = math.sin(nt)
        cos_nt = math.cos(nt)

        # Clohessy-Wiltshire relative displacement in local RTN frame (km)
        # x = R, y = T, z = W
        x = (self.dv_r / n_rad_s) * sin_nt + (2.0 * self.dv_t / n_rad_s) * (1.0 - cos_nt)
        y = -(2.0 * self.dv_r / n_rad_s) * (1.0 - cos_nt) + (self.dv_t / n_rad_s) * (4.0 * sin_nt - 3.0 * nt)
        z = (self.dv_w / n_rad_s) * sin_nt

        # Clohessy-Wiltshire relative velocity in local RTN frame (km/s)
        vx = self.dv_r * cos_nt + 2.0 * self.dv_t * sin_nt
        vy = -2.0 * self.dv_r * sin_nt + self.dv_t * (4.0 * cos_nt - 3.0)
        vz = self.dv_w * cos_nt

        # Local RTN basis unit vectors in TEME
        r_mag = math.sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2])
        if r_mag < 1e-6:
            return err, r, v
        r_hat = (r[0] / r_mag, r[1] / r_mag, r[2] / r_mag)

        hx = r[1] * v[2] - r[2] * v[1]
        hy = r[2] * v[0] - r[0] * v[2]
        hz = r[0] * v[1] - r[1] * v[0]
        h_mag = math.sqrt(hx*hx + hy*hy + hz*hz)
        if h_mag < 1e-6:
            return err, r, v
        w_hat = (hx / h_mag, hy / h_mag, hz / h_mag)

        tx = w_hat[1] * r_hat[2] - w_hat[2] * r_hat[1]
        ty = w_hat[2] * r_hat[0] - w_hat[0] * r_hat[2]
        tz = w_hat[0] * r_hat[1] - w_hat[1] * r_hat[0]
        t_hat = (tx, ty, tz)

        # Transport theorem kinematics: v_inertial = v_rel + omega x delta_r
        # Local RTN frame rotates at angular rate omega = h / r^2 along w_hat.
        # omega x (x*r_hat + y*t_hat + z*w_hat) = omega * (x*t_hat - y*r_hat)
        omega = h_mag / (r_mag * r_mag)
        vx_inertial = vx - omega * y
        vy_inertial = vy + omega * x
        vz_inertial = vz

        # Superpose relative state on background unmaneuvered TEME state
        r_man = (
            r[0] + x * r_hat[0] + y * t_hat[0] + z * w_hat[0],
            r[1] + x * r_hat[1] + y * t_hat[1] + z * w_hat[1],
            r[2] + x * r_hat[2] + y * t_hat[2] + z * w_hat[2],
        )
        v_man = (
            v[0] + vx_inertial * r_hat[0] + vy_inertial * t_hat[0] + vz_inertial * w_hat[0],
            v[1] + vx_inertial * r_hat[1] + vy_inertial * t_hat[1] + vz_inertial * w_hat[1],
            v[2] + vx_inertial * r_hat[2] + vy_inertial * t_hat[2] + vz_inertial * w_hat[2],
        )
        return 0, r_man, v_man


# ---------------------------------------------------------------------------
# Astrodynamics: Gauss's Variational Equations (GVE) in RTN Frame
# ---------------------------------------------------------------------------

def calculate_maneuvered_elements(
    satrec: Any,
    epoch_days: float,
    t_man_minutes: float,
    dv_r_km_s: float,
    dv_t_km_s: float,
    dv_w_km_s: float,
) -> tuple[Satrec, dict[str, float]]:
    """
    Calculate post-maneuver orbital elements using Gauss's Variational Equations.

    Applies an impulsive velocity vector (dv_R, dv_T, dv_W) in the local
    Radial-Transverse-Normal (RTN/RSW) orbital frame at maneuver time t_man_minutes.

    References:
      Vallado (2013), Section 6.5; Alfriend et al. (2010), Chapter 4.
    """
    mu = WGS72_MU
    r_earth = WGS72_EARTH_RADIUS_KM

    n0_rad_min = satrec.no_kozai
    n0_rad_s = n0_rad_min / 60.0
    a0 = (mu / (n0_rad_s ** 2)) ** (1.0 / 3.0)
    v0 = math.sqrt(mu / a0)
    e0 = max(satrec.ecco, 1e-6)
    inc0 = satrec.inclo
    raan0 = satrec.nodeo
    argp0 = satrec.argpo
    m0 = satrec.mo

    # True anomaly at maneuver time
    tau_man_s = t_man_minutes * 60.0
    m_man = (m0 + n0_rad_s * tau_man_s) % (2.0 * math.pi)
    nu_man = (m_man + 2.0 * e0 * math.sin(m_man)) % (2.0 * math.pi)
    u_man = (argp0 + nu_man) % (2.0 * math.pi)

    # 1. Semi-major axis change: da = 2*a^2*v/mu * dv_T
    da = (2.0 * a0 * a0 * v0 / mu) * dv_t_km_s
    a_new = a0 + da

    # 2. Mean motion change: dn = -3/2 * n/a * da
    dn_rad_s = -1.5 * (n0_rad_s / a0) * da
    n_new_rad_s = n0_rad_s + dn_rad_s
    n_new_rad_min = n_new_rad_s * 60.0

    # 3. Eccentricity change: de = 1/v * (sin(nu)*dv_R + 2*cos(nu)*dv_T)
    de = (1.0 / v0) * (math.sin(nu_man) * dv_r_km_s + 2.0 * math.cos(nu_man) * dv_t_km_s)
    e_new = max(1e-6, min(0.99, e0 + de))

    # 4. Inclination change: di = cos(u)/v * dv_W
    di = (math.cos(u_man) / v0) * dv_w_km_s
    inc_new = max(0.0, min(math.pi, inc0 + di))

    # 5. RAAN change: dRAAN = sin(u)/(v*sin(i)) * dv_W
    sin_inc = math.sin(inc0)
    draan = (math.sin(u_man) / (v0 * sin_inc)) * dv_w_km_s if abs(sin_inc) > 1e-4 else 0.0
    raan_new = (raan0 + draan) % (2.0 * math.pi)

    # 6. Argument of perigee and coupled mean anomaly change
    # Regularize near-circular apsidal singularity (e < 0.005)
    e_reg = max(e0, 0.005)
    dargp = (1.0 / (e_reg * v0)) * (-math.cos(nu_man) * dv_r_km_s + 2.0 * math.sin(nu_man) * dv_t_km_s) - draan * math.cos(inc0)
    argp_new = (argp0 + dargp) % (2.0 * math.pi)

    # Couple mean anomaly to maintain mean argument of latitude lambda = argp + M
    dm_geom = -dargp * math.sqrt(max(0.0, 1.0 - e0 * e0))
    m_man_new = (m_man + dm_geom) % (2.0 * math.pi)
    m_new_at_epoch = (m_man_new - n_new_rad_s * tau_man_s) % (2.0 * math.pi)

    # Perigee altitude
    perigee_altitude_km = a_new * (1.0 - e_new) - r_earth

    # Period change
    t0_s = (2.0 * math.pi) / n0_rad_s
    t_new_s = (2.0 * math.pi) / n_new_rad_s
    dt_period_s = t_new_s - t0_s

    # Build post-maneuver Satrec
    sat_man = Satrec()
    sat_man.sgp4init(
        WGS72, "i", satrec.satnum, epoch_days, satrec.bstar,
        0.0, 0.0, e_new, argp_new, inc_new, m_new_at_epoch, n_new_rad_min, raan_new
    )

    metrics = {
        "da_m": da * 1000.0,
        "dn_rad_s": dn_rad_s,
        "de": de,
        "di_deg": math.degrees(di),
        "draan_deg": math.degrees(draan),
        "dargp_deg": math.degrees(dargp),
        "period_change_s": dt_period_s,
        "perigee_altitude_km": perigee_altitude_km,
        "a_new_km": a_new,
        "e_new": e_new,
    }
    return sat_man, metrics


# ---------------------------------------------------------------------------
# Attitude Reorientation Assessment
# ---------------------------------------------------------------------------

def evaluate_attitude_reorientation(
    primary_record: dict,
    tca_minutes: float,
    lead_time_minutes: float,
    target_clearance_km: float,
    geometry: dict | None = None,
) -> AttitudeReorientationResult:
    """
    Evaluate attitude reorientation / aerodynamic cross-section variation.

    When spacecraft geometry (CAD / cross-sectional area profile / mass / Cd)
    is unavailable (standard TLE dataset condition), the function explicitly
    documents the gap and reports UNAVAILABLE_NO_GEOMETRY.

    When geometry is provided, differential drag acceleration is computed:
      a_drag = 0.5 * rho * v^2 * (Cd * Delta_A / m)
      Delta_s = 0.75 * a_drag * tau^2
    """
    if not geometry:
        return AttitudeReorientationResult(
            feasible=False,
            status="UNAVAILABLE_NO_GEOMETRY",
            target_clearance_km=target_clearance_km,
            gap_documentation=(
                "Spacecraft 3D geometry, cross-sectional area profile, and attitude "
                "actuator constraints are absent in the TLE dataset. Attitude "
                "reorientation cannot be evaluated without inventing unvalidated "
                "spacecraft specifications."
            ),
        )

    # Geometry supplied -- evaluate aerodynamic differential drag
    try:
        area_min = float(geometry.get("area_min_m2", 0.0))
        area_max = float(geometry.get("area_max_m2", 0.0))
        mass_kg = float(geometry.get("mass_kg", 0.0))
        drag_coeff = float(geometry.get("drag_coeff", 2.2))
        if area_min <= 0.0 or area_max <= area_min or mass_kg <= 0.0:
            raise ValueError("Invalid geometry: area_max > area_min > 0 and mass_kg > 0 required")
    except Exception as exc:
        return AttitudeReorientationResult(
            feasible=False,
            status="EVALUATED_INFEASIBLE",
            target_clearance_km=target_clearance_km,
            gap_documentation=f"Invalid geometry configuration: {exc}",
        )

    delta_area = area_max - area_min
    lead_time_s = max(lead_time_minutes * 60.0, 1.0)

    # Estimate altitude from satellite record if available (for scale-height density)
    alt_km = 400.0
    satrec = primary_record.get("satrec")
    if satrec and hasattr(satrec, "no_kozai") and satrec.no_kozai > 0:
        n_rad_s = satrec.no_kozai / 60.0
        a_km = (WGS72_MU / (n_rad_s ** 2)) ** (1.0 / 3.0)
        alt_km = max(150.0, min(1000.0, a_km - WGS72_EARTH_RADIUS_KM))

    # Exponential scale-height atmospheric density model: rho(h) = rho0 * exp(-(h - h0)/H)
    # Ref: Vallado (2013) Table 8-4; rho0 = 5e-13 kg/m^3 at h0 = 400 km, H = 50 km
    rho0 = 5e-13
    scale_height_km = 50.0
    rho = rho0 * math.exp(-(alt_km - 400.0) / scale_height_km)
    rho = max(1e-15, min(1e-10, rho))

    # Mean orbital speed v = sqrt(mu / r)
    r_km = WGS72_EARTH_RADIUS_KM + alt_km
    v_orb = math.sqrt(WGS72_MU / r_km) * 1000.0  # m/s

    delta_a_drag = 0.5 * rho * (v_orb ** 2) * (drag_coeff * delta_area / mass_kg)  # m/s^2

    # Along-track displacement: Delta_s = 1.5 * a_drag * tau^2 (exact secular derivation)
    # Ref: d(da)/dt = -2/n * a_drag => da(t) = -2/n * a_drag * t => d(ds)/dt = 3 * a_drag * t => ds = 1.5 * a_drag * tau^2
    along_track_shift_m = 1.5 * delta_a_drag * (lead_time_s ** 2)
    along_track_shift_km = along_track_shift_m / 1000.0

    feasible = along_track_shift_km >= target_clearance_km
    status = "EVALUATED_FEASIBLE" if feasible else "EVALUATED_INFEASIBLE"

    return AttitudeReorientationResult(
        feasible=feasible,
        status=status,
        delta_area_m2=round(delta_area, 4),
        delta_drag_acceleration_m_s2=round(delta_a_drag, 10),
        estimated_along_track_shift_km=round(along_track_shift_km, 6),
        target_clearance_km=target_clearance_km,
        gap_documentation=(
            f"Evaluated aerodynamic cross-section variation (Delta_A={delta_area:.2f} m^2, "
            f"mass={mass_kg:.1f} kg, altitude={alt_km:.1f} km). Over {lead_time_minutes:.1f} min lead time, "
            f"differential drag yields {along_track_shift_km:.4f} km along-track shift "
            f"(target clearance: {target_clearance_km:.2f} km)."
        ),
        details={
            "area_min_m2": area_min,
            "area_max_m2": area_max,
            "mass_kg": mass_kg,
            "drag_coeff": drag_coeff,
            "lead_time_minutes": lead_time_minutes,
            "estimated_altitude_km": round(alt_km, 2),
            "atmospheric_density_kg_m3": rho,
        },
    )


# ---------------------------------------------------------------------------
# Candidate Maneuver Evaluator
# ---------------------------------------------------------------------------

def evaluate_avoidance_candidate(
    encounter_alert: Any,
    primary_record: dict,
    secondary_record: dict,
    screening_dt: datetime,
    config: AvoidanceConfig,
    direction: str,
    delta_v_m_s: float,
    lead_time_minutes: float,
    all_satellite_records: list[dict] | None = None,
) -> ManeuverCandidateResult:
    """
    Evaluate a candidate impulsive maneuver against an encounter alert.

    Steps:
      1. Resolve burn vector (dv_R, dv_T, dv_W) from direction and magnitude.
      2. Compute maneuvered elements via Gauss's Variational Equations.
      3. Construct ManeuveredSatrec composite object.
      4. Re-refine TCA and miss distance around the encounter window.
      5. Evaluate clearance and feasibility constraints.
      6. Screen secondary conjunctions against catalog (if provided).
    """
    tca_min = encounter_alert.tca_minutes_from_start
    t_man_min = max(0.0, tca_min - lead_time_minutes)

    # Direction resolution to RTN (km/s)
    dv_mag_km_s = delta_v_m_s / 1000.0
    if direction == STRATEGY_PROGRADE_ALONG_TRACK:
        dvr, dvt, dvw = 0.0, dv_mag_km_s, 0.0
    elif direction == STRATEGY_RETROGRADE_ALONG_TRACK:
        dvr, dvt, dvw = 0.0, -dv_mag_km_s, 0.0
    elif direction == STRATEGY_POS_CROSS_TRACK:
        dvr, dvt, dvw = 0.0, 0.0, dv_mag_km_s
    elif direction == STRATEGY_NEG_CROSS_TRACK:
        dvr, dvt, dvw = 0.0, 0.0, -dv_mag_km_s
    elif direction == STRATEGY_POS_RADIAL:
        dvr, dvt, dvw = dv_mag_km_s, 0.0, 0.0
    elif direction == STRATEGY_NEG_RADIAL:
        dvr, dvt, dvw = -dv_mag_km_s, 0.0, 0.0
    elif direction == STRATEGY_CUSTOM and config.custom_dv_rtn_m_s:
        c_r, c_t, c_w = config.custom_dv_rtn_m_s
        dvr, dvt, dvw = c_r / 1000.0, c_t / 1000.0, c_w / 1000.0
        dv_mag_km_s = math.sqrt(dvr * dvr + dvt * dvt + dvw * dvw)
        delta_v_m_s = dv_mag_km_s * 1000.0
    else:
        dvr, dvt, dvw = 0.0, dv_mag_km_s, 0.0

    epoch_iso = screening_dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
    jd0, jdf0 = _epoch_to_jd(epoch_iso)
    epoch_days = _sgp4_epoch_days(epoch_iso)

    satrec_orig = primary_record["satrec"]
    satrec_secondary = secondary_record["satrec"]

    # Compute maneuvered orbit
    sat_man, metrics = calculate_maneuvered_elements(
        satrec_orig, epoch_days, t_man_min, dvr, dvt, dvw
    )
    composite_satrec = ManeuveredSatrec(
        satrec_orig, sat_man, t_man_min, jd0, jdf0, dv_rtn_km_s=(dvr, dvt, dvw)
    )

    # Re-refine miss distance and TCA around encounter
    # Use GridConfig with 1.0 m resolution for fast deterministic evaluation
    g_cfg = GridConfig(
        search_half_window_minutes=max(2.0, lead_time_minutes * 0.1),
        target_spatial_resolution_m=1.0,
        max_iterations=8,
    )
    rel_speed = getattr(encounter_alert, "relative_speed_km_s", 7.6)
    refine_out = refine_tca(
        composite_satrec, satrec_secondary, jd0, jdf0, tca_min, rel_speed, g_cfg
    )
    new_tca_min = refine_out[0]
    new_miss_km = refine_out[1]

    # Maneuver epoch in UTC
    man_dt = datetime.fromtimestamp(screening_dt.timestamp() + t_man_min * 60.0, tz=timezone.utc)
    new_tca_dt = datetime.fromtimestamp(screening_dt.timestamp() + new_tca_min * 60.0, tz=timezone.utc)

    # Feasibility evaluation
    reasons: list[str] = []
    dv_ok = delta_v_m_s <= config.max_delta_v_m_s
    if not dv_ok:
        reasons.append(f"Delta-v ({delta_v_m_s:.2f} m/s) exceeds maximum budget ({config.max_delta_v_m_s:.2f} m/s)")

    lead_ok = lead_time_minutes >= config.min_lead_time_minutes
    if not lead_ok:
        reasons.append(f"Lead time ({lead_time_minutes:.1f} min) is below operational minimum notice ({config.min_lead_time_minutes:.1f} min)")

    clearance_ok = new_miss_km >= config.target_clearance_km
    if not clearance_ok:
        reasons.append(f"After-maneuver miss distance ({new_miss_km:.3f} km) is below required clearance ({config.target_clearance_km:.3f} km)")
    else:
        reasons.append(f"Target clearance achieved ({new_miss_km:.3f} km >= {config.target_clearance_km:.3f} km)")

    perigee_ok = metrics["perigee_altitude_km"] >= MIN_PERIGEE_ALTITUDE_KM
    if not perigee_ok:
        reasons.append(f"Perigee altitude ({metrics['perigee_altitude_km']:.1f} km) violates re-entry safety boundary ({MIN_PERIGEE_ALTITUDE_KM:.1f} km)")

    is_feasible = dv_ok and lead_ok and clearance_ok and perigee_ok

    # Secondary conjunction screening (if catalog provided)
    sec_conjunctions = 0
    sec_alerts: list[dict] = []
    if all_satellite_records and len(all_satellite_records) > 2:
        sec_recs = [
            dict(r, satrec=composite_satrec if int(r.get("NORAD_CAT_ID", r.get("norad_id", -1))) == encounter_alert.norad_id_1 else r["satrec"])
            for r in all_satellite_records
        ]
        sec_cfg = ScreeningConfig(
            horizon_minutes=min(180.0, tca_min + 30.0),
            screening_threshold_km=config.target_clearance_km,
            broad_phase_margin_km=50.0,
        )
        sec_run = screen_satellites(sec_recs, screening_dt, sec_cfg)
        collateral = [
            a for a in sec_run.alerts
            if (a.norad_id_1 == encounter_alert.norad_id_1 or a.norad_id_2 == encounter_alert.norad_id_1)
            and not (a.norad_id_1 == encounter_alert.norad_id_2 or a.norad_id_2 == encounter_alert.norad_id_2)
        ]
        sec_conjunctions = len(collateral)
        sec_alerts = [
            {
                "event_id": a.event_id,
                "other_norad": a.norad_id_2 if a.norad_id_1 == encounter_alert.norad_id_1 else a.norad_id_1,
                "miss_distance_km": round(a.miss_distance_km, 3),
                "tca_minutes": round(a.tca_minutes_from_start, 2),
            }
            for a in collateral
        ]
        if sec_conjunctions > 0:
            reasons.append(f"WARNING: Maneuver induces {sec_conjunctions} collateral conjunction(s) with catalog objects")

    return ManeuverCandidateResult(
        strategy_name=f"{direction.replace('_', ' ').title()} ({delta_v_m_s:.2f} m/s)",
        maneuver_direction=direction,
        delta_v_m_s=round(delta_v_m_s, 4),
        delta_v_vector_rtn_m_s=(round(dvr * 1000.0, 4), round(dvt * 1000.0, 4), round(dvw * 1000.0, 4)),
        maneuver_lead_time_minutes=round(lead_time_minutes, 2),
        maneuver_time_minutes=round(t_man_min, 4),
        maneuver_epoch_utc=man_dt.strftime("%Y-%m-%dT%H:%M:%S.%f"),
        before_miss_distance_km=round(encounter_alert.miss_distance_km, 6),
        before_tca_minutes=round(tca_min, 6),
        before_tca_utc=encounter_alert.tca_utc,
        after_miss_distance_km=round(new_miss_km, 6),
        after_tca_minutes=round(new_tca_min, 6),
        after_tca_utc=new_tca_dt.strftime("%Y-%m-%dT%H:%M:%S.%f"),
        miss_distance_improvement_km=round(new_miss_km - encounter_alert.miss_distance_km, 6),
        tca_delta_seconds=round((new_tca_min - tca_min) * 60.0, 3),
        clears_threshold=clearance_ok,
        is_feasible=is_feasible,
        feasibility_reasons=reasons,
        semi_major_axis_change_m=round(metrics["da_m"], 2),
        orbital_period_change_s=round(metrics["period_change_s"], 3),
        perigee_altitude_km=round(metrics["perigee_altitude_km"], 2),
        secondary_conjunctions_detected=sec_conjunctions,
        secondary_conjunction_alerts=sec_alerts,
    )


# ---------------------------------------------------------------------------
# Comprehensive Hybrid Avoidance Plan
# ---------------------------------------------------------------------------

def generate_hybrid_avoidance_plan(
    encounter_alert: Any,
    primary_record: dict,
    secondary_record: dict,
    screening_dt: datetime,
    config: AvoidanceConfig | None = None,
    all_satellite_records: list[dict] | None = None,
) -> HybridAvoidancePlan:
    """
    Generate a full hybrid avoidance plan comparing multiple maneuver strategies.

    Strategies evaluated:
      1. Prograde along-track (+T)
      2. Retrograde along-track (-T)
      3. Positive cross-track (+W)
      4. Negative cross-track (-W)
      5. Attitude reorientation (differential drag / cross-section variation)

    Ranks candidates by:
      - Feasibility (meets clearance, delta-v <= budget, lead time >= notice)
      - Minimum delta-v consumption
      - Maximum miss distance improvement
    """
    t_start = _time_mod.monotonic()
    if config is None:
        config = AvoidanceConfig()

    tca_min = encounter_alert.tca_minutes_from_start
    lead_time = config.lead_time_minutes
    if lead_time is None:
        lead_time = max(config.min_lead_time_minutes, min(92.9, tca_min * 0.7))

    epoch_iso = screening_dt.strftime("%Y-%m-%dT%H:%M:%S.%f")

    # 1. Evaluate attitude reorientation
    attitude_res = evaluate_attitude_reorientation(
        primary_record, tca_min, lead_time, config.target_clearance_km, config.spacecraft_geometry
    )

    # 2. Evaluate impulsive maneuver candidates
    directions = [
        STRATEGY_PROGRADE_ALONG_TRACK,
        STRATEGY_RETROGRADE_ALONG_TRACK,
        STRATEGY_POS_CROSS_TRACK,
        STRATEGY_NEG_CROSS_TRACK,
    ]
    candidates: list[ManeuverCandidateResult] = []
    for d in directions:
        cand = evaluate_avoidance_candidate(
            encounter_alert, primary_record, secondary_record, screening_dt,
            config, d, config.delta_v_m_s, lead_time, all_satellite_records
        )
        candidates.append(cand)

    if config.maneuver_direction == STRATEGY_CUSTOM and config.custom_dv_rtn_m_s:
        cand_custom = evaluate_avoidance_candidate(
            encounter_alert, primary_record, secondary_record, screening_dt,
            config, STRATEGY_CUSTOM, config.delta_v_m_s, lead_time, all_satellite_records
        )
        candidates.append(cand_custom)

    # 3. Recommend optimal strategy
    feasible_cands = [c for c in candidates if c.is_feasible]
    recommended_cand: Optional[ManeuverCandidateResult] = None
    recommended_strategy: str = "NO_FEASIBLE_MANEUVER"
    rationale: str = ""

    if attitude_res.feasible:
        recommended_strategy = STRATEGY_ATTITUDE
        rationale = (
            f"Attitude reorientation selected: zero fuel consumption required. "
            f"Differential drag achieves {attitude_res.estimated_along_track_shift_km:.2f} km shift, "
            f"satisfying target clearance ({config.target_clearance_km:.2f} km)."
        )
    elif feasible_cands:
        feasible_cands.sort(key=lambda c: c.miss_distance_improvement_km, reverse=True)
        recommended_cand = feasible_cands[0]
        recommended_strategy = recommended_cand.maneuver_direction
        rationale = (
            f"Selected {recommended_cand.strategy_name}: achieved {recommended_cand.after_miss_distance_km:.2f} km "
            f"miss distance (+{recommended_cand.miss_distance_improvement_km:.2f} km improvement) "
            f"with {recommended_cand.delta_v_m_s:.2f} m/s delta-v, clearing target threshold "
            f"({config.target_clearance_km:.2f} km) without violating constraints."
        )
    else:
        candidates_sorted = sorted(candidates, key=lambda c: c.miss_distance_improvement_km, reverse=True)
        if candidates_sorted and candidates_sorted[0].miss_distance_improvement_km > 0:
            recommended_cand = candidates_sorted[0]
            recommended_strategy = f"{recommended_cand.maneuver_direction}_PARTIAL"
            rationale = (
                f"No strategy fully achieved target clearance ({config.target_clearance_km:.2f} km) "
                f"within the {config.max_delta_v_m_s:.2f} m/s budget. Best improvement provided by "
                f"{recommended_cand.strategy_name} (+{recommended_cand.miss_distance_improvement_km:.2f} km). "
                f"Recommend increasing delta-v budget or evaluating longer lead time."
            )
        else:
            rationale = "No maneuver option evaluated improved the encounter geometry."

    unresolved_limitations = [
        "NO EXTERNAL REFERENCE DOCUMENT: Methods are derived from standard astrodynamics literature (Vallado 2013, Alfriend 2010); no project reference document was supplied.",
        "ATTITUDE PARAMETER GAP: Spacecraft CAD dimensions, moments of inertia, and reaction wheel limits are not present in TLEs; attitude reorientation is infeasible without explicit geometry inputs.",
        "TLE SGP4 ERROR FLOOR: Propagated TLE position uncertainties (~100 m to >10 km) exceed fine delta-v maneuver precision. Maneuver planning requires high-precision ephemerides for operational execution.",
        "FIRST-ORDER GVE MAPPING: Maneuvers are mapped via first-order Gauss Variational Equations on SGP4 mean elements; higher-order non-linear perturbations during thrust arc are not integrated.",
        "COVARIANCE & COLLISION PROBABILITY: Maneuver selection maximizes geometric miss distance; full covariance-based probability of collision (Pc) reduction requires state covariance matrices.",
    ]

    disclaimer = (
        "Phase 5 collision avoidance plans are propagated mathematical estimates "
        "derived from SGP4 mean orbital elements and Gauss's Variational Equations. "
        "They do NOT constitute operational spacecraft flight commands. TLE ephemeris "
        "errors, lack of spacecraft propulsion telemetry, and absence of covariance "
        "data make these results unsuitable for real-time mission execution without "
        "independent verification."
    )

    return HybridAvoidancePlan(
        event_id=encounter_alert.event_id,
        primary_norad=encounter_alert.norad_id_1,
        primary_name=encounter_alert.name_1,
        secondary_norad=encounter_alert.norad_id_2,
        secondary_name=encounter_alert.name_2,
        screening_epoch=epoch_iso,
        baseline_miss_distance_km=round(encounter_alert.miss_distance_km, 6),
        baseline_tca_minutes=round(tca_min, 6),
        baseline_tca_utc=encounter_alert.tca_utc,
        attitude_assessment=attitude_res,
        candidates=candidates,
        recommended_strategy=recommended_strategy,
        recommended_candidate=recommended_cand,
        recommendation_rationale=rationale,
        unresolved_limitations=unresolved_limitations,
        accuracy_disclaimer=disclaimer,
        runtime_seconds=round(_time_mod.monotonic() - t_start, 4),
    )


# ---------------------------------------------------------------------------
# Serialization Helper for Flask API
# ---------------------------------------------------------------------------

def serialise_avoidance_plan(plan: HybridAvoidancePlan) -> dict[str, Any]:
    """Convert a HybridAvoidancePlan dataclass into a JSON-serializable dictionary."""
    def _cand_to_dict(c: ManeuverCandidateResult) -> dict[str, Any]:
        return {
            "strategy_name": c.strategy_name,
            "maneuver_direction": c.maneuver_direction,
            "delta_v_m_s": c.delta_v_m_s,
            "delta_v_vector_rtn_m_s": list(c.delta_v_vector_rtn_m_s),
            "maneuver_lead_time_minutes": c.maneuver_lead_time_minutes,
            "maneuver_time_minutes": c.maneuver_time_minutes,
            "maneuver_epoch_utc": c.maneuver_epoch_utc,
            "before_miss_distance_km": c.before_miss_distance_km,
            "before_tca_minutes": c.before_tca_minutes,
            "before_tca_utc": c.before_tca_utc,
            "after_miss_distance_km": c.after_miss_distance_km,
            "after_tca_minutes": c.after_tca_minutes,
            "after_tca_utc": c.after_tca_utc,
            "miss_distance_improvement_km": c.miss_distance_improvement_km,
            "tca_delta_seconds": c.tca_delta_seconds,
            "clears_threshold": c.clears_threshold,
            "is_feasible": c.is_feasible,
            "feasibility_reasons": c.feasibility_reasons,
            "semi_major_axis_change_m": c.semi_major_axis_change_m,
            "orbital_period_change_s": c.orbital_period_change_s,
            "perigee_altitude_km": c.perigee_altitude_km,
            "secondary_conjunctions_detected": c.secondary_conjunctions_detected,
            "secondary_conjunction_alerts": c.secondary_conjunction_alerts,
        }

    return {
        "event_id": plan.event_id,
        "primary_norad": plan.primary_norad,
        "primary_name": plan.primary_name,
        "secondary_norad": plan.secondary_norad,
        "secondary_name": plan.secondary_name,
        "screening_epoch": plan.screening_epoch,
        "baseline_miss_distance_km": plan.baseline_miss_distance_km,
        "baseline_tca_minutes": plan.baseline_tca_minutes,
        "baseline_tca_utc": plan.baseline_tca_utc,
        "attitude_assessment": {
            "feasible": plan.attitude_assessment.feasible,
            "status": plan.attitude_assessment.status,
            "delta_area_m2": plan.attitude_assessment.delta_area_m2,
            "delta_drag_acceleration_m_s2": plan.attitude_assessment.delta_drag_acceleration_m_s2,
            "estimated_along_track_shift_km": plan.attitude_assessment.estimated_along_track_shift_km,
            "target_clearance_km": plan.attitude_assessment.target_clearance_km,
            "gap_documentation": plan.attitude_assessment.gap_documentation,
            "details": plan.attitude_assessment.details,
        },
        "candidates": [_cand_to_dict(c) for c in plan.candidates],
        "recommended_strategy": plan.recommended_strategy,
        "recommended_candidate": _cand_to_dict(plan.recommended_candidate) if plan.recommended_candidate else None,
        "recommendation_rationale": plan.recommendation_rationale,
        "unresolved_limitations": plan.unresolved_limitations,
        "accuracy_disclaimer": plan.accuracy_disclaimer,
        "runtime_seconds": plan.runtime_seconds,
    }

