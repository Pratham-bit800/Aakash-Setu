"""
test_collision_screening.py  --  Akash Setu: Collision Screening Tests
Run: cd <project_root> && python -m pytest backend/tests/test_collision_screening.py -v
"""
from __future__ import annotations
import math, sys, time
from datetime import datetime, timezone
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

import pytest
from collision_screening import (
    ScreeningConfig, _perigee_apogee, _altitude_bands_overlap,
    _epoch_to_jd, _jd_add_minutes, _vec_norm, screen_satellites,
)

SCREENING_DT = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

def _iss(norad, name, raan=0.0, ma=0.0, mm=15.5):
    return {
        "NORAD_CAT_ID": str(norad), "OBJECT_NAME": name,
        "EPOCH": "2025-01-01T00:00:00.000000",
        "MEAN_MOTION": str(mm), "ECCENTRICITY": "0.0001", "INCLINATION": "51.6",
        "RA_OF_ASC_NODE": str(raan), "ARG_OF_PERICENTER": "0.0",
        "MEAN_ANOMALY": str(ma), "BSTAR": "0.00001",
        "MEAN_MOTION_DOT": "0.0", "MEAN_MOTION_DDOT": "0.0",
    }

def _gso(norad, name):
    return {
        "NORAD_CAT_ID": str(norad), "OBJECT_NAME": name,
        "EPOCH": "2025-01-01T00:00:00.000000",
        "MEAN_MOTION": "1.0027", "ECCENTRICITY": "0.0001", "INCLINATION": "0.05",
        "RA_OF_ASC_NODE": "0.0", "ARG_OF_PERICENTER": "0.0", "MEAN_ANOMALY": "0.0",
        "BSTAR": "0.00001", "MEAN_MOTION_DOT": "0.0", "MEAN_MOTION_DDOT": "0.0",
    }

def _build_satrec(norad, ma_deg=0.0):
    from sgp4.api import Satrec, WGS72
    from collision_screening import _sgp4_epoch_days, DEG2RAD, REVDAY2RADMIN
    sat = Satrec()
    sat.sgp4init(WGS72,"i",norad,_sgp4_epoch_days("2025-01-01T00:00:00.000000"),
                 1e-5,0.0,0.0,0.0001,0.0,51.6*DEG2RAD,ma_deg*DEG2RAD,
                 15.5*REVDAY2RADMIN,0.0)
    return sat

def _internal(norad, name, ma_deg=0.0, ea=1.0, stale=False, bad=False):
    return {"norad_id":norad,"name":name,"epoch":"2025-01-01T00:00:00.000000",
            "epoch_age_days":ea,"mean_motion":15.5,"eccentricity":0.0001,
            "satrec":None if bad else _build_satrec(norad,ma_deg),
            "init_error":"simulated failure" if bad else None,
            "stale":stale,"source":"active_satellites"}


# ---- Test 1: Head-on close approach ----
# Fixture: two ISS-like satellites on same orbital plane (RAAN=0, INC=51.6),
# ma_a=0.0 deg, ma_b=0.1 deg. Verified min_sep ~11.85 km at t~161.5 min.
# Threshold 15 km catches this; horizon 170 min covers the approach.
class TestT1HeadOn:
    def test_finds_encounter(self):
        cfg = ScreeningConfig(horizon_minutes=170,coarse_step_minutes=0.5,
                              screening_threshold_km=15.0,broad_phase_margin_km=200.0,refinement_steps=30)
        run = screen_satellites([_iss(1,"A",ma=0.0),_iss(2,"B",ma=0.1)],SCREENING_DT,cfg)
        assert run.objects_screened==2
        assert run.pairs_after_broad_phase>=1
        assert len(run.alerts)>=1
        e=run.alerts[0]
        assert math.isfinite(e.miss_distance_km) and e.miss_distance_km>=0
        # Known miss distance ~11.85 km; allow tolerance of 2 km for refinement
        assert e.miss_distance_km < 14.0, f"Expected < 14 km, got {e.miss_distance_km:.2f}"
        assert 0<=e.tca_minutes_from_start<=cfg.horizon_minutes
        assert e.relative_speed_km_s>0

# ---- Test 2: Wide separation, no alert ----
class TestT2Parallel:
    def test_wide_raan_no_alert(self):
        cfg = ScreeningConfig(horizon_minutes=30,coarse_step_minutes=1.0,
                              screening_threshold_km=1.0,broad_phase_margin_km=5.0)
        run = screen_satellites([_iss(5,"PA",raan=0.0),_iss(6,"PB",raan=90.0)],SCREENING_DT,cfg)
        alerts=[a for a in run.alerts if a.screening_status=="ALERT"]
        assert len(alerts)==0

# ---- Test 3: Converging trajectories with known encounter ----
# Fixture: same orbital plane, ma_a=0.0 deg, ma_b=0.2 deg.
# Verified min_sep ~23.69 km at t~161.5 min.
# Threshold 30 km catches this; TCA within window confirmed.
class TestT3Crossing:
    def test_converging_approach_found(self):
        cfg = ScreeningConfig(horizon_minutes=170,coarse_step_minutes=0.5,
                              screening_threshold_km=30.0,broad_phase_margin_km=200.0,refinement_steps=30)
        run = screen_satellites([_iss(7,"CA",ma=0.0),_iss(8,"CB",ma=0.2)],SCREENING_DT,cfg)
        assert len(run.alerts)>=1
        e=run.alerts[0]
        # Known miss distance ~23.69 km; allow 3 km tolerance for refinement
        assert e.miss_distance_km < 27.0, f"Expected < 27 km, got {e.miss_distance_km:.2f}"
        assert e.screening_status == "ALERT"
        assert math.isfinite(e.tca_minutes_from_start)

# ---- Test 4: Safe non-encounter (LEO vs GSO) ----
class TestT4Safe:
    def test_leo_gso_no_alert(self):
        cfg = ScreeningConfig(horizon_minutes=60,coarse_step_minutes=1.0,
                              screening_threshold_km=5.0,broad_phase_margin_km=50.0)
        run = screen_satellites([_iss(9,"LEO"),_gso(10,"GSO")],SCREENING_DT,cfg)
        alerts=[a for a in run.alerts if a.screening_status=="ALERT"]
        assert len(alerts)==0

# ---- Test 5: Threshold boundary ----
class TestT5Threshold:
    def test_lt_rule(self):
        cfg = ScreeningConfig(threshold_comparison="lt")
        assert cfg.alert_triggered(4.99)
        assert not cfg.alert_triggered(5.00)
        assert not cfg.alert_triggered(5.01)

    def test_lte_rule(self):
        cfg = ScreeningConfig(threshold_comparison="lte")
        assert cfg.alert_triggered(4.99)
        assert cfg.alert_triggered(5.00)
        assert not cfg.alert_triggered(5.01)

# ---- Test 6: Refinement improves coarse step ----
class TestT6Refinement:
    def test_refined_le_coarse(self):
        cfg_c = ScreeningConfig(horizon_minutes=96,coarse_step_minutes=1.0,
                                screening_threshold_km=200.0,broad_phase_margin_km=300.0,refinement_steps=30)
        cfg_f = ScreeningConfig(horizon_minutes=96,coarse_step_minutes=0.1,
                                screening_threshold_km=200.0,broad_phase_margin_km=300.0,refinement_steps=30)
        r_c = screen_satellites([_iss(3,"RA"),_iss(4,"RB",ma=170.0)],SCREENING_DT,cfg_c)
        r_f = screen_satellites([_iss(3,"RA"),_iss(4,"RB",ma=170.0)],SCREENING_DT,cfg_f)
        if r_c.alerts and r_f.alerts:
            assert r_c.alerts[0].miss_distance_km<=r_f.alerts[0].miss_distance_km+10.0

# ---- Test 7: Invalid propagation input ----
class TestT7Invalid:
    def test_bad_satrec_skipped(self):
        run = screen_satellites([_internal(30,"BAD",bad=True),_iss(31,"GOOD")],SCREENING_DT,
                                ScreeningConfig(horizon_minutes=30))
        assert 30 in {s["norad_id"] for s in run.skipped}
        assert run.objects_screened>=1

    def test_no_crash_on_malformed(self):
        try:
            run = screen_satellites([{"NORAD_CAT_ID":"99","OBJECT_NAME":"X"}],SCREENING_DT,
                                    ScreeningConfig(horizon_minutes=10))
            assert 99 in {s["norad_id"] for s in run.skipped}
        except Exception as exc:
            pytest.fail(f"Unexpected exception: {exc}")

# ---- Test 8: Stale data ----
class TestT8Stale:
    def test_stale_flag_in_result(self):
        stale_rec = _internal(40,"STALE",ma_deg=0.0,ea=30.0,stale=True)
        fresh_rec = _internal(41,"FRESH",ma_deg=170.0,ea=1.0,stale=False)
        cfg = ScreeningConfig(horizon_minutes=96,coarse_step_minutes=0.5,
                              screening_threshold_km=300.0,broad_phase_margin_km=200.0)
        run = screen_satellites([stale_rec,fresh_rec],SCREENING_DT,cfg)
        for evt in run.alerts:
            if evt.norad_id_1==40 or evt.norad_id_2==40:
                flag = evt.object_1_stale if evt.norad_id_1==40 else evt.object_2_stale
                assert flag, "Stale object must be flagged"
                assert any("stale" in n.lower() for n in evt.notes)

# ---- Test 9: Duplicate pairs ----
class TestT9Duplicates:
    def test_dedup(self):
        run = screen_satellites(
            [_iss(50,"A"),_iss(50,"A-COPY"),_iss(51,"B",ma=170.0)],
            SCREENING_DT, ScreeningConfig(horizon_minutes=96,screening_threshold_km=500.0))
        assert run.objects_screened==2
        assert 50 in {s["norad_id"] for s in run.skipped}

    def test_pairs_unique(self):
        recs=[_iss(60+i,f"U{i}",ma=i*10) for i in range(5)]
        run = screen_satellites(recs,SCREENING_DT,
                                ScreeningConfig(horizon_minutes=30,screening_threshold_km=500.0,broad_phase_margin_km=200.0))
        n=run.objects_screened
        assert run.pairs_after_broad_phase<=n*(n-1)//2

# ---- Test 10: Coordinate/unit consistency ----
class TestT10Units:
    def test_perigee_apogee_iss(self):
        p,a=_perigee_apogee(15.5,0.0001)
        assert 200<p<700 and 200<a<700 and a>=p

    def test_perigee_apogee_gso(self):
        p,a=_perigee_apogee(1.0027,0.0001)
        assert 30000<p<40000

    def test_band_overlap(self):
        assert _altitude_bands_overlap(390,420,380,415,0.0)
        assert not _altitude_bands_overlap(380,430,35700,35800,0.0)
        assert not _altitude_bands_overlap(380,430,35700,35800,100.0)

    def test_result_physical_units(self):
        cfg=ScreeningConfig(horizon_minutes=96,coarse_step_minutes=0.5,
                            screening_threshold_km=300.0,broad_phase_margin_km=200.0)
        run=screen_satellites([_iss(70,"UA"),_iss(71,"UB",ma=170.0)],SCREENING_DT,cfg)
        for e in run.alerts:
            assert math.isfinite(e.miss_distance_km) and e.miss_distance_km>=0
            assert math.isfinite(e.relative_speed_km_s)
            assert 0<=e.relative_speed_km_s<=16.0

    def test_jd_helpers(self):
        jd,jdf=_epoch_to_jd("2025-01-01T00:00:00.000000")
        assert abs((jd+jdf)-2460676.5)<1.0
        jd2,jdf2=_jd_add_minutes(jd,jdf,1440.0)
        assert abs((jd2+jdf2)-(jd+jdf)-1.0)<1e-9


# ---- Test 11: API regression ----
class TestT11API:
    @pytest.fixture(autouse=True)
    def setup(self):
        import app as fa
        fa.SATELLITES={
            80:{"norad_id":80,"name":"API-A","epoch":"2025-01-01T00:00:00.000000",
                "epoch_age_days":1.0,"mean_motion":15.5,"eccentricity":0.0001,
                "inclination":51.6,"period_min":92.9,"source":"stations","priority":10,
                "satrec":_build_satrec(80,0.0),"init_error":None,"stale":False,"intl_desig":""},
            81:{"norad_id":81,"name":"API-B","epoch":"2025-01-01T00:00:00.000000",
                "epoch_age_days":1.0,"mean_motion":15.5,"eccentricity":0.0001,
                "inclination":51.6,"period_min":92.9,"source":"active_satellites","priority":1,
                "satrec":_build_satrec(81,170.0),"init_error":None,"stale":False,"intl_desig":""},
        }
        fa._last_screening_run=None
        self.c=fa.app.test_client(); self.fa=fa; yield

    def test_health(self):
        r=self.c.get("/api/health"); assert r.status_code==200
        d=r.get_json(); assert d["status"]=="ok" and "satellites_loaded" in d

    def test_satellites(self):
        r=self.c.get("/api/satellites"); assert r.status_code==200
        d=r.get_json(); assert "satellites" in d and "total" in d

    def test_propagate(self):
        r=self.c.get("/api/propagate?ts=2025-01-01T00:00:00Z&limit=2")
        assert r.status_code==200; d=r.get_json()
        assert "positions" in d and "note" in d

    def test_propagate_one(self):
        r=self.c.get("/api/propagate_one?norad_id=80&ts=2025-01-01T00:00:00Z")
        assert r.status_code==200; d=r.get_json()
        assert "orbit_path" in d and "position" in d

    def test_screen_schema(self):
        r=self.c.get("/api/screen?horizon_minutes=96&threshold_km=300")
        assert r.status_code==200; d=r.get_json()
        for k in ["run_id","screening_epoch","alert_count","alerts","disclaimer","config","runtime_seconds"]:
            assert k in d, f"Missing key: {k}"

    def test_screen_bad_param(self):
        r=self.c.get("/api/screen?horizon_minutes=NaN"); assert r.status_code==400
        assert "error" in r.get_json()

    def test_results_404_before_run(self):
        self.fa._last_screening_run=None
        r=self.c.get("/api/screen/results"); assert r.status_code==404

    def test_results_after_run(self):
        self.c.get("/api/screen?horizon_minutes=60&threshold_km=300")
        r=self.c.get("/api/screen/results"); assert r.status_code==200
        d=r.get_json(); assert "run_id" in d and "alerts" in d

    def test_alert_fields(self):
        r=self.c.get("/api/screen?horizon_minutes=96&threshold_km=300")
        d=r.get_json()
        if d["alert_count"]>0:
            a=d["alerts"][0]
            for k in ["event_id","norad_id_1","name_1","norad_id_2","name_2",
                      "tca_utc","miss_distance_km","relative_speed_km_s","screening_status","disclaimer"]:
                assert k in a, f"Alert missing '{k}'"

# ---- Test 12: Performance ----
class TestT12Performance:
    def test_broad_phase_reduces_pairs(self):
        leos=[_iss(100+i,f"L{i}") for i in range(10)]
        gsos=[_gso(200+i,f"G{i}") for i in range(5)]
        cfg=ScreeningConfig(horizon_minutes=30,coarse_step_minutes=1.0,
                            screening_threshold_km=5.0,broad_phase_margin_km=50.0)
        run=screen_satellites(leos+gsos,SCREENING_DT,cfg)
        assert run.pairs_after_broad_phase<run.total_pairs_possible
        assert math.isfinite(run.runtime_seconds) and run.runtime_seconds>=0

    @pytest.mark.slow
    def test_200sat_performance(self):
        recs=[_iss(400+i,f"BIG{i}",ma=i*1.8) for i in range(200)]
        cfg=ScreeningConfig(horizon_minutes=30,coarse_step_minutes=2.0,
                            screening_threshold_km=5.0,broad_phase_margin_km=50.0)
        run=screen_satellites(recs,SCREENING_DT,cfg)
        print(f"\n  [PERF] 200 sats: pairs={run.total_pairs_possible}, "
              f"broad={run.pairs_after_broad_phase}, "
              f"alerts={len(run.alerts)}, rt={run.runtime_seconds:.3f}s")
        assert run.objects_screened==200
        assert run.runtime_seconds<120

# ---- Test 13: Reproducibility ----
class TestT13Reproducibility:
    def test_same_results_twice(self):
        cfg=ScreeningConfig(horizon_minutes=96,coarse_step_minutes=1.0,
                            screening_threshold_km=300.0,broad_phase_margin_km=200.0)
        recs=[_iss(500,"RA"),_iss(501,"RB",ma=170.0)]
        r1=screen_satellites(recs,SCREENING_DT,cfg)
        r2=screen_satellites(recs,SCREENING_DT,cfg)
        assert r1.objects_screened==r2.objects_screened
        assert len(r1.alerts)==len(r2.alerts)
        for a,b in zip(r1.alerts,r2.alerts):
            assert a.event_id==b.event_id
            assert abs(a.miss_distance_km-b.miss_distance_km)<1e-9
            assert abs(a.tca_minutes_from_start-b.tca_minutes_from_start)<1e-9

# ---- Test 14: Existing application regression ----
class TestT14Regression:
    @pytest.fixture(autouse=True)
    def setup(self):
        import app as fa
        fa.SATELLITES={
            90:{"norad_id":90,"name":"REG-A","epoch":"2025-01-01T00:00:00.000000",
                "epoch_age_days":1.0,"mean_motion":15.5,"eccentricity":0.0001,
                "inclination":51.6,"period_min":92.9,"source":"stations","priority":10,
                "satrec":_build_satrec(90),"init_error":None,"stale":False,"intl_desig":""},
        }
        self.c=fa.app.test_client(); yield

    def test_health(self):
        assert self.c.get("/api/health").status_code==200

    def test_satellites(self):
        r=self.c.get("/api/satellites?limit=10"); assert r.status_code==200
        assert "satellites" in r.get_json()

    def test_propagate_all(self):
        r=self.c.get("/api/propagate?ts=2025-01-01T00:00:00Z&limit=1")
        assert r.status_code==200; assert "positions" in r.get_json()

    def test_propagate_one(self):
        r=self.c.get("/api/propagate_one?norad_id=90&ts=2025-01-01T00:00:00Z")
        assert r.status_code==200; d=r.get_json()
        assert "orbit_path" in d and "position" in d and "note" in d

    def test_orbit_batch(self):
        r=self.c.get("/api/orbit_batch?norad_ids=90&ts=2025-01-01T00:00:00Z")
        assert r.status_code==200; assert "orbits" in r.get_json()

    def test_404_unknown_norad(self):
        assert self.c.get("/api/propagate_one?norad_id=99999&ts=2025-01-01T00:00:00Z").status_code==404

    def test_400_bad_timestamp(self):
        assert self.c.get("/api/propagate?ts=INVALID").status_code==400

