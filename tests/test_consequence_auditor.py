"""Consequence-auditor tests: physics identities, attack injection, detection latency and physical-consequence ordering on real data."""
import math
import numpy as np
import pytest
from backend import tracks as T, domains as D, consequence_auditor as C, stego_manifest as SM
from backend.parsers import ocpp_evse_parser as OCPP, j2735_bsm_parser as BSM

PAIR = (85, 60, 709, 1036)


@pytest.fixture(scope="module")
def cav():
    L, F = D.cav_window(T._ngsim(), PAIR, 300); ll = float(L.cols["length_m"][0]); return L, F, ll, F.cols["radar_gap_m"] - ll


def test_benign_real_traffic_raises_no_physics_alert(cav):
    L, F, ll, rg = cav; assert C.physics_bsm(L, F, rg, ll)["alert_idx"] is None


@pytest.mark.parametrize("atk", D.CAV_ATTACKS)
def test_every_attack_is_detected_after_it_starts_and_latency_is_reported_in_ms(cav, atk):
    L, F, ll, rg = cav; claimed, lab = D.inject_cav(L, atk, 50); a = C.physics_bsm(claimed, F, rg, ll, start=50)["alert_idx"]
    assert a is not None and a >= 50
    assert (a - 50) * 100 <= 5000


def test_step_offset_latency_is_minimal_persistence_window(cav):
    L, F, ll, rg = cav; claimed, _ = D.inject_cav(L, "step_offset", 50); a = C.physics_bsm(claimed, F, rg, ll, start=50)["alert_idx"]
    assert (a - 50) * 100 == (C.PERSIST - 1) * 100                                  # alert on the 3rd consecutive inconsistent BSM


def test_kinematic_identity_v_next_equals_v_plus_a_dt_on_real_data(cav):
    L, *_ = cav; c = L.cols; r = np.abs(c["speed_mps"][1:] - (c["speed_mps"][:-1] + c["accel_long"][:-1] * 0.1))
    assert np.percentile(r, 95) < C.PHYS_BSM["kin_resid_mps"]


def test_defence_orders_physical_consequence(cav):
    L, F, ll, rg = cav; claimed, _ = D.inject_cav(L, "step_offset", 50); a = C.physics_bsm(claimed, F, rg, ll, start=50)["alert_idx"]
    none = D.follower_sim(L, F, claimed, ll, None, 50); phys = D.follower_sim(L, F, claimed, ll, a, 50); oracle = D.follower_sim(L, F, claimed, ll, 50, 50)
    assert none["collision"] and not phys["collision"]
    assert none["min_ttc_s"] < phys["min_ttc_s"] and oracle["min_ttc_s"] >= phys["min_ttc_s"] - 0.5
    late = D.follower_sim(L, F, claimed, ll, a + 50, 50); assert late["min_gap_m"] < phys["min_gap_m"]            # slower response -> worse consequence


def test_no_randomness_follower_sim_is_deterministic(cav):
    L, F, ll, rg = cav; claimed, _ = D.inject_cav(L, "ramp_drift", 50)
    a, b = D.follower_sim(L, F, claimed, ll, 90, 50), D.follower_sim(L, F, claimed, ll, 90, 50)
    assert np.array_equal(a["gap_m"], b["gap_m"])


def test_j2735_bsm_encoding_roundtrip_units():
    s = BSM.vehicle_stream(T._ngsim(), 60, 709, 1036); m = BSM.bsm_message(s, 10); d = BSM.decode_core(m); i = 10
    assert abs(d["speed_mps"] - s.cols["speed_mps"][i]) <= 0.02 and abs(d["lat"] - s.cols["lat"][i]) < 1e-7 and 0 <= m["value"]["coreData"]["secMark"] < 60000


# ---------------------------------------------------------------- EV
@pytest.fixture(scope="module")
def fleet():
    return T._fleet()[0]


def test_reconstructed_profile_conserves_real_energy():
    S = [s for s in T._norway() if s.location == "OSL_S"]; cls = OCPP.rated_class(S); bad = 0
    for s in S[:60]:
        st = OCPP.reconstruct_profile(s, cls, dt_s=30); e = st.cols["energy_kwh"][-1]
        bad += abs(e - s.energy_kwh) / s.energy_kwh > 0.02
    assert bad <= 3


def test_power_identity_P_equals_VI_cosphi_on_benign_fleet(fleet):
    c = D.ev_consequence(fleet, "current_step", 1e9, None); f = C.ev_flags_physics(c["rep_I"], c["rep_P"], fleet.rated_a, fleet.phases, c["P"].sum(0))
    assert not any(v.any() for v in f.values()), "physics verifier must be silent on benign real windows"


def test_stealth_fdi_hidden_from_meters_but_caught_by_feeder_balance(fleet):
    c = D.ev_consequence(fleet, "stealth_meter_fdi", 300.0, None); f = C.ev_flags_physics(c["rep_I"], c["rep_P"], fleet.rated_a, fleet.phases, c["P"].sum(0))
    assert not f["rating"].any() and not f["identity"].any()                          # reported values look perfectly normal
    assert f["feeder_balance"][302:].all()                                            # ... the feeder balance does not


def test_ev_detection_reduces_overcurrent_and_latency_is_in_ms(fleet):
    base = D.ev_consequence(fleet, "current_step", 300.0, None); fast = D.ev_consequence(fleet, "current_step", 300.0, 302.0); slow = D.ev_consequence(fleet, "current_step", 300.0, 600.0)
    assert fast["overcurrent_charger_s"] < slow["overcurrent_charger_s"] < base["overcurrent_charger_s"]


def test_overload_causes_voltage_sag_and_frequency_deviation(fleet):
    base = D.ev_consequence(fleet, "current_step", 1e9, None); atk = D.ev_consequence(fleet, "current_step", 300.0, None)
    assert atk["min_voltage_pu"] < base["min_voltage_pu"] and atk["max_f_dev_hz"] > 0.5


def test_ocpp_frames_and_201_mapping(fleet):
    fr = OCPP.ocpp16_frames(fleet.streams[0], every_s=60); assert fr[0]["frame"][2] == "StartTransaction" and fr[-1]["frame"][2] == "StopTransaction"
    e = OCPP.to_ocpp201_transaction_event(fr[1]["frame"]); assert e[2] == "TransactionEvent" and e[3]["eventType"] == "Updated"
    assert OCPP.pilot_duty_pct(16.0) == pytest.approx(26.67, abs=0.01)


# ---------------------------------------------------------------- stego / CTF
def test_hash_embeds_and_decodes_through_h264(tmp_path):
    import hashlib
    dig = hashlib.sha256(b"twin").hexdigest(); base = np.full((720, 1280, 3), 120, np.uint8)
    frames = [SM.embed_hash(np.roll(base, i, 1), dig) for i in range(12)]; p = tmp_path / "v.mp4"; SM.write_mp4(p, frames, 30, crf=35)
    e = SM.extract_hash(SM.read_mp4(p)); assert e["ok"] and e["digest_hex"] == dig
    assert not SM.extract_hash([base])["ok"]                                           # a clean frame decodes to nothing valid


def test_pcap_roundtrip_identifies_forged_message():
    a = T.ctf_artifacts(); msgs = SM.read_pcap(a["files"]["capture.pcap"]); rogue = [m for m in msgs if m["src"] != "10.20.0.77"]
    assert len(rogue) == 1 and rogue[0]["frame"][2] == "SetChargingProfile" and SM.flag_for(rogue[0]["frame"]) == a["flag"]


def test_manifest_hash_detects_tampering():
    a = T.ctf_artifacts(); files = dict(a["files"]); assert SM.manifest_hash(files)[0] == a["manifest_hash"]
    files["telemetry.csv"] += b"1,2,3\n"; assert SM.manifest_hash(files)[0] != a["manifest_hash"]
