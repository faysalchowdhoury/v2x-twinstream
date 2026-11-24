"""Track runtime: (scenario JSON, decision path) -> synchronized streams + measured outcome.

Everything is computed from real recorded data with the attack injected and the physical consequence simulated by backend.domains. No random
numbers are used anywhere, so a (track, path) pair always yields bit-identical telemetry. Telemetry BEFORE the decision gate is independent of
the chosen option (the effect only changes the response time), which is what the branching-sync tests assert.
"""
from __future__ import annotations
import functools, json, pathlib
import numpy as np
from . import domains as D, consequence_auditor as C, stego_manifest as SM
from .parsers import j2735_bsm_parser as BSM, ocpp_evse_parser as OCPP, av_sensor_parser as AV
from .streams import Stream
from .sync_engine import SyncEngine, BranchGraph, load_scenario

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCEN = ROOT / "scenarios"; SAMPLES = ROOT / "data" / "samples"; RESULTS = ROOT / "results"
INF99 = 99.0


def list_tracks() -> list[dict]:
    out = []
    for p in sorted(SCEN.glob("track*.json")):
        s = load_scenario(p); out.append({k: s[k] for k in ("id", "title", "domain", "summary", "duration_ms", "fps") if k in s} | {"requires": s.get("requires", [])})
    return out


@functools.lru_cache(maxsize=None)
def scenario(track_id: str) -> dict:
    for p in SCEN.glob("track*.json"):
        s = load_scenario(p)
        if s["id"] == track_id: return s
    raise KeyError(track_id)


@functools.lru_cache(maxsize=None)
def graph(track_id: str) -> BranchGraph: return BranchGraph(scenario(track_id))


@functools.lru_cache(maxsize=1)
def _ngsim():
    return BSM.load_ngsim(SAMPLES / "ngsim_cases.csv")


@functools.lru_cache(maxsize=1)
def _norway():
    return OCPP.load_norway(SAMPLES / "ev_norway_OSL_S.csv")


@functools.lru_cache(maxsize=1)
def _alerts():
    return json.load(open(RESULTS / "cav_alerts.json"))


def _mk(name, kind, t_ms, cols, **meta) -> Stream:
    return Stream(name, kind, np.asarray(t_ms, dtype=np.int64), {k: np.nan_to_num(np.asarray(v, float), posinf=INF99, neginf=-INF99) for k, v in cols.items()}, meta)


def _effect_of(sc: dict, path: dict) -> str | None:
    return path["effects"][-1]["effect"] if path["effects"] else None


# ====================================================================== CAV (tracks 1 and 4)
def _cav_world(sc):
    d = sc["data"]; pair = tuple(d["pair"]); L, F = D.cav_window(_ngsim(), pair, 300); ll = float(L.cols["length_m"][0]); rg = F.cols["radar_gap_m"] - ll
    claimed, lab = D.inject_cav(L, d["attack"], d["t_a_idx"]); return pair, L, F, ll, rg, claimed, lab


def cav_detect_idx(sc: dict, effect: str | None) -> int | None:
    d = sc["data"]; pair, L, F, ll, rg, claimed, lab = _cav_world(sc)
    phys = C.physics_bsm(claimed, F, rg, ll, start=d["t_a_idx"])["alert_idx"]; gate_idx = int(sc["nodes"][0]["gate"]["t_ms"] // 100)
    key = "-".join(map(str, pair)) + f"|{d['attack']}|{d['t_a_idx']}"
    table = {"ignore": None, "fallback_now": gate_idx, "wait_confirm": gate_idx + 50,
             "deploy_if": _alerts()[key]["if"]["alert_idx"], "deploy_rf": _alerts()[key]["rf"]["alert_idx"],
             "deploy_rf_temporal": _alerts()[key]["rf_temporal"]["alert_idx"], "deploy_physics": _alerts()[key]["physics"]["alert_idx"]}
    if effect is None: return None
    return table[effect]


def build_cav(sc, effect):
    d = sc["data"]; pair, L, F, ll, rg, claimed, lab = _cav_world(sc); n = len(L); t = np.arange(n) * 100
    det = cav_detect_idx(sc, effect); sim = D.follower_sim(L, F, claimed, ll, det, d["t_a_idx"])
    res = C.bsm_residuals(claimed, F, rg, ll); ph = C.physics_bsm(claimed, F, rg, ll)
    y0 = F.cols["local_y_m"][0]; fy = L.cols["local_y_m"] - ll - sim["gap_m"]
    alert_step = np.zeros(n)
    if sc["id"] == "track4_ids_audit":                                  # the deployed IDS's own alert (measured in the 9-fold group-held-out benchmark)
        if det is not None: alert_step[det:] = 1.0
        ver = {"alert": alert_step, "kin_resid": res["kin"], "disp_resid": res["disp"], "gap_resid": res["gap"]}
    else:
        pa = ph["alert_idx"]
        if pa is not None: alert_step[pa:] = 1.0
        ver = {"alert": alert_step, "kin_resid": res["kin"], "disp_resid": res["disp"], "gap_resid": res["gap"]}
    st = [_mk("leader_true", "bsm", t, {"y_m": L.cols["local_y_m"] - y0, "speed_mps": L.cols["speed_mps"]}, dataset="NGSIM Peachtree"),
          _mk("leader_claimed", "bsm", t, {"y_m": claimed.cols["local_y_m"] - y0, "speed_mps": claimed.cols["speed_mps"], "attack": lab.astype(float)}),
          _mk("follower", "bsm", t, {"y_m": fy - y0, "speed_mps": sim["v_follower"], "accel": sim["accel"], "gap_m": sim["gap_m"], "ttc_s": np.minimum(sim["ttc_s"], INF99)}),
          _mk("verifier", "ids", t, ver)]
    out = {"detect_idx": det, "detect_ms": None if det is None else det * 100, "latency_ms": None if det is None else (det - d["t_a_idx"]) * 100,
           "min_ttc_s": sim["min_ttc_s"], "min_gap_m": sim["min_gap_m"], "collision": sim["collision"], "hard_brake_s": sim["hard_brake_s"],
           "attack": d["attack"], "attack_ms": d["t_a_idx"] * 100, "lead_len_m": ll, "physics_alert_ms": None if ph["alert_idx"] is None else int(ph["alert_idx"] * 100)}
    return st, out


# ====================================================================== EV (track 2 and 5)
@functools.lru_cache(maxsize=1)
def _fleet():
    S = [s for s in _norway() if s.location == "OSL_S"]; cls = OCPP.rated_class(S)
    cc, t0 = D.busiest_windows(S, cls, 1800, n=1, min_cc=5)[0]; fl, info = D.fleet_at(S, t0, cls); info["cc_active"] = cc; return fl, info


def ev_detect_s(sc, effect, attack):
    d = sc["data"]; fl, _ = _fleet(); c = D.ev_consequence(fl, attack, d["t_a_s"], None)
    f = C.ev_flags_physics(c["rep_I"], c["rep_P"], fl.rated_a, fl.phases, c["P"].sum(0)); anyf = np.zeros(len(fl.t_s), bool)
    for v in f.values(): anyf |= v
    i = C.first_sustained(anyf, k=C.PHYS_EV["persist"], start=int(d["t_a_s"])); phys_s = None if i is None else float(fl.t_s[i])
    gate_s = sc["nodes"][0]["gate"]["t_ms"] / 1000.0 * d["time_scale"]
    table = {"ignore": None, "revert_now": gate_s, "wait_confirm": gate_s + 300.0}
    return (None if effect is None else table[effect]), phys_s, f


def build_ev(sc, effect):
    d = sc["data"]; sc_t = d["time_scale"]; fl, info = _fleet(); attack = d["attack"]
    det_s, phys_s, flags = ev_detect_s(sc, effect, attack); c = D.ev_consequence(fl, attack, d["t_a_s"], det_s)
    t = (fl.t_s * 1000.0 / sc_t).astype(np.int64); n = len(fl.t_s); nc = min(len(fl.streams), 10)
    alert = np.zeros(n)
    if phys_s is not None: alert[int(phys_s):] = 1.0
    cur = {f"I_{i + 1}": c["I"][i] for i in range(nc)}; rep = {f"Irep_{i + 1}": c["rep_I"][i] for i in range(nc)}
    st = [_mk("feeder", "ocpp", t, {"V_pu": c["V"], "f_hz": c["f"], "P_kw": c["P"].sum(0), "P_reported_kw": c["rep_P"].sum(0)}, dataset="Norway EV (Zenodo 13896176)"),
          _mk("chargers", "ocpp", t, {**cur, **rep, "rated_a": np.full(n, float(fl.rated_a[0]))}),
          _mk("verifier", "ids", t, {"alert": alert, "feeder_balance": flags["feeder_balance"].astype(float), "rating": flags["rating"].astype(float),
                                      "identity": flags["identity"].astype(float), "cc_cv_shape": flags["cc_cv_shape"].astype(float)})]
    out = {"detect_s": det_s, "latency_ms": None if det_s is None else (det_s - d["t_a_s"]) * 1000.0, "attack": attack, "attack_s": d["t_a_s"], "time_scale": sc_t,
           "min_voltage_pu": c["min_voltage_pu"], "voltage_dev_pu": c["voltage_dev_pu"], "max_f_dev_hz": c["max_f_dev_hz"], "overcurrent_charger_s": c["overcurrent_charger_s"],
           "undervoltage": c["undervoltage"], "f_violation": c["f_violation"], "n_chargers": info["n_chargers"], "window_start": info["window_start"],
           "physics_alert_s": phys_s, "charger_class": info["class"]}
    return st, out


# ====================================================================== AV (track 3)
def _kitti_ok() -> bool: return AV.available()


def build_av(sc, effect):
    d = sc["data"]
    if not _kitti_ok(): raise FileNotFoundError("KITTI raw drive 2011_09_26_drive_0001 not found: run `python scripts/fetch_data.py kitti` (CC BY-NC-SA data, downloaded at run time).")
    ox = AV.oxts_stream(); can = AV.can_stream(ox); n = len(can)
    t = np.arange(n) * 100                       # nominal 10 Hz grid (real OXTS period is ~103 ms; the consequence models use dt = 0.1 s)
    k_a = d["k_a"]; mag = d["decel"]
    a_plan = can.cols["accel_long"].copy(); a_req = a_plan.copy(); end = k_a + int(round(D.AV_CONST["can_attack_s"] / 0.1)); a_req[k_a:end] += mag
    resid = np.abs(a_req - a_plan); flag = resid > C.PHYS_AV["cmd_resid_mps2"]; alert_idx = C.first_sustained(flag, start=k_a)
    gate_idx = int(sc["nodes"][0]["gate"]["t_ms"] // 100)
    det = {"ignore": None, "cancel_now": gate_idx, "wait_confirm": gate_idx + 10}[effect] if effect else None
    v = D.av_can_tamper_profile(can, k_a, mag, det); fo = D.follower_ttc_after_ego_profile(v); al = np.zeros(n)
    if alert_idx is not None: al[alert_idx:] = 1.0
    st = [_mk("can", "can", t, {"speed_plan_mps": can.cols["speed_mps"], "accel_plan": a_plan, "accel_req": a_req, "steer_rad": can.cols["steer_rad"]}, dataset="KITTI raw 0001 (OXTS-derived)"),
          _mk("ego", "can", t, {"speed_mps": v, "accel": np.r_[0.0, np.diff(v)] / 0.1}),
          _mk("follower", "can", t, {"gap_m": fo["gap"], "ttc_s": np.minimum(fo["ttc"], INF99)}),
          _mk("verifier", "ids", t, {"alert": al, "cmd_resid": resid})]
    out = {"detect_idx": det, "latency_ms": None if det is None else (det - k_a) * 100, "min_ttc_s": fo["min_ttc_s"], "min_gap_m": fo["min_gap_m"],
           "attack_ms": k_a * 100, "physics_alert_ms": None if alert_idx is None else alert_idx * 100, "decel_mps2": mag,
           "clean_min_ttc_s": D.follower_ttc_after_ego_profile(can.cols["speed_mps"])["min_ttc_s"]}
    return st, out


def lidar_frame(k: int, max_pts: int = 4000) -> dict:
    """BEV point cloud of real Velodyne frame k (x forward, y left; ground removed), plus tracklet boxes (real KITTI annotations)."""
    P = AV.load_points(int(k)); m = (P[:, 2] > AV.SENSOR_Z + 0.25) & (np.abs(P[:, 0]) < 45) & (np.abs(P[:, 1]) < 25); P = P[m]
    if len(P) > max_pts: P = P[np.linspace(0, len(P) - 1, max_pts).astype(int)]
    boxes = []
    for tr in AV.tracklets():
        o = AV.object_points(AV.load_points(int(k)), tr, int(k))
        if len(o) >= 5: boxes.append({"cx": float(o[:, 0].mean()), "cy": float(o[:, 1].mean()), "n": int(len(o)), "type": tr.get("type", "obj")})
    return {"k": int(k), "points": np.round(P[:, :2], 2).tolist(), "boxes": boxes}


# ====================================================================== CTF (track 5)
def build_ctf(sc, effect):
    d = sc["data"]; fl, info = _fleet(); sc_t = d["time_scale"]; c = D.ev_consequence(fl, "current_step", d["t_a_s"], None); t = (fl.t_s * 1000.0 / sc_t).astype(np.int64)
    st0 = fl.streams[0]; n = len(fl.t_s)
    st = [_mk("session", "ocpp", t, {"power_kw": st0.cols["power_kw"][:n], "current_a": c["I"][0], "energy_kwh": st0.cols["energy_kwh"][:n], "pilot_limit_a": np.full(n, 32.0) * (fl.t_s >= d["t_a_s"]) + np.full(n, float(fl.rated_a[0])) * (fl.t_s < d["t_a_s"])},
                dataset="Norway EV (Zenodo 13896176)"),
          _mk("feeder", "ocpp", t, {"V_pu": c["V"], "P_kw": c["P"].sum(0)})]
    return st, {"verified_first": effect == "extract_and_verify", "attack_s": d["t_a_s"], "time_scale": sc_t}


@functools.lru_cache(maxsize=1)
def ctf_artifacts() -> dict:
    """Deterministic CTF evidence: telemetry CSV + PCAP (WebSocket/OCPP-J) holding ONE forged SetChargingProfile from a rogue address."""
    sc = scenario("track5_ctf"); d = sc["data"]; fl, _ = _fleet(); st = fl.streams[0]
    frames = OCPP.ocpp16_frames(st, tx_id=7, every_s=10)
    forged = OCPP.set_charging_profile(32.0, start_s=0, profile_id=666); forged[1] = "scp-rogue-666"
    frames.append({"t_ms": int(d["t_a_s"] * 1000), "frame": forged}); frames.sort(key=lambda f: f["t_ms"])
    pcap = SM.write_pcap(frames, {"scp-rogue-666"})
    rows = [{"t_s": float(fl.t_s[i]), "power_kw": float(st.cols["power_kw"][i]), "current_a": float(st.cols["current_a"][i]), "energy_kwh": float(st.cols["energy_kwh"][i])} for i in range(0, len(fl.t_s), 5)]
    files = {"telemetry.csv": SM.telemetry_csv(rows), "capture.pcap": pcap}
    h, per = SM.manifest_hash(files)
    return {"files": files, "manifest_hash": h, "per_file": per, "flag": SM.flag_for(forged), "forged_uid": "scp-rogue-666"}


def ctf_frames(n: int = 90, fps: int = 30):
    """Procedurally drawn 'dashcam + HUD' frames, hash strip embedded (pure NumPy, no fonts needed)."""
    h = ctf_artifacts()["manifest_hash"]; H, W = 720, 1280; out = []
    yy, xx = np.mgrid[0:H, 0:W]
    for i in range(n):
        f = np.zeros((H, W, 3), np.uint8); f[...] = (np.clip(30 + 80 * yy / H, 0, 255)[..., None] * np.array([0.5, 0.6, 0.9])).astype(np.uint8)
        road = (yy > 300) & (np.abs(xx - W / 2) < 80 + (yy - 300) * 1.2); f[road] = (60, 60, 64)
        dash = (yy > 300) & (np.abs(xx - W / 2) < 3) & (((yy - 300 + i * 12) // 24) % 2 == 0); f[dash] = (230, 220, 90)
        bar = int(900 * (i + 1) / n); f[40:60, 40:40 + bar] = (80, 200, 120)
        out.append(SM.embed_hash(f, h))
    return out


if __name__ == "__main__":
    import sys
    print(json.dumps(list_tracks(), indent=1))
