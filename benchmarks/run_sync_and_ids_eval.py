"""Reproduce every number in the paper: (1) synchronization/alignment on real data, (2) consequence-aware IDS audit.

  python benchmarks/run_sync_and_ids_eval.py            # sync eval + reuse cached IDS results in results/*.pkl (runs them if missing)
  python benchmarks/run_sync_and_ids_eval.py --rerun    # recompute the IDS evaluations (needs `python scripts/fetch_data.py`; ~10 min)

Writes results/sync_eval.json, results/audit_summary.json (consumed by /api/audit and generate_paper_figures.py), results/cav_alerts.json.
"""
from __future__ import annotations
import argparse, json, pathlib, pickle, sys, time
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from backend import tracks as T, consequence_auditor as C, domains as D, stego_manifest as SM
from backend.sync_engine import SyncEngine
from backend.parsers import j2735_bsm_parser as BSM, ocpp_evse_parser as OCPP, av_sensor_parser as AV
RES = ROOT / "results"; RAW = ROOT / "data" / "raw"


def jdump(obj, path):
    json.dump(obj, open(path, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else (o.tolist() if hasattr(o, "tolist") else str(o)))


# ============================================================ 1. synchronization
def build_world(tid, effect):
    sc = T.scenario(tid); st, out = {"cav": T.build_cav, "ev": T.build_ev, "av": T.build_av, "ctf": T.build_ctf}[sc["domain"]](sc, effect)
    e = SyncEngine(sc["fps"]); [e.add(s) for s in st]; return sc, e, out


def sync_eval() -> dict:
    out = {"tracks": {}, "offset_recovery": [], "reconstruction": {}, "clock_arith": {}}
    for tid in ("track1_cav_bsm", "track2_ev_ocpp", "track3_av_can", "track4_ids_audit", "track5_ctf"):
        sc = T.scenario(tid); g = T.graph(tid); gate = next(n for n in sc["nodes"] if n.get("gate"))["gate"]; k_gate = int(gate["t_ms"] * sc["fps"] // 1000); n = int(sc["duration_ms"] * sc["fps"] // 1000)
        opts = [o["id"] for o in gate["options"]]; digs_pre, digs_all, err = {}, {}, {"interp": 0.0, "hold": 0.0, "nearest": 0.0}; oor = 0
        try:
            for o in opts:
                eff = g.advance(gate_node := next(x["id"] for x in sc["nodes"] if x.get("gate")), o)["effect"]
                _, e, _ = build_world(tid, eff); _, e2, _ = build_world(tid, eff)
                digs_pre[o] = e.digest(0, k_gate - 1); digs_all[o] = (e.digest(0, n), e2.digest(0, n))
                for mode in err:
                    rep = e.alignment_report(0, n, mode); err[mode] = max(err[mode], max(v["max_query_err_ms"] for v in rep.values())); oor += sum(v["out_of_range_frames"] for v in rep.values())
        except FileNotFoundError as ex:
            out["tracks"][tid] = {"skipped": str(ex)}; continue
        out["tracks"][tid] = {"branches": len(opts), "frames": n + 1, "pre_gate_digest_identical_across_branches": len(set(digs_pre.values())) == 1,
                              "repeat_run_bit_identical": all(a == b for a, b in digs_all.values()), "post_gate_digests_distinct": len(set(d[0] for d in digs_all.values())) == len(opts),
                              "max_query_err_ms": err, "out_of_range_frames": oor}
    # clock arithmetic: frame -> ms is exact integer arithmetic; max |ms - ideal| is half a millisecond
    from backend.sync_engine import MasterClock
    for fps in (24, 25, 30, 50, 60):
        c = MasterClock(fps); e = max(abs(c.frame_to_ms(k) - k * 1000.0 / fps) for k in range(0, 36000)); out["clock_arith"][str(fps)] = {"max_rounding_err_ms": e}
    # cross-correlation clock-offset recovery on REAL signals shifted by a known offset
    df = T._ngsim(); L, F = D.cav_window(df, (85, 60, 709, 1036), 300); t = L.t_ms - L.t_ms[0]
    for name, sig in (("NGSIM leader speed", L.cols["speed_mps"]), ("NGSIM follower speed", F.cols["speed_mps"])):
        for off in (-230.0, -137.0, 41.0, 137.0, 220.0):
            est = SyncEngine.estimate_offset_ms(t, sig, t + off, sig, 300); out["offset_recovery"].append({"signal": name, "true_ms": -off, "est_ms": est, "abs_err_ms": abs(est + off)})
    # reconstruction (leave-one-out linear interpolation vs held-out real samples)
    e = SyncEngine(30); e.add(L); e.add(F); out["reconstruction"]["ngsim_leader"] = {c: e.reconstruction_error(L.name, c) for c in ("speed_mps", "local_y_m")}
    fl, _ = T._fleet(); e2 = SyncEngine(30); e2.add(fl.streams[0]); out["reconstruction"]["ev_session_power_1Hz"] = e2.reconstruction_error(fl.streams[0].name, "power_kw")
    if AV.available():
        ox = AV.oxts_stream(); lid = AV.lidar_stream(); n = min(len(ox), len(lid)); d = (lid.t_ms[:n] - ox.t_ms[:n]).astype(float)
        out["kitti_real_clock_offset_ms"] = {"velodyne_minus_oxts_mean": float(d.mean()), "min": float(d.min()), "max": float(d.max())}
        e3 = SyncEngine(30); e3.add(ox); out["reconstruction"]["kitti_oxts_speed"] = e3.reconstruction_error(ox.name, "vf")
    return out


# ============================================================ 2. consequence-aware IDS audit
def ids_eval(rerun: bool) -> dict:
    out = {}
    p = RES / "cav_eval.pkl"
    if rerun or not p.exists():
        t = time.time(); df = BSM.load_ngsim(RAW / "ngsim_peachtree.csv"); pairs = [tuple(x) for x in json.load(open(ROOT / "data" / "cav_pairs.json"))]
        pickle.dump(C.evaluate_cav(df, pairs, folds=9), open(p, "wb")); print("cav", round(time.time() - t), "s")
    cav = pickle.load(open(p, "rb")); out["cav"] = cav["summary"]
    alerts = {}
    for x in cav["records"]:
        if x["attack"] != "none": alerts["-".join(map(str, x["pair"])) + f"|{x['attack']}|{x['t_a']}"] = {d: {k: x[d][k] for k in ("alert_idx", "min_ttc", "collision")} for d in ("physics", "if", "rf", "rf_temporal")}
    jdump(alerts, RES / "cav_alerts.json")
    p = RES / "ev_eval.pkl"
    if rerun or not p.exists(): t = time.time(); pickle.dump(C.evaluate_ev(OCPP.load_norway(RAW / "ev_norway_sessions.csv"), n_windows=6), open(p, "wb")); print("ev", round(time.time() - t), "s")
    ev = pickle.load(open(p, "rb")); out["ev"] = ev["summary"]; out["ev_windows"] = ev["windows"]
    if AV.available():
        p1, p2 = RES / "av_can_eval.pkl", RES / "av_ghost_eval.pkl"
        if rerun or not p1.exists(): pickle.dump(C.evaluate_av_can(AV.can_stream(AV.oxts_stream())), open(p1, "wb"))
        if rerun or not p2.exists(): pickle.dump(C.evaluate_av_ghost([AV.load_points(i) for i in range(len(AV.lidar_stream()))]), open(p2, "wb"))
        out["av_can"] = pickle.load(open(p1, "rb"))["summary"]; g = pickle.load(open(p2, "rb")); out["av_ghost"] = g["rows"]; out["av_ghost_threshold"] = g["physics_threshold"]
    out["config"] = {"PHYS_BSM": C.PHYS_BSM, "PHYS_EV": C.PHYS_EV, "PHYS_AV": C.PHYS_AV, "CACC": D.CACC, "ATTACKS": D.ATTACK_PARAMS, "EV_CONST": D.EV_CONST, "AV_CONST": D.AV_CONST, "persist_messages": C.PERSIST}
    out["data"] = {"cav": "NGSIM Peachtree (US DOT, public domain), 27 leader-follower pairs, 9-fold group-held-out CV",
                   "ev": "Norway EV charging sessions, Zenodo 13896176 (CC BY 4.0); CC-CV power profile reconstructed to conserve the real energy; 6 busy real windows, leave-window-out ML",
                   "av": "KITTI raw 2011_09_26_drive_0001 (CC BY-NC-SA 3.0), CAN signals derived from OXTS, real Velodyne scans"}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--rerun", action="store_true"); a = ap.parse_args(); RES.mkdir(exist_ok=True)
    s = sync_eval(); jdump(s, RES / "sync_eval.json"); print(json.dumps({k: s[k] for k in ("tracks", "clock_arith")}, indent=1, default=str)[:1800])
    r = ids_eval(a.rerun); r["sync"] = s; jdump(r, RES / "audit_summary.json"); print("wrote results/audit_summary.json")
