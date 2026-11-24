"""Synchronization tests on REAL data (bundled NGSIM / Norway samples). Core requirement: deterministic alignment error < 1 ms on every branching path."""
import numpy as np
import pytest
from backend import tracks as T, domains as D
from backend.streams import Stream
from backend.sync_engine import MasterClock, SyncEngine, BranchGraph, BranchError

BUILD = {"cav": T.build_cav, "ev": T.build_ev, "ctf": T.build_ctf}
OFFLINE_TRACKS = ["track1_cav_bsm", "track2_ev_ocpp", "track4_ids_audit", "track5_ctf"]      # track3 needs KITTI (downloaded at run time)


def world(tid, effect):
    sc = T.scenario(tid); st, out = BUILD[sc["domain"]](sc, effect); e = SyncEngine(sc["fps"]); [e.add(s) for s in st]; return sc, e, out


def options(tid):
    g = next(n for n in T.scenario(tid)["nodes"] if n.get("gate"))["gate"]; return g, [o["effect"] for o in g["options"]]


@pytest.mark.parametrize("fps", [24, 25, 30, 50, 60])
def test_clock_integer_exact_and_sub_ms(fps):
    c = MasterClock(fps)
    for k in range(0, 20000):
        assert abs(c.frame_to_ms(k) - k * 1000 / fps) <= 0.5 + 1e-9            # round-half-up: never more than half a ms from ideal
        assert isinstance(c.frame_to_ms(k), int)
        assert c.ms_to_frame(c.frame_to_ms(k)) in (k, k - 1) or fps > 1000
    assert c.frame_to_ms(fps) == 1000                                            # one second of video is exactly 1000 ms (no drift)


@pytest.mark.parametrize("tid", OFFLINE_TRACKS)
def test_alignment_error_below_1ms_on_every_branch(tid):
    sc = T.scenario(tid); n = sc["duration_ms"] * sc["fps"] // 1000; g, effs = options(tid)
    for eff in effs:
        _, e, _ = world(tid, eff)
        rep = e.alignment_report(0, n, "interp")
        for name, r in rep.items():
            assert r["max_query_err_ms"] < 1.0, (tid, eff, name)
            assert r["out_of_range_frames"] == 0


@pytest.mark.parametrize("tid", OFFLINE_TRACKS)
def test_pre_gate_telemetry_identical_across_branches(tid):
    sc = T.scenario(tid); g, effs = options(tid); k_gate = g["t_ms"] * sc["fps"] // 1000
    digs = {e: world(tid, e)[1].digest(0, k_gate - 1) for e in effs}
    assert len(set(digs.values())) == 1, "the decision must not change telemetry that was already shown"


@pytest.mark.parametrize("tid", OFFLINE_TRACKS)
def test_digest_deterministic_across_rebuilds(tid):
    sc = T.scenario(tid); n = sc["duration_ms"] * sc["fps"] // 1000; g, effs = options(tid)
    for eff in effs[:2]: assert world(tid, eff)[1].digest(0, n) == world(tid, eff)[1].digest(0, n)


def test_branches_diverge_after_gate_when_physics_differs():
    sc = T.scenario("track1_cav_bsm"); n = sc["duration_ms"] * sc["fps"] // 1000; g, effs = options("track1_cav_bsm"); k = g["t_ms"] * sc["fps"] // 1000
    d = {e: world("track1_cav_bsm", e)[1].digest(k, n) for e in effs}; assert len(set(d.values())) == len(effs)


def test_columns_bit_identical_to_frame():
    sc, e, _ = world("track2_ev_ocpp", "revert_now"); cols = e.columns(100, 400, 7)
    for i, k in enumerate(cols["k"]):
        f = e.frame(k)
        assert f["t_ms"] == cols["t_ms"][i]
        for s, d in f["samples"].items():
            for c, v in cols["streams"][s].items(): assert d[c] == v[i]


def test_all_streams_share_master_clock_and_frame_to_ms_mapping():
    sc, e, _ = world("track1_cav_bsm", "ignore"); f = e.frame(159)
    assert f["t_ms"] == e.clock.frame_to_ms(159) == 5300
    assert all(s["t_eval_ms"] == 5300.0 for s in f["samples"].values())


def test_offset_estimator_recovers_known_shift_on_real_signal():
    L, F = D.cav_window(T._ngsim(), (85, 60, 709, 1036), 300); t = L.t_ms - L.t_ms[0]
    for off in (-230.0, -137.0, 41.0, 220.0):
        est = SyncEngine.estimate_offset_ms(t, L.cols["speed_mps"], t + off, L.cols["speed_mps"], 300)
        assert abs(est + off) < 1.0


def test_skew_and_offset_are_applied_then_removed():
    t = np.arange(0, 10000, 100); v = np.sin(t / 1000.0)
    s = Stream("x", "bsm", t, {"v": v}); e = SyncEngine(30); e.add(s, offset_ms=40.0)
    assert abs(e.sample("x", 1040.0)["v"] - np.sin(1.0)) < 1e-9                 # sample recorded at 1000 ms on a clock 40 ms behind master
    assert e.streams["x"].master_t()[10] == 1040.0


def test_hold_and_nearest_error_bounded_by_sample_period():
    sc, e, _ = world("track1_cav_bsm", "ignore"); rep_h = e.alignment_report(0, 800, "hold"); rep_n = e.alignment_report(0, 800, "nearest")
    for n_, r in rep_h.items(): assert r["max_query_err_ms"] <= r["sample_period_ms"] + 1
    for n_, r in rep_n.items(): assert r["max_query_err_ms"] <= r["sample_period_ms"] / 2 + 1


def test_leave_one_out_reconstruction_is_small_on_real_speed():
    L, _ = D.cav_window(T._ngsim(), (85, 60, 709, 1036), 300); e = SyncEngine(30); e.add(L)
    assert e.reconstruction_error(L.name, "speed_mps")["mae"] < 0.1               # m/s, vs real held-out 10 Hz samples


# ---------------------------------------------------------------- branching state machine
@pytest.mark.parametrize("tid", ["track1_cav_bsm", "track2_ev_ocpp", "track3_av_can", "track4_ids_audit", "track5_ctf"])
def test_graph_valid_and_every_option_reaches_terminal(tid):
    g = T.graph(tid); gate_node = next(n for n in g.nodes.values() if n.get("gate"))
    for o in gate_node["gate"]["options"]:
        p = g.resolve_path([o["id"]]); assert p["done"] and g.nodes[p["at"]].get("terminal")


def test_invalid_option_and_dangling_edge_rejected():
    g = T.graph("track1_cav_bsm")
    with pytest.raises(BranchError): g.resolve_path(["nope"])
    bad = {"start": "a", "nodes": [{"id": "a", "gate": {"options": [{"id": "x", "next": "ghost"}]}}]}
    with pytest.raises(BranchError): BranchGraph(bad)
    with pytest.raises(BranchError): BranchGraph({"start": "a", "nodes": [{"id": "a", "next": "b"}, {"id": "b", "next": "a"}]})      # no terminal reachable
