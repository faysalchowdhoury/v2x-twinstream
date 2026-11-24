"""Attack injection (on REAL recorded data) and physical-consequence models for the three TwinStream domains.

Everything here is a deterministic function of (real data, attack parameters, detection time): no random numbers are drawn anywhere.
Attacks are injected into recorded real traffic/charging/sensor data (the standard way to evaluate misbehaviour detectors); the injected
portion is always labelled. Model constants (controller gains, feeder impedance, island inertia ...) are declared at the top of each section.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np
from .streams import Stream
from .parsers import j2735_bsm_parser as BSM, ocpp_evse_parser as OCPP, av_sensor_parser as AV

DT = 0.1


# ============================================================ CAV: leader BSM false-data injection -> follower CACC
CACC = dict(d0=2.0, T=1.2, k_gap=0.23, k_v=0.8, a_min=-6.0, a_max=2.5, ttc_brake=2.0, actuation_delay_s=0.5)
CAV_ATTACKS = ("step_offset", "ramp_drift", "speed_spoof", "stale_replay")
ATTACK_PARAMS = dict(step_offset_m=12.0, ramp_rate_mps=0.5, ramp_cap_m=15.0, speed_spoof_mps=4.0, replay_delay_s=3.0)


@dataclass
class CavCase:
    pair: tuple; attack: str; t_a_idx: int; true_lead: Stream; follower: Stream; claimed: Stream; lead_len: float; label: np.ndarray


def cav_window(df, pair, max_frames=600):
    f, l, a, b = pair
    L, F = BSM.vehicle_stream(df, l, a, b), BSM.vehicle_stream(df, f, a, b)
    t0 = max(L.t_ms[0], F.t_ms[0]); t1 = min(L.t_ms[-1], F.t_ms[-1])
    L, F = L.slice_ms(t0, t1), F.slice_ms(t0, t1)
    n = min(len(L), len(F), max_frames)
    for s in (L, F):
        s.t_ms = s.t_ms[:n]; s.cols = {k: v[:n] for k, v in s.cols.items()}
    return L, F


def inject_cav(L: Stream, attack: str, t_a_idx: int, P=ATTACK_PARAMS) -> tuple[Stream, np.ndarray]:
    """Return the CLAIMED leader stream (what the follower receives in BSMs) and the per-message attack label."""
    c = {k: v.copy() for k, v in L.cols.items()}; n = len(L); lab = np.zeros(n, bool); i0 = t_a_idx
    hd = np.radians(c["heading_deg"])
    def shift(i, d):                       # move claimed position forward along the heading by d metres
        c["local_y_m"][i] += d; c["x_m"][i] += d * np.sin(hd[i]); c["y_m"][i] += d * np.cos(hd[i])
    if attack == "step_offset":
        for i in range(i0, n): shift(i, P["step_offset_m"]); lab[i] = True
    elif attack == "ramp_drift":
        for i in range(i0, n): shift(i, min(P["ramp_rate_mps"] * (i - i0) * DT, P["ramp_cap_m"])); lab[i] = True
    elif attack == "speed_spoof":
        for i in range(i0, n): c["speed_mps"][i] += P["speed_spoof_mps"]; lab[i] = True
    elif attack == "stale_replay":
        k = int(P["replay_delay_s"] / DT)
        for i in range(max(i0, k), n):
            for key in ("local_y_m", "x_m", "y_m", "speed_mps", "accel_long", "heading_deg", "yaw_rate"): c[key][i] = L.cols[key][i - k]
            lab[i] = True
    else:
        raise ValueError(attack)
    return Stream(L.name + "*", "bsm", L.t_ms.copy(), c, {**L.meta, "attack": attack, "claimed": True}), lab


def follower_sim(true_lead: Stream, follower: Stream, claimed: Stream, lead_len: float, detect_idx: int | None = None, start_idx: int = 0) -> dict:
    """Car-following (CACC-style) follower driven by the leader's CLAIMED BSM; after detection (+actuation delay) it falls back to
    onboard ranging (true gap, true lead speed). Returns the true physical outcome (gap, TTC) against the TRUE leader."""
    n = len(true_lead); C = CACC
    y = follower.cols["local_y_m"][0]; v = follower.cols["speed_mps"][0]
    gap = np.zeros(n); vv = np.zeros(n); acc = np.zeros(n); ttc = np.full(n, np.inf)
    sw = None if detect_idx is None else detect_idx + int(round(C["actuation_delay_s"] / DT))
    for i in range(n):
        g_true = true_lead.cols["local_y_m"][i] - y - lead_len; vl_true = true_lead.cols["speed_mps"][i]
        radar = sw is not None and i >= sw
        g = g_true if radar else claimed.cols["local_y_m"][i] - y - lead_len
        vl = vl_true if radar else claimed.cols["speed_mps"][i]
        a = C["k_gap"] * (g - (C["d0"] + C["T"] * v)) + C["k_v"] * (vl - v)
        closing = v - vl
        if closing > 0.1 and g / closing < C["ttc_brake"]: a = C["a_min"]
        a = float(np.clip(a, C["a_min"], C["a_max"]))
        gap[i], vv[i], acc[i] = g_true, v, a
        if v > vl_true + 0.1: ttc[i] = g_true / (v - vl_true) if g_true > 0 else 0.0
        v = max(0.0, v + a * DT); y += v * DT
    s = slice(start_idx, n)
    return {"gap_m": gap, "v_follower": vv, "accel": acc, "ttc_s": ttc, "min_ttc_s": float(np.min(ttc[s])), "min_gap_m": float(np.min(gap[s])),
            "collision": bool(np.min(gap[s]) <= 0.0), "hard_brake_s": float(np.sum(acc[s] <= -5.9) * DT)}


# ============================================================ EV: OCPP charging-current spoofing -> feeder / island
EV_CONST = dict(s_base_kva=100.0, r_pu=0.04, x_pu=0.03, h_s=5.0, d_pu=1.0, droop_r=0.05, tg_s=2.0, f_nom=50.0,
                obc_max_a=32.0, reaction_cmd_delay_s=2.0, dt=1.0, v_low_pu=0.95, f_dev_limit_hz=1.0)
EV_ATTACKS = ("current_step", "stealth_meter_fdi", "oscillation")


def island_natural_period_s() -> float:
    """Natural period of the single-machine island frequency model (so the oscillation attack can hit the resonance)."""
    C = EV_CONST; a = 1.0 / (2 * C["h_s"]); d = C["d_pu"] / (2 * C["h_s"]); k = 1.0 / (C["droop_r"] * 2 * C["h_s"] * C["tg_s"])
    wn = math.sqrt(max(k + 0.0, 1e-9) + d / C["tg_s"])
    return 2 * math.pi / wn


@dataclass
class EvFleet:
    streams: list[Stream]; rated_a: np.ndarray; phases: np.ndarray; t_s: np.ndarray; plug_offset_s: np.ndarray; cls: str


def busiest_windows(S, cls, window_s=1800, n=6, min_cc=5, one_per_day=True):
    """Real windows with the most chargers in their constant-current phase at the window start (vectorised search over all 30-min starts)."""
    import pandas as pd
    t_in = np.array([s.plugin.value for s in S], dtype=np.int64) / 1e9; t_out = np.array([s.plugout.value for s in S], dtype=np.int64) / 1e9
    ok_rate = np.array([OCPP.peak_power_kw(s, cls) <= 1.01 * OCPP.rated_power_kw(cls) for s in S])
    E = np.array([s.energy_kwh for s in S]); hrs = (t_out - t_in) / 3600.0
    t_cc_end = t_in + 0.8 * E / OCPP.rated_power_kw(cls) * 3600.0
    starts = np.unique(np.floor(t_in / 1800.0) * 1800.0); out = []
    for t0 in starts:
        act = (t_in <= t0 + 300) & (t_out >= t0 + window_s) & (hrs < 16) & ok_rate
        out.append((int(np.sum(act & (t0 < t_cc_end))), float(t0)))
    out.sort(key=lambda x: -x[0]); chosen, days = [], set()
    for cc, t0 in out:
        ts = pd.Timestamp(t0, unit="s")
        if cc < min_cc or (one_per_day and ts.date() in days): continue
        chosen.append((cc, ts)); days.add(ts.date())
        if len(chosen) >= n: break
    return chosen


def fleet_at(S, t0, cls, window_s=1800, dt_s=1):
    import pandas as pd
    t1 = t0 + pd.Timedelta(seconds=window_s)
    act0 = [s for s in S if s.plugin <= t0 + pd.Timedelta(seconds=300) and s.plugout >= t1 and s.hours < 16]
    cap = OCPP.rated_power_kw(cls)                                   # sessions whose real average power exceeds the class rating imply a bigger charger: excluded
    act = [s for s in act0 if OCPP.peak_power_kw(s, cls) <= 1.01 * cap]
    ts = np.arange(0, window_s + 1e-9, dt_s); streams = []; rated = []; ph = []; poff = []
    for s in act:
        off = max((t0 - s.plugin).total_seconds(), 0.0)
        st = OCPP.reconstruct_profile(s, cls, dt_s=dt_s, t_start_s=off, t_len_s=window_s)
        if len(st) < len(ts): continue
        st = Stream(st.name, "ocpp", (ts * 1000).astype(np.int64), {k: v[:len(ts)] for k, v in st.cols.items()}, st.meta)
        streams.append(st); rated.append(st.meta["rated_a"]); ph.append(st.meta["phases"]); poff.append(off)
    return EvFleet(streams, np.array(rated), np.array(ph), ts, np.array(poff), cls), {"location": S[0].location, "window_start": str(t0), "n_chargers": len(streams), "n_excluded_over_rating": len(act0) - len(act), "class": cls}


def build_fleet(sessions, window_s=1800, dt_s=1, location="OSL_S", min_active=5):
    S = [s for s in sessions if s.location == location]; cls = OCPP.rated_class(S)
    cc, t0 = busiest_windows(S, cls, window_s, n=1, min_cc=min_active)[0]
    fl, info = fleet_at(S, t0, cls, window_s, dt_s); info["cc_active"] = cc
    return fl, info


def fleet_arrays(fl: EvFleet):
    I = np.array([s.cols["current_a"] for s in fl.streams]); P = np.array([s.cols["power_kw"] for s in fl.streams])
    return I, P


def ev_attack_physical(fl: EvFleet, attack: str, t_a_s: float, mitigate_at_s: float | None = None, limit_a: float = 32.0, period_s: float | None = None):
    """True physical per-charger current/power (A, kW) under the attack, and what MeterValues REPORT.
    Chargers in the CC phase follow the (spoofed) SetChargingProfile limit up to the vehicle on-board-charger maximum; chargers in CV taper
    draw less than any limit so are unaffected. Mitigation (re-issued profile at the rated limit) ends the effect."""
    C = EV_CONST; I0, P0 = fleet_arrays(fl); n, T = I0.shape
    ts = fl.t_s
    cc = np.isclose(P0, P0.max(axis=1, keepdims=True), rtol=0.02)                             # CC phase = at the session's plateau power
    per = period_s or island_natural_period_s()
    V = np.where(fl.phases == 1, 230.0, 400.0)[:, None]; pf = np.where(fl.phases == 1, 1.0, math.sqrt(3))[:, None]
    active = (ts >= t_a_s) & ((ts < mitigate_at_s) if mitigate_at_s is not None else True)
    if attack in ("current_step", "stealth_meter_fdi"): lim = np.full(len(ts), limit_a)
    elif attack == "oscillation": lim = np.where(((ts - t_a_s) % per) < per / 2, limit_a, 6.0)
    else: raise ValueError(attack)
    ia = np.minimum(lim, C["obc_max_a"])[None, :]
    hit = cc & active[None, :]
    I = np.where(hit, ia, I0); P = np.where(hit, V * ia * pf * OCPP.cos_phi(1.0) / 1000.0, P0)
    rep_I, rep_P = (I0, P0) if attack == "stealth_meter_fdi" else (I, P)
    return {"I": I, "P": P, "rep_I": rep_I.copy(), "rep_P": rep_P.copy(), "rated": fl.rated_a, "cc": cc, "period_s": per}


def feeder_voltage(P_kw: np.ndarray) -> np.ndarray:
    C = EV_CONST; p = P_kw.sum(0) / C["s_base_kva"]; q = p * math.tan(math.acos(0.98))
    return 1.0 - (C["r_pu"] * p + C["x_pu"] * q)


def island_frequency(P_kw: np.ndarray, P_ref_kw: np.ndarray) -> np.ndarray:
    """Single-machine island: 2H d(df)/dt = dPm - dPL - D df ; Tg dPm/dt = -df/R - dPm (per unit on S_base). Returns f (Hz)."""
    C = EV_CONST; dpl = (P_kw.sum(0) - P_ref_kw.sum(0)) / C["s_base_kva"]
    df = 0.0; dpm = 0.0; out = np.zeros(len(dpl)); h = 0.1; steps = int(C["dt"] / h)
    for j, x in enumerate(dpl):
        for _ in range(steps):
            ddf = (dpm - x - C["d_pu"] * df) / (2 * C["h_s"]); ddpm = (-df / C["droop_r"] - dpm) / C["tg_s"]
            df += ddf * h; dpm += ddpm * h
        out[j] = C["f_nom"] * (1 + df)
    return out


def ev_consequence(fl: EvFleet, attack: str, t_a_s: float, detect_s: float | None, **kw) -> dict:
    C = EV_CONST
    mit = None if detect_s is None else detect_s + C["reaction_cmd_delay_s"]
    r = ev_attack_physical(fl, attack, t_a_s, mit, **kw); base = ev_attack_physical(fl, "current_step", 1e9)
    V = feeder_voltage(r["P"]); f = island_frequency(r["P"], base["P"]); ts = fl.t_s; post = ts >= t_a_s
    if not post.any(): post = np.ones(len(ts), bool)
    over = (r["I"] > 1.05 * r["rated"][:, None]) & post[None, :]
    return {"min_voltage_pu": float(V[post].min()), "voltage_dev_pu": float(1.0 - V[post].min()), "min_f_hz": float(f[post].min()), "max_f_hz": float(f[post].max()),
            "max_f_dev_hz": float(np.max(np.abs(f[post] - C["f_nom"]))), "overcurrent_charger_s": float(over.sum() * (ts[1] - ts[0])),
            "undervoltage": bool(V[post].min() < C["v_low_pu"]), "f_violation": bool(np.max(np.abs(f[post] - C["f_nom"])) > C["f_dev_limit_hz"]),
            "V": V, "f": f, "P": r["P"], "I": r["I"], "rep_P": r["rep_P"], "rep_I": r["rep_I"], "period_s": r["period_s"], "cc": r["cc"]}


# ============================================================ AV: LiDAR ghost injection & CAN arbitration tamper
AV_CONST = dict(aeb_decel=6.0, ghost_range_m=20.0, aeb_persist_frames=2, follower_headway_s=1.5, follower_reaction_s=1.2, follower_decel=4.0,
                can_attack_s=1.5, can_actuation_delay_s=0.2)


def follower_ttc_after_ego_profile(v_ego: np.ndarray, dt=DT) -> dict:
    """A human-driven following car (constant time gap 1.5 s at t=0, 1.2 s reaction, then 4 m/s^2 braking) behind the AV."""
    C = AV_CONST; n = len(v_ego); y_e = 0.0; y_f = -(C["follower_headway_s"] * v_ego[0] + 4.5); vf = v_ego[0]
    ttc = np.full(n, np.inf); gap = np.zeros(n)
    for i in range(n):
        gap[i] = y_e - y_f - 4.5
        if vf > v_ego[i] + 0.05: ttc[i] = gap[i] / (vf - v_ego[i]) if gap[i] > 0 else 0.0
        braking = i * dt >= C["follower_reaction_s"] and (v_ego[i] < vf - 1.0 or ttc[i] < 4.0)
        a_f = -C["follower_decel"] if (braking and vf > 0) else 0.0
        vf = max(0.0, vf + a_f * dt); y_f += vf * dt; y_e += v_ego[i] * dt
    return {"min_ttc_s": float(ttc.min()), "min_gap_m": float(gap.min()), "ttc": ttc, "gap": gap}


def _apply_decel(v: np.ndarray, i0: int, i1: int, decel: float) -> np.ndarray:
    """Recorded ego speed with an extra deceleration `decel` (m/s^2) applied on frames [i0, i1); the recorded natural speed changes are kept."""
    out = v.copy()
    for i in range(1, len(v)):
        out[i] = max(0.0, out[i - 1] + (v[i] - v[i - 1]) + (decel * DT if i0 <= i < i1 else 0.0))
    return out


def av_can_tamper_profile(can: Stream, k_a: int, decel: float, detect_idx: int | None) -> np.ndarray:
    """Ego speed when CAN brake-arbitration frames request `decel` (negative) for 1.5 s from frame k_a; detection cancels the override
    after the actuation delay."""
    C = AV_CONST; end = k_a + int(round(C["can_attack_s"] / DT))
    if detect_idx is not None: end = min(end, detect_idx + int(round(C["can_actuation_delay_s"] / DT)))
    return _apply_decel(can.cols["speed_mps"], k_a, end, decel)


def av_ghost_profile(can: Stream, k_a: int, k_len: int, detect_idx: int | None) -> np.ndarray:
    """Ego speed when a ghost obstacle injected at k_a for k_len frames triggers AEB (6 m/s^2) after 2 frames; if the ghost is detected the
    AEB request is cancelled after the actuation delay."""
    C = AV_CONST; i0 = k_a + C["aeb_persist_frames"]; end = k_a + k_len
    if detect_idx is not None: end = min(end, detect_idx + int(round(C["can_actuation_delay_s"] / DT)))
    return _apply_decel(can.cols["speed_mps"], i0, end, -C["aeb_decel"])
