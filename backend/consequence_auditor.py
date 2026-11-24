"""Consequence-aware IDS audit: baseline statistical/ML detectors vs a physics-informed consistency verifier, deterministic replay.

For every (domain, attack, case) both detector families see exactly the same message stream. We report
  * detection rate and CYBER detection latency (ms from first forged message to first sustained alert),
  * message-level precision / recall / F1,
  * PHYSICAL consequence of the attack when the alert (plus actuation delay) triggers the defence: min TTC / collision (CAV, AV),
    feeder voltage deviation / island frequency deviation (EV).
A detector with a high F1 can still leave a worse physical outcome than a lower-F1 detector that fires earlier on the dangerous attack, which
is why latency and consequence are first-class outputs.

Baselines : Isolation Forest (unsupervised, trained on benign data only), Random Forest (supervised, trained on labelled injected attacks from
            OTHER pairs/windows/frames), and ablations that hand the same residual features to the RF ("rf_temporal"/"rf_feeder") so any
            advantage of the verifier is not just a feature-engineering artefact.
Physics   : kinematic feasibility  v_{t+1} = v_t + a_t dt  (BSM), displacement/speed identity ||dp|| = v dt, V2X-vs-ranging gap cross-check (CAV);
            power balance P = V I cos(phi) n_ph, rating, feeder energy balance, CC-CV current shape (EV);
            command-vs-response consistency and line-of-sight/free-space consistency (AV).
"""
from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from .streams import Stream
from . import domains as D
from .parsers import j2735_bsm_parser as BSM, ocpp_evse_parser as OCPP, av_sensor_parser as AV

PERSIST = 3


def first_sustained(flag: np.ndarray, k: int = PERSIST, start: int = 0) -> int | None:
    run = 0
    for i in range(start, len(flag)):
        run = run + 1 if flag[i] else 0
        if run >= k: return i
    return None


def prf(pred: np.ndarray, truth: np.ndarray) -> tuple[float, float, float]:
    tp = int(np.sum(pred & truth)); fp = int(np.sum(pred & ~truth)); fn = int(np.sum(~pred & truth))
    p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


# ====================================================================== CAV
# thresholds calibrated on REAL benign NGSIM statistics (99.99th percentile x ~1.1 margin; NGSIM clips |a| at 3.74 m/s^2 so the kinematic
# residual legitimately reaches ~1.4 m/s); the gap cross-check floor 1.5 m is the assumed ranging-sensor accuracy.
PHYS_BSM = dict(kin_resid_mps=1.5, disp_resid_m=0.30, gap_resid_m=1.5, a_max=9.0)


def bsm_residuals(claimed: Stream, ego: Stream, radar_gap: np.ndarray, lead_len: float) -> dict[str, np.ndarray]:
    c = claimed.cols; n = len(claimed); dt = DT = 0.1
    kin = np.zeros(n); disp = np.zeros(n)
    kin[1:] = np.abs(c["speed_mps"][1:] - (c["speed_mps"][:-1] + c["accel_long"][:-1] * dt))
    disp[1:] = np.abs(np.hypot(np.diff(c["x_m"]), np.diff(c["y_m"])) - c["speed_mps"][:-1] * dt)
    gap = c["local_y_m"] - ego.cols["local_y_m"] - lead_len
    return {"kin": kin, "disp": disp, "gap": np.abs(gap - radar_gap), "claimed_gap": gap, "accel_mag": np.abs(c["accel_long"])}


def physics_bsm(claimed, ego, radar_gap, lead_len, start=0, P=PHYS_BSM) -> dict:
    r = bsm_residuals(claimed, ego, radar_gap, lead_len)
    flags = {"kinematic": (r["kin"] > P["kin_resid_mps"]) | (r["accel_mag"] > P["a_max"]), "pos_speed": r["disp"] > P["disp_resid_m"], "gap_xcheck": r["gap"] > P["gap_resid_m"]}
    any_f = flags["kinematic"] | flags["pos_speed"] | flags["gap_xcheck"]
    i = first_sustained(any_f, start=start)
    why = None if i is None else [k for k, f in flags.items() if f[i]]
    return {"alert_idx": i, "why": why, "flag": any_f}


def bsm_features(claimed, ego, radar_gap, lead_len, temporal=False) -> np.ndarray:
    c = claimed.cols; r = bsm_residuals(claimed, ego, radar_gap, lead_len); dt = 0.1
    X = [c["speed_mps"], c["accel_long"], c["yaw_rate"], c["local_x_m"], r["claimed_gap"], c["speed_mps"] - ego.cols["speed_mps"], ego.cols["speed_mps"]]
    if temporal:
        dv = np.r_[0.0, np.diff(c["speed_mps"])] / dt
        X += [r["kin"], r["disp"], r["gap"], c["accel_long"] - dv]
    return np.column_stack(X)


@dataclass
class MLDetector:
    """Isolation Forest (benign-only) or Random Forest (supervised). The alert threshold is calibrated to `fpr` on benign rows of HELD-OUT
    groups (pairs / windows / frame blocks) so neither baseline is handicapped by an uncalibrated cut-off."""
    kind: str                     # "if" | "rf"
    model: object = None
    thr: float = 0.0

    def fit(self, X, y, groups=None, fpr: float = 0.01):
        if groups is not None:
            ug = np.unique(groups); cal = np.isin(groups, ug[::5]); tr = ~cal
        else:
            tr = np.ones(len(y), bool); cal = np.ones(len(y), bool)
        if self.kind == "if":
            self.model = IsolationForest(n_estimators=200, random_state=0, n_jobs=-1).fit(X[tr & ~y])
            self.thr = float(np.percentile(self.model.decision_function(X[cal & ~y]), 100 * fpr))
        else:
            self.model = RandomForestClassifier(n_estimators=150, max_depth=12, class_weight="balanced", random_state=0, n_jobs=-1).fit(X[tr], y[tr])
            self.thr = max(0.5, float(np.percentile(self.model.predict_proba(X[cal & ~y])[:, 1], 100 * (1 - fpr))))
        return self

    def flags(self, X) -> np.ndarray:
        if self.kind == "if": return self.model.decision_function(X) < self.thr
        return self.model.predict_proba(X)[:, 1] > self.thr

    def score(self, X) -> np.ndarray:
        return -self.model.decision_function(X) if self.kind == "if" else self.model.predict_proba(X)[:, 1]


def cav_cases(df, pairs, starts=(50, 80)):
    cases = []
    for pair in pairs:
        L, F = D.cav_window(df, pair, 300); ll = float(L.cols["length_m"][0]); rg = F.cols["radar_gap_m"] - ll
        cases.append({"pair": pair, "L": L, "F": F, "ll": ll, "rg": rg})
    return cases


def evaluate_cav(df, pairs, starts=(50, 80), folds=9, attacks=D.CAV_ATTACKS, ablation=True) -> dict:
    cases = cav_cases(df, pairs, starts); n = len(cases)
    rec = []; fold = np.arange(n) % folds
    dets = ["physics", "if", "rf"] + (["rf_temporal"] if ablation else [])
    for f in range(folds):
        train = [c for c, k in zip(cases, fold) if k != f]; test = [c for c, k in zip(cases, fold) if k == f]
        Xr, Xt, Y, Gr = [], [], [], []
        for gi, c in enumerate(train):
            clean = c["L"]
            for tmpl, lab in [(clean, np.zeros(len(clean), bool))] + [(D.inject_cav(clean, a, t)[0], D.inject_cav(clean, a, t)[1]) for a in attacks for t in starts]:
                Xr.append(bsm_features(tmpl, c["F"], c["rg"], c["ll"])); Xt.append(bsm_features(tmpl, c["F"], c["rg"], c["ll"], True)); Y.append(lab); Gr.append(np.full(len(lab), gi))
        Xr, Xt, Y, Gr = np.vstack(Xr), np.vstack(Xt), np.concatenate(Y), np.concatenate(Gr)
        ml = {"if": MLDetector("if").fit(Xr, Y, Gr), "rf": MLDetector("rf").fit(Xr, Y, Gr)}
        if ablation: ml["rf_temporal"] = MLDetector("rf").fit(Xt, Y, Gr)
        for c in test:
            L, F, ll, rg = c["L"], c["F"], c["ll"], c["rg"]
            for atk in ("none",) + tuple(attacks):
                for t_a in starts:
                    claimed, lab = (L, np.zeros(len(L), bool)) if atk == "none" else D.inject_cav(L, atk, t_a)
                    if atk == "none" and t_a != starts[0]: continue
                    out = {"pair": c["pair"], "attack": atk, "t_a": t_a, "label": lab}
                    flags = {"physics": physics_bsm(claimed, F, rg, ll)["flag"]}
                    Xr_, Xt_ = bsm_features(claimed, F, rg, ll), bsm_features(claimed, F, rg, ll, True)
                    flags["if"], flags["rf"] = ml["if"].flags(Xr_), ml["rf"].flags(Xr_)
                    if ablation: flags["rf_temporal"] = ml["rf_temporal"].flags(Xt_)
                    for d in dets:
                        i = first_sustained(flags[d])
                        # alerts raised before the attack start are false alarms; alerts at/after start are detections
                        fa = first_sustained(flags[d][: t_a]) if atk != "none" else i
                        det = None if (atk == "none") else first_sustained(flags[d], start=t_a)
                        alert = (i if atk == "none" else det)
                        res = D.follower_sim(L, F, claimed, ll, alert, t_a) if atk != "none" else D.follower_sim(L, F, claimed, ll, None, 100)
                        out[d] = {"alert_idx": alert, "latency_ms": None if (alert is None or atk == "none") else (alert - t_a) * 100.0, "false_alarm": fa is not None,
                                  "pred": flags[d], "min_ttc": res["min_ttc_s"], "collision": res["collision"], "min_gap": res["min_gap_m"]}
                    if atk != "none":
                        for tag, idx in (("no_defence", None), ("oracle", t_a)):
                            res = D.follower_sim(L, F, claimed, ll, idx, t_a); out[tag] = {"min_ttc": res["min_ttc_s"], "collision": res["collision"], "min_gap": res["min_gap_m"]}
                    rec.append(out)
    return {"records": rec, "detectors": dets, "attacks": list(attacks), "summary": summarize_cav(rec, dets, attacks)}


def summarize_cav(rec, dets, attacks) -> dict:
    S = {}
    for d in dets:
        S[d] = {}
        for a in attacks:
            R = [r for r in rec if r["attack"] == a]
            lat = [r[d]["latency_ms"] for r in R if r[d]["latency_ms"] is not None]
            pred = np.concatenate([r[d]["pred"][r["t_a"]:] for r in R]); truth = np.concatenate([r["label"][r["t_a"]:] for r in R])
            p, rc, f1 = prf(pred, truth)
            S[d][a] = {"n": len(R), "detect_rate": len(lat) / len(R), "median_latency_ms": float(np.median(lat)) if lat else None, "p_msg": p, "r_msg": rc, "f1_msg": f1,
                       "collision_rate": float(np.mean([r[d]["collision"] for r in R])), "median_min_ttc": float(np.median([r[d]["min_ttc"] for r in R])),
                       "median_min_gap": float(np.median([r[d]["min_gap"] for r in R]))}
        B = [r for r in rec if r["attack"] == "none"]
        S[d]["benign_false_alarm_rate"] = float(np.mean([r[d]["false_alarm"] for r in B]))
    S["no_defence"] = {a: {"collision_rate": float(np.mean([r["no_defence"]["collision"] for r in rec if r["attack"] == a])),
                           "median_min_ttc": float(np.median([r["no_defence"]["min_ttc"] for r in rec if r["attack"] == a])),
                           "median_min_gap": float(np.median([r["no_defence"]["min_gap"] for r in rec if r["attack"] == a]))} for a in attacks}
    S["oracle"] = {a: {"collision_rate": float(np.mean([r["oracle"]["collision"] for r in rec if r["attack"] == a])),
                       "median_min_ttc": float(np.median([r["oracle"]["min_ttc"] for r in rec if r["attack"] == a])),
                       "median_min_gap": float(np.median([r["oracle"]["min_gap"] for r in rec if r["attack"] == a]))} for a in attacks}
    S["clean_reference"] = {"collision_rate": float(np.mean([r["physics"]["collision"] for r in rec if r["attack"] == "none"])),
                            "median_min_ttc": float(np.median([r["physics"]["min_ttc"] for r in rec if r["attack"] == "none"]))}
    return S


# ====================================================================== EV
PHYS_EV = dict(rating_margin=1.05, identity_tol=0.05, feeder_tol=0.10, reversal_a=2.0, reversal_count=3, reversal_window=12, quorum=3, persist=3)


def ev_features(rep_I, rep_P, rated, phases, feeder_resid=None, feeder=False):
    n, T = rep_I.shape; V = np.where(phases == 1, 230.0, 400.0)[:, None]
    pf = rep_P * 1000.0 / np.maximum(V * rep_I * np.where(phases == 1, 1.0, math.sqrt(3))[:, None], 1e-6)
    cols = [rep_I, rep_P, np.repeat(V, T, 1), pf, rep_I / rated[:, None], np.gradient(np.cumsum(rep_P, 1), axis=1) / np.maximum(rep_P, 1e-3)]
    if feeder: cols.append(np.repeat(feeder_resid[None, :], n, 0))
    return np.stack([c.reshape(-1) for c in cols], 1)


def ev_flags_physics(rep_I, rep_P, rated, phases, P_feeder, P=PHYS_EV):
    n, T = rep_I.shape; V = np.where(phases == 1, 230.0, 400.0)[:, None]; pf = np.where(phases == 1, 1.0, math.sqrt(3))[:, None]
    rating = rep_I > P["rating_margin"] * rated[:, None]
    expect = V * rep_I * pf * OCPP.cos_phi(rep_P / np.maximum(rep_P.max(axis=1, keepdims=True), 1e-6))          # P = V I cos(phi) n_ph
    ident = np.abs(rep_P * 1000.0 - expect) / np.maximum(rep_P * 1000.0, 1.0) > P["identity_tol"]
    bal = np.abs(rep_P.sum(0) - P_feeder) / np.maximum(P_feeder, 1e-3) > P["feeder_tol"]
    dI = np.diff(rep_I, axis=1); sg = np.sign(dI) * (np.abs(dI) > P["reversal_a"])           # CC-CV current is monotone: reversals mean external modulation
    last = np.zeros(n); rev_events = np.zeros((n, T - 1))
    for j in range(T - 1):                                                                    # vectorised over chargers
        s = sg[:, j]; ch = (s != 0)
        rev_events[:, j] = ch & (last != 0) & (s != last); last = np.where(ch, s, last)
    w = P["reversal_window"]; cs = np.cumsum(np.c_[np.zeros(n), rev_events], axis=1)
    cnt = cs[:, 1:] - np.c_[np.zeros((n, w)), cs[:, 1:-w]] if T - 1 > w else cs[:, 1:]
    rev = np.c_[np.zeros(n, bool), cnt >= (P["reversal_count"] - 1)]
    return {"rating": rating.sum(0) >= 1, "identity": ident.sum(0) >= P["quorum"], "feeder_balance": bal, "cc_cv_shape": rev.sum(0) >= 1}


def evaluate_ev(sessions, windows=(("OSL_S", 1800),), n_windows=6, t_a_s=300.0, attacks=D.EV_ATTACKS, quorum=3) -> dict:
    """Build several REAL busy windows (different days) at the location, run attacks, leave-window-out ML."""
    S = [s for s in sessions if s.location == "OSL_S"]; cls = OCPP.rated_class(S)
    chosen = D.busiest_windows(S, cls, 1800, n=n_windows, min_cc=5)
    fleets = []
    for cc, t0 in chosen:
        fl, info = D.fleet_at(S, t0, cls); info["cc_active"] = cc; fleets.append((fl, info))
    rec = []; dets = ["physics", "if", "rf", "rf_feeder"]
    for w in range(len(fleets)):
        train = [f for k, f in enumerate(fleets) if k != w]; test = fleets[w][0]
        Xr, Xf, Y, Gr = [], [], [], []
        for gi, (fl, _) in enumerate(train):
            for atk in ("none",) + tuple(a for a in attacks if a != "stealth_meter_fdi") + ("stealth_meter_fdi",):
                c = D.ev_consequence(fl, "current_step" if atk == "none" else atk, 1e9 if atk == "none" else t_a_s, None, **({"limit_a": 32.0} if atk != "oscillation" else {"limit_a": float(fl.rated_a[0])}))
                T = c["rep_I"].shape[1]; Pf = c["P"].sum(0); resid = (c["rep_P"].sum(0) - Pf) / np.maximum(Pf, 1e-3)
                lab = np.zeros(T, bool) if atk == "none" else (fl.t_s >= t_a_s)
                Xr.append(ev_features(c["rep_I"], c["rep_P"], fl.rated_a, fl.phases)); Xf.append(ev_features(c["rep_I"], c["rep_P"], fl.rated_a, fl.phases, resid, True))
                Y.append(np.tile(lab, len(fl.rated_a))); Gr.append(np.full(T * len(fl.rated_a), gi))
        Xr, Xf, Y, Gr = np.vstack(Xr), np.vstack(Xf), np.concatenate(Y), np.concatenate(Gr)
        ml = {"if": MLDetector("if").fit(Xr, Y, Gr), "rf": MLDetector("rf").fit(Xr, Y, Gr), "rf_feeder": MLDetector("rf").fit(Xf, Y, Gr)}
        for atk in ("none",) + tuple(attacks):
            kw = {"limit_a": 32.0} if atk != "oscillation" else {"limit_a": float(test.rated_a[0])}
            base = D.ev_consequence(test, "current_step", 1e9, None)
            c = D.ev_consequence(test, "current_step" if atk == "none" else atk, 1e9 if atk == "none" else t_a_s, None, **kw)
            T = len(test.t_s); n = len(test.rated_a); Pf = c["P"].sum(0); resid = (c["rep_P"].sum(0) - Pf) / np.maximum(Pf, 1e-3)
            lab = np.zeros(T, bool) if atk == "none" else (test.t_s >= t_a_s)
            fl_p = ev_flags_physics(c["rep_I"], c["rep_P"], test.rated_a, test.phases, Pf)
            flags = {"physics": np.any(np.vstack(list(fl_p.values())), 0)}
            Xr_ = ev_features(c["rep_I"], c["rep_P"], test.rated_a, test.phases); Xf_ = ev_features(c["rep_I"], c["rep_P"], test.rated_a, test.phases, resid, True)
            for d, m, X in (("if", ml["if"], Xr_), ("rf", ml["rf"], Xr_), ("rf_feeder", ml["rf_feeder"], Xf_)):
                flags[d] = m.flags(X).reshape(n, T).sum(0) >= quorum
            out = {"window": fleets[w][1], "attack": atk, "label": lab, "n": n}
            dt = float(test.t_s[1] - test.t_s[0])
            for d in dets:
                i0 = None if atk == "none" else int(round(t_a_s / dt))
                alert = first_sustained(flags[d], k=PHYS_EV["persist"], start=0 if atk == "none" else i0)
                fa = first_sustained(flags[d][: int(round(t_a_s / dt))], k=PHYS_EV["persist"]) is not None if atk != "none" else alert is not None
                det_s = None if alert is None else float(test.t_s[alert])
                cons = D.ev_consequence(test, "current_step" if atk == "none" else atk, 1e9 if atk == "none" else t_a_s, None if atk == "none" else det_s, **kw)
                out[d] = {"alert_idx": alert, "latency_s": None if (alert is None or atk == "none") else det_s - t_a_s, "false_alarm": fa, "pred": flags[d],
                          **{k: cons[k] for k in ("min_voltage_pu", "voltage_dev_pu", "min_f_hz", "max_f_dev_hz", "overcurrent_charger_s", "undervoltage", "f_violation")}}
            if atk != "none":
                for tag, ds in (("no_defence", None), ("oracle", t_a_s)):
                    cons = D.ev_consequence(test, atk, t_a_s, ds, **kw)
                    out[tag] = {k: cons[k] for k in ("min_voltage_pu", "voltage_dev_pu", "min_f_hz", "max_f_dev_hz", "overcurrent_charger_s", "undervoltage", "f_violation")}
            rec.append(out)
    return {"records": rec, "detectors": dets, "attacks": list(attacks), "windows": [i for _, i in fleets], "summary": summarize_ev(rec, dets, attacks, t_a_s, float(fleets[0][0].t_s[1] - fleets[0][0].t_s[0]))}


def summarize_ev(rec, dets, attacks, t_a, dt) -> dict:
    S = {}
    for d in dets:
        S[d] = {}
        for a in attacks:
            R = [r for r in rec if r["attack"] == a]
            lat = [r[d]["latency_s"] * 1000 for r in R if r[d]["latency_s"] is not None]
            pred = np.concatenate([r[d]["pred"][int(t_a / dt):] for r in R]); truth = np.concatenate([r["label"][int(t_a / dt):] for r in R])
            p, rc, f1 = prf(pred, truth)
            S[d][a] = {"n": len(R), "detect_rate": len(lat) / len(R), "median_latency_ms": float(np.median(lat)) if lat else None, "f1_msg": f1,
                       "median_voltage_dev_pu": float(np.median([r[d]["voltage_dev_pu"] for r in R])), "median_max_f_dev_hz": float(np.median([r[d]["max_f_dev_hz"] for r in R])),
                       "median_overcurrent_charger_s": float(np.median([r[d]["overcurrent_charger_s"] for r in R])),
                       "undervoltage_rate": float(np.mean([r[d]["undervoltage"] for r in R])), "f_violation_rate": float(np.mean([r[d]["f_violation"] for r in R]))}
        B = [r for r in rec if r["attack"] == "none"]; S[d]["benign_false_alarm_rate"] = float(np.mean([r[d]["false_alarm"] for r in B]))
    for tag in ("no_defence", "oracle"):
        S[tag] = {a: {"median_voltage_dev_pu": float(np.median([r[tag]["voltage_dev_pu"] for r in rec if r["attack"] == a])),
                      "median_max_f_dev_hz": float(np.median([r[tag]["max_f_dev_hz"] for r in rec if r["attack"] == a])),
                      "median_overcurrent_charger_s": float(np.median([r[tag]["overcurrent_charger_s"] for r in rec if r["attack"] == a]))} for a in attacks}
    return S


# ====================================================================== AV
PHYS_AV = dict(cmd_resid_mps2=1.0, ghost_behind_cfg=(0.15, 0.4, 4.0))


def av_can_features(can: Stream, a_req: np.ndarray) -> np.ndarray:
    jerk = np.r_[0.0, np.diff(a_req)] / 0.1
    return np.column_stack([can.cols["speed_mps"], a_req, can.cols["steer_rad"], can.cols["yaw_rate"], np.minimum(a_req, 0.0), jerk])


def evaluate_av_can(can: Stream, mags=(-5.0, -1.5), starts_test=(20, 35, 50, 65, 80), starts_train=(15, 30, 45, 60, 75, 90), train_mags=(-4.0, -2.5)) -> dict:
    n = len(can); a_plan = can.cols["accel_long"].copy(); rec = []
    Xr, Y, Gr = [], [], []
    def req(k_a, mag):
        a = a_plan.copy(); end = k_a + int(round(D.AV_CONST["can_attack_s"] / 0.1)); a[k_a:end] += mag; return a, np.arange(n) >= k_a, np.arange(n) < end
    Xr.append(av_can_features(can, a_plan)); Y.append(np.zeros(n, bool)); Gr.append(np.arange(n) // 10)
    for m in train_mags:
        for k in starts_train:
            a, _, act = req(k, m); Xr.append(av_can_features(can, a)); Y.append(act & (np.arange(n) >= k)); Gr.append(np.arange(n) // 10)
    Xr, Y, Gr = np.vstack(Xr), np.concatenate(Y), np.concatenate(Gr)
    ml = {"if": MLDetector("if").fit(Xr, Y, Gr), "rf": MLDetector("rf").fit(Xr, Y, Gr)}
    for mag in mags:
        for k in starts_test:
            a, _, act = req(k, mag); lab = act & (np.arange(n) >= k); X = av_can_features(can, a)
            flags = {"physics": np.abs(a - a_plan) > PHYS_AV["cmd_resid_mps2"], "if": ml["if"].flags(X), "rf": ml["rf"].flags(X)}
            out = {"mag": mag, "k_a": k, "label": lab}
            for d, f in flags.items():
                alert = first_sustained(f, start=k); v = D.av_can_tamper_profile(can, k, mag, alert); fo = D.follower_ttc_after_ego_profile(v)
                out[d] = {"alert_idx": alert, "latency_ms": None if alert is None else (alert - k) * 100.0, "pred": f, "min_ttc": fo["min_ttc_s"], "min_gap": fo["min_gap_m"],
                          "false_alarm": first_sustained(f[:k]) is not None}
            for tag, idx in (("no_defence", None), ("oracle", k)):
                fo = D.follower_ttc_after_ego_profile(D.av_can_tamper_profile(can, k, mag, idx)); out[tag] = {"min_ttc": fo["min_ttc_s"], "min_gap": fo["min_gap_m"]}
            out["clean"] = D.follower_ttc_after_ego_profile(can.cols["speed_mps"])["min_ttc_s"]
            rec.append(out)
    S = {}
    for d in ("physics", "if", "rf"):
        S[d] = {}
        for mag in mags:
            R = [r for r in rec if r["mag"] == mag]; lat = [r[d]["latency_ms"] for r in R if r[d]["latency_ms"] is not None]
            p, rc, f1 = prf(np.concatenate([r[d]["pred"][r["k_a"]:] for r in R]), np.concatenate([r["label"][r["k_a"]:] for r in R]))
            S[d][str(mag)] = {"n": len(R), "detect_rate": len(lat) / len(R), "median_latency_ms": float(np.median(lat)) if lat else None, "f1_msg": f1,
                              "median_min_ttc": float(np.median([r[d]["min_ttc"] for r in R])), "median_min_gap": float(np.median([r[d]["min_gap"] for r in R]))}
        S[d]["benign_false_alarm_rate"] = float(np.mean([r[d]["false_alarm"] for r in rec]))
    for tag in ("no_defence", "oracle"):
        S[tag] = {str(m): {"median_min_ttc": float(np.median([r[tag]["min_ttc"] for r in rec if r["mag"] == m]))} for m in mags}
    S["clean_reference_min_ttc"] = float(rec[0]["clean"])
    return {"records": rec, "summary": S}


def _real_scene_score(P, trs, fr, cfg) -> float:
    """Same free-space check applied to the REAL tracked objects of a clean frame (max over objects); 0 if the frame has no checkable object."""
    v = [AV.behind_fraction(P, o, *cfg) for o in (AV.object_points(P, tr, fr) for tr in trs) if len(o) >= 40 and np.linalg.norm(o[:, :2], axis=1).mean() < 30]
    return float(max(v)) if v else 0.0


def evaluate_av_ghost(lidar_frames: list[np.ndarray], spacings=(0.12, 0.5), dists=(12, 16, 20, 24), lats=(-0.5, 0.5), train_frac=0.5, stride=3) -> dict:
    """Frame-level ghost detection on REAL KITTI scans: Isolation Forest / RF on frame statistics vs free-space (line-of-sight) consistency."""
    n = len(lidar_frames); cut = int(n * train_frac); cfg = PHYS_AV["ghost_behind_cfg"]
    F = lambda P: AV.frame_features(P)
    # frame-level score = max free-space violation over the checkable objects; threshold = 99th percentile of REAL clean training frames (1% FPR)
    trs = AV.tracklets(); real = [_real_scene_score(lidar_frames[fr], trs, fr, cfg) for fr in range(0, cut)]
    thr = float(np.percentile(real, 99)); real = [r for r in real if r > 0]
    Xtr, Ytr, Gtr = [], [], []
    for fr in range(0, cut):
        Xtr.append(F(lidar_frames[fr])); Ytr.append(False); Gtr.append(fr // 5)
        for sp in spacings:
            for d in (10, 14, 18, 22, 26):
                inj, g = AV.inject_ghost(lidar_frames[fr], d, 0.0, spacing=sp); Xtr.append(F(inj)); Ytr.append(True); Gtr.append(fr // 5)
    Xtr, Ytr, Gtr = np.array(Xtr), np.array(Ytr), np.array(Gtr)
    ml = {"if": MLDetector("if").fit(Xtr, Ytr, Gtr), "rf": MLDetector("rf").fit(Xtr, Ytr, Gtr)}
    rows = []
    for sp in spacings:
        sc = {"physics": [], "if": [], "rf": []}; y = []; flags = {"physics": [], "if": [], "rf": []}
        for fr in range(cut, n, stride):
            P = lidar_frames[fr]
            for kind in ("clean",) + tuple((d, l) for d in dists for l in lats):
                if kind == "clean": Q, g = P, None
                else: Q, g = AV.inject_ghost(P, kind[0], kind[1], spacing=sp)
                x = F(Q)[None, :]; ph = _real_scene_score(Q, trs, fr, cfg) if g is None else max(AV.behind_fraction(Q, g, *cfg), _real_scene_score(Q, trs, fr, cfg))
                y.append(kind != "clean"); sc["physics"].append(ph); flags["physics"].append(ph > thr)
                for d in ("if", "rf"): sc[d].append(float(ml[d].score(x)[0])); flags[d].append(bool(ml[d].flags(x)[0]))
        y = np.array(y)
        for d in ("physics", "if", "rf"):
            fl = np.array(flags[d]); tp = np.sum(fl & y); fn = np.sum(~fl & y); fp = np.sum(fl & ~y)
            rows.append({"spacing_m": sp, "points": int(len(AV.ghost_points(16, 0, spacing=sp))), "detector": d, "auc": float(roc_auc_score(y, np.array(sc[d]))),
                         "detect_rate": float(tp / (tp + fn)), "false_alarm_rate": float(fp / max(np.sum(~y), 1)), "n_frames": int(len(y))})
    return {"rows": rows, "physics_threshold": thr, "n_real_objects_calibration": len(real)}
