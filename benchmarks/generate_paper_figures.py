"""Paper figures from results/audit_summary.json (written by run_sync_and_ids_eval.py).

Style (as specified): strictly #FFFFFF figure AND axes background, Times New Roman (serif fallbacks if absent), 300 DPI,
each figure exported as .svg, .pdf and .jpg into results/figures/. Colours: Okabe-Ito (colour-vision-deficiency safe), fixed detector order.
"""
from __future__ import annotations
import json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]; OUT = ROOT / "results" / "figures"; OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"], "figure.facecolor": "#FFFFFF", "axes.facecolor": "#FFFFFF",
                     "savefig.facecolor": "#FFFFFF", "savefig.edgecolor": "#FFFFFF", "savefig.transparent": False, "figure.dpi": 300, "savefig.dpi": 300, "font.size": 9,
                     "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.7,
                     "axes.grid": True, "grid.color": "#E3E6EA", "grid.linewidth": 0.5, "svg.fonttype": "path", "pdf.fonttype": 42})
DET = [("none", "No defence", "#999999"), ("if", "Isolation Forest", "#E69F00"), ("rf", "Random Forest", "#56B4E9"), ("rf_temporal", "RF + physics features", "#009E73"),
       ("physics", "Physics verifier", "#0072B2"), ("oracle", "Oracle (instant)", "#000000")]
R = json.load(open(ROOT / "results" / "audit_summary.json"))


def save(fig, name, legend_from=None):
    if legend_from is not None:
        h, l = legend_from.get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=min(len(l), 6), frameon=False, fontsize=7.5)
        fig.tight_layout(rect=(0, 0.09, 1, 1))
    else: fig.tight_layout()
    for ext in ("svg", "pdf", "jpg"):
        kw = {"pil_kwargs": {"quality": 95}} if ext == "jpg" else {}
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, facecolor="#FFFFFF", **kw)
    plt.close(fig); print("wrote", name)


def metric(S, det, atk, key):
    if det == "none": return S["no_defence"][atk].get(key)
    if det == "oracle": return S["oracle"][atk].get(key)
    return S[det][atk].get(key)


def grouped(ax, groups, series, vals, ylabel, title, fmt="{:.2f}", logy=False):
    n = len(series); w = 0.8 / n; x = np.arange(len(groups))
    for i, (key, lab, col) in enumerate(series):
        v = [vals(key, g) for g in groups]; v = [np.nan if a is None else a for a in v]
        ax.bar(x + (i - (n - 1) / 2) * w, v, w * 0.92, color=col, label=lab, linewidth=0)
    ax.set_xticks(x); ax.set_xticklabels([g.replace("_", chr(10)) for g in groups], fontsize=7.5); ax.set_ylabel(ylabel); ax.set_title(title, loc="left")
    if logy: ax.set_yscale("log")


# ---- Fig 1: CAV consequence + latency
cav = R["cav"]; atk = ["step_offset", "ramp_drift", "speed_spoof", "stale_replay"]
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.6))
grouped(ax[0], atk, DET, lambda d, a: metric(cav, d, a, "collision_rate"), "Follower collision rate", "(a) Physical consequence")
grouped(ax[1], atk, DET[1:5], lambda d, a: cav[d][a]["median_latency_ms"], "Median detection latency (ms)", "(b) Detection latency", logy=True)
grouped(ax[2], atk, DET[1:5], lambda d, a: cav[d][a]["detect_rate"], "Detection rate", "(c) Detection rate")
save(fig, "fig1_cav_consequence", ax[0])

# ---- Fig 2: benign false alarms (all domains) + CAV min TTC
fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.6))
dets = [("physics", "Physics", "#0072B2"), ("if", "Isolation Forest", "#E69F00"), ("rf", "Random Forest", "#56B4E9")]
ev = R["ev"]; fa = {"CAV (NGSIM)": cav, "EV (OCPP)": ev}
x = np.arange(len(fa)); w = 0.26
for i, (k, lab, c) in enumerate(dets): ax[0].bar(x + (i - 1) * w, [S[k]["benign_false_alarm_rate"] for S in fa.values()], w * 0.92, color=c, label=lab, linewidth=0)
ax[0].set_xticks(x); ax[0].set_xticklabels(fa.keys()); ax[0].set_ylabel("Benign-window false-alarm rate"); ax[0].set_title("(a) False alarms on real benign data", loc="left")
grouped(ax[1], atk, DET, lambda d, a: metric(cav, d, a, "median_min_ttc"), "Median minimum TTC (s)", "(b) Safety margin")
save(fig, "fig2_false_alarms_ttc", ax[1])

# ---- Fig 3: EV
eva = ["current_step", "stealth_meter_fdi", "oscillation"]
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.6)); EVD = [("none", "No defence", "#999999")] + DET[1:2] + DET[2:3] + [("rf_feeder", "RF + feeder feature", "#009E73"), ("physics", "Physics verifier", "#0072B2"), ("oracle", "Oracle", "#000000")]
grouped(ax[0], eva, EVD, lambda d, a: metric(ev, d, a, "detect_rate") if d not in ("none", "oracle") else None, "Detection rate", "(a) Detection rate")
grouped(ax[1], eva, EVD, lambda d, a: metric(ev, d, a, "median_overcurrent_charger_s"), "Overcurrent (charger-s)", "(b) Overcurrent exposure", logy=True)
grouped(ax[2], eva, EVD, lambda d, a: metric(ev, d, a, "median_max_f_dev_hz"), "Max frequency deviation (Hz)", "(c) Island frequency")
save(fig, "fig3_ev_consequence", ax[1])

# ---- Fig 4: AV (CAN + LiDAR ghost)
if "av_can" in R:
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.6)); S = R["av_can"]; mags = ["-5.0", "-1.5"]
    series = [("no_defence", "No defence", "#999999"), ("if", "Isolation Forest", "#E69F00"), ("rf", "Random Forest", "#56B4E9"), ("physics", "Physics verifier", "#0072B2"), ("oracle", "Oracle", "#000000")]
    grouped(ax[0], mags, series, lambda d, m: S[d][m]["median_min_ttc"] if d in ("if", "rf", "physics") else S[d][m]["median_min_ttc"], "Median min. TTC (s)", "(a) CAN brake tamper")
    ax[0].set_xlabel("Injected deceleration (m/s$^2$)"); ax[0].axhline(S["clean_reference_min_ttc"], color="#000000", lw=0.6, ls="--")
    rows = R["av_ghost"]; sp = sorted({r["spacing_m"] for r in rows}); x = np.arange(len(sp)); w = 0.26
    for i, (k, lab, c) in enumerate([("physics", "Free-space check", "#0072B2"), ("if", "Isolation Forest", "#E69F00"), ("rf", "Random Forest", "#56B4E9")]):
        ax[1].bar(x + (i - 1) * w, [next(r["auc"] for r in rows if r["detector"] == k and r["spacing_m"] == s) for s in sp], w * 0.92, color=c, label=lab, linewidth=0)
    ax[1].set_xticks(x); ax[1].set_xticklabels([f"{s} m ({next(r['points'] for r in rows if r['spacing_m'] == s)} pts)" for s in sp]); ax[1].set_ylabel("ROC AUC"); ax[1].set_ylim(0.4, 1.02)
    ax[1].set_xlabel("Ghost-object point spacing"); ax[1].set_title("(b) LiDAR ghost injection (real KITTI scans)", loc="left")
    h, l = ax[0].get_legend_handles_labels(); h2, l2 = ax[1].get_legend_handles_labels(); d = dict(zip(l + l2, h + h2)); fig.legend(list(d.values()), list(d.keys()), loc="lower center", ncol=3, frameon=False, fontsize=7.5); fig.tight_layout(rect=(0, 0.16, 1, 1))
    [fig.savefig(OUT / f"fig4_av.{e}", dpi=300, facecolor="#FFFFFF", **({"pil_kwargs": {"quality": 95}} if e == "jpg" else {})) for e in ("svg", "pdf", "jpg")]; plt.close(fig); print("wrote fig4_av")

# ---- Fig 5: synchronization
sy = R["sync"]; fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5))
off = sy["offset_recovery"]; ax[0].scatter([o["true_ms"] for o in off], [o["est_ms"] for o in off], s=14, color="#0072B2", zorder=3); lim = [-260, 260]; ax[0].plot(lim, lim, color="#000000", lw=0.6, ls="--")
ax[0].set_xlabel("Applied clock offset (ms)"); ax[0].set_ylabel("Estimated offset (ms)"); ax[0].set_title("(a) Offset recovery (real BSM speed)", loc="left")
names = ["ngsim_leader:speed_mps", "ngsim_leader:local_y_m", "ev_session_power_1Hz:", "kitti_oxts_speed:"]; lab = ["NGSIM speed\n(m/s)", "NGSIM position\n(m)", "EV power\n(kW)", "KITTI speed\n(m/s)"]; vals = []
for n in names:
    k, _, c = n.partition(":"); d = sy["reconstruction"].get(k)
    if d is None: vals.append(np.nan)
    else: vals.append((d[c] if c else d)["mae"])
ax[1].bar(range(4), vals, color="#009E73", width=0.6); ax[1].set_xticks(range(4)); ax[1].set_xticklabels(lab, fontsize=7); ax[1].set_ylabel("Leave-one-out MAE"); ax[1].set_title("(b) Interpolation error vs real samples", loc="left")
tr = [t for t, v in sy["tracks"].items() if "skipped" not in v]; modes = ["interp", "nearest", "hold"]; x = np.arange(len(tr)); w = 0.26
for i, (m, c) in enumerate(zip(modes, ["#0072B2", "#E69F00", "#56B4E9"])): ax[2].bar(x + (i - 1) * w, [sy["tracks"][t]["max_query_err_ms"][m] for t in tr], w * 0.92, color=c, label=m, linewidth=0)
ax[2].set_xticks(x); ax[2].set_xticklabels([t.split("_")[0].replace("track", "T") for t in tr]); ax[2].set_ylabel("Max query error (ms)"); ax[2].set_title("(c) Alignment error, all branches", loc="left"); ax[2].legend(frameon=False, title="sampling", title_fontsize=7)
save(fig, "fig5_sync")
