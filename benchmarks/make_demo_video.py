"""Render ONE demo video (docs/v2x_twinstream_demo.mp4): Track 1 on real NGSIM data, two branches side by side.
Both panels are identical until the decision gate (5.3 s); then the left follower ignores the verifier alert, the right falls back to ranging."""
import pathlib, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from backend import tracks as T
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "figure.facecolor": "#FFFFFF", "axes.facecolor": "#FFFFFF", "font.size": 11})
B, R, G, GR, O = "#0072B2", "#D55E00", "#009E73", "#6b7480", "#E69F00"
FPS, SC = 15, T.scenario("track1_cav_bsm")
W = {}
for key, eff in (("ignore", "ignore"), ("fallback", "fallback_now")):
    st, out = T.build_cav(SC, eff); W[key] = ({s.name: s for s in st}, out)
ll = W["ignore"][1]["lead_len_m"]; n = len(W["ignore"][0]["follower"].t_ms); GATE, ATT, ALERT = 53, 50, 52
fig = plt.figure(figsize=(12.8, 7.2), dpi=100); gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.25], hspace=0.38, wspace=0.14, left=0.06, right=0.97, top=0.86, bottom=0.08)
frames = []
for i in range(0, n, 1):                                   # data is 10 Hz; render every 2nd sample at 15 fps (~1.33x real time -> slow-mo off)
    fig.clf(); t_s = i / 10.0
    fig.suptitle("V2X-TwinStream  |  Track 1: forged BSM position (+12 m) on real NGSIM traffic", fontsize=15, y=0.965)
    fig.text(0.5, 0.915, f"t = {t_s:4.1f} s   " + ("DECISION GATE  (verifier alert at 5.2 s)" if GATE <= i < GATE + 8 else ("verifier ALERT" if i >= ALERT else ("ATTACK: claimed leader position +12 m" if i >= ATT else "normal operation"))),
             ha="center", fontsize=13, color=R if i >= ATT else "#1b1f24")
    for c, (key, title) in enumerate((("ignore", "Branch A: alert ignored"), ("fallback", "Branch B: immediate fallback to ranging"))):
        S, out = W[key]; f = S["follower"]; lt = S["leader_true"]; lc = S["leader_claimed"]; fy = f.cols["y_m"][i]
        ax = fig.add_subplot(gs[0, c]); ax.set_xlim(-10, 70); ax.set_ylim(-1, 1); ax.axis("off"); ax.set_title(title, fontsize=13, loc="left")
        ax.add_patch(plt.Rectangle((-10, -0.35), 80, 0.7, color="#e9edf1")); ax.plot([-10, 70], [0, 0], color="white", lw=2, ls=(0, (6, 5)))
        car = lambda y, l, col, ls="-", fc=None, yy=0, lab="": (ax.add_patch(plt.Rectangle((y - fy - l, yy - 0.18), l, 0.36, ec=col, fc=fc or col, ls=ls, lw=2)), ax.text(y - fy - l, yy + 0.24, lab, fontsize=9.5))
        car(fy, 4.5, B, lab="follower"); car(lt.cols["y_m"][i], ll, GR, lab="leader (true)", yy=0)
        if abs(lc.cols["y_m"][i] - lt.cols["y_m"][i]) > 0.5: car(lc.cols["y_m"][i], ll, R, ls="--", fc="#fbe3d6", yy=0.6, lab="leader as claimed in BSM")
        gap = f.cols["gap_m"][i]; ax.text(-9, -0.9, f"true gap {gap:5.1f} m", fontsize=13, color=R if gap < 3 else "#1b1f24", weight="bold")
        if gap <= 0: ax.text(30, -0.1, "COLLISION", fontsize=26, color=R, weight="bold", ha="center")
        a2 = fig.add_subplot(gs[1, c]); tt = np.arange(n) / 10.0
        a2.plot(tt, lc.cols["y_m"] - f.cols["y_m"] - ll, color=R, lw=1.6, label="gap claimed in BSM"); a2.plot(tt, f.cols["gap_m"], color=B, lw=1.8, label="true gap (ranging)")
        a2.axhline(0, color="#000", lw=0.6); a2.axvline(ATT / 10, color=R, ls=":", lw=1); a2.axvline(GATE / 10, color=O, ls="--", lw=1); a2.axvline(t_s, color="#1b1f24", lw=1.2)
        a2.text(ATT / 10 + 0.1, a2.get_ylim()[0] if False else -9, "attack", color=R, fontsize=9); a2.text(GATE / 10 + 0.1, 17, "gate", color=O, fontsize=9)
        a2.set_xlim(0, n / 10); a2.set_ylim(-12, 22); a2.set_xlabel("time (s)"); a2.set_ylabel("gap (m)"); a2.grid(True, color="#E3E6EA", lw=0.5); a2.spines[["top", "right"]].set_visible(False)
        if c == 0: a2.legend(loc="upper right", frameon=False, fontsize=9)
        a2.set_title(("min TTC %.1f s, collision: %s" % (out["min_ttc_s"], "YES" if out["collision"] else "no")) if i > n - 3 else "", fontsize=11, loc="left")
    fig.canvas.draw(); fr = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy(); frames += [fr] * (int(FPS * 1.5) if i == GATE else 1)
frames += [frames[-1]] * (FPS * 2)
out = ROOT / "docs" / "v2x_twinstream_demo.mp4"; out.parent.mkdir(exist_ok=True)
w = imageio.get_writer(str(out), fps=FPS, codec="libx264", ffmpeg_params=["-crf", "20"], macro_block_size=16)
[w.append_data(f) for f in frames]; w.close(); print("wrote", out, len(frames), "frames", out.stat().st_size // 1024, "KB")
