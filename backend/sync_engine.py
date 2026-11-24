"""Deterministic millisecond-level alignment of video frames with BSM / OCPP / CAN / LiDAR telemetry, plus a branching scenario state machine.

Master clock
------------
Video frame k at `fps` maps to an INTEGER millisecond  t_k = t0 + floor((2*k*1000 + fps) / (2*fps))  (round-half-up, exact integer arithmetic,
no floating-point drift). Every telemetry stream is first moved to the master clock (per-stream offset + skew), then evaluated at t_k.

Alignment error definitions (all reported by `alignment_report`)
  * query error     |t_k - t_eval|  where t_eval is the instant the returned value actually corresponds to (0 for interpolation; the sample age
                    for zero-order hold; distance to the sample for nearest-neighbour).
  * reconstruction  leave-one-out error of linear interpolation against the real held-out samples (physical units).
  * clock estimate  error of the cross-correlation clock-offset estimator on REAL signals shifted by a known offset.
Determinism: the same (scenario, branch path, frame range) always yields bit-identical output (sha256 digest, see `digest`).
"""
from __future__ import annotations
import hashlib, json, pathlib
import numpy as np
from .streams import Stream


class MasterClock:
    def __init__(self, fps: int = 30, t0_ms: int = 0):
        assert fps > 0
        self.fps, self.t0 = int(fps), int(t0_ms)

    def frame_to_ms(self, k: int) -> int:
        return self.t0 + (2 * int(k) * 1000 + self.fps) // (2 * self.fps)

    def ms_to_frame(self, t_ms: float) -> int:
        return int(((t_ms - self.t0) * self.fps) // 1000)

    def frames_in(self, t_a: int, t_b: int) -> range:
        return range(self.ms_to_frame(t_a), self.ms_to_frame(t_b) + 1)


class SyncEngine:
    def __init__(self, fps: int = 30, t0_ms: int = 0):
        self.clock = MasterClock(fps, t0_ms)
        self.streams: dict[str, Stream] = {}

    def add(self, s: Stream, offset_ms: float | None = None, skew_ppm: float | None = None) -> "SyncEngine":
        if offset_ms is not None: s.clock_offset_ms = float(offset_ms)
        if skew_ppm is not None: s.skew_ppm = float(skew_ppm)
        self.streams[s.name] = s
        return self

    # ------------------------------------------------------------------ sampling
    def sample(self, name: str, t_ms: float, mode: str = "interp") -> dict:
        s = self.streams[name]; tm = s.master_t()
        if mode == "interp":
            if t_ms < tm[0] or t_ms > tm[-1]:
                i = 0 if t_ms < tm[0] else len(tm) - 1
                return {"in_range": False, "t_eval_ms": float(tm[i]), "err_ms": abs(float(tm[i]) - t_ms), **{k: float(v[i]) for k, v in s.cols.items()}}
            vals = {k: float(np.interp(t_ms, tm, v)) for k, v in s.cols.items()}
            return {"in_range": True, "t_eval_ms": float(t_ms), "err_ms": 0.0, **vals}
        i = int(np.searchsorted(tm, t_ms, side="right") - 1) if mode == "hold" else int(np.argmin(np.abs(tm - t_ms)))
        i = max(0, min(i, len(tm) - 1))
        return {"in_range": bool(tm[0] <= t_ms <= tm[-1]), "t_eval_ms": float(tm[i]), "err_ms": abs(float(tm[i]) - t_ms), "index": i,
                **{k: float(v[i]) for k, v in s.cols.items()}}

    def frame(self, k: int, mode: str = "interp") -> dict:
        t = self.clock.frame_to_ms(k)
        return {"frame": int(k), "t_ms": int(t), "samples": {n: self.sample(n, t, mode) for n in self.streams}}

    def frames(self, k0: int, k1: int, mode: str = "interp", step: int = 1) -> list[dict]:
        return [self.frame(k, mode) for k in range(k0, k1 + 1, step)]

    def columns(self, k0: int, k1: int, step: int = 1) -> dict:
        """Columnar bulk sampling (linear interpolation, edge values held outside the stream) for frames k0..k1. Uses exactly the same
        np.interp as `sample`, so values are bit-identical to `frame(k)['samples']`."""
        ks = list(range(k0, k1 + 1, step)); t = np.array([self.clock.frame_to_ms(k) for k in ks], dtype=float)
        out = {"k": ks, "t_ms": [int(x) for x in t], "streams": {}}
        for n, s in self.streams.items():
            tm = s.master_t(); out["streams"][n] = {c: np.interp(t, tm, v).tolist() for c, v in s.cols.items()}
        return out

    def digest(self, k0: int, k1: int, mode: str = "interp") -> str:
        h = hashlib.sha256()
        for f in self.frames(k0, k1, mode):
            h.update(json.dumps(f, sort_keys=True, separators=(",", ":")).encode())
        return h.hexdigest()

    # ------------------------------------------------------------------ clock estimation / reports
    @staticmethod
    def estimate_offset_ms(ref_t, ref_v, oth_t, oth_v, max_lag_ms: int = 300) -> float:
        """Offset to ADD to `oth` timestamps so that it lines up with `ref` (cross-correlation on a 1 ms grid + parabolic refinement)."""
        a, b = np.asarray(ref_t, float), np.asarray(oth_t, float)
        lo, hi = max(a[0], b[0]) + max_lag_ms, min(a[-1], b[-1]) - max_lag_ms
        g = np.arange(lo, hi, 1.0)
        r = np.interp(g, a, ref_v); r = r - r.mean()
        best = []
        for lag in range(-max_lag_ms, max_lag_ms + 1):
            o = np.interp(g - lag, b, oth_v); o = o - o.mean()
            best.append(float(np.dot(r, o) / (np.linalg.norm(r) * np.linalg.norm(o) + 1e-12)))
        j = int(np.argmax(best)); lag = float(j - max_lag_ms)
        if 0 < j < len(best) - 1:
            y0, y1, y2 = best[j - 1], best[j], best[j + 1]
            den = (y0 - 2 * y1 + y2)
            if den != 0: lag += 0.5 * (y0 - y2) / den
        return lag

    def alignment_report(self, k0: int, k1: int, mode: str = "interp") -> dict:
        rep = {}
        for n, s in self.streams.items():
            tm = s.master_t(); errs = []; oor = 0
            for k in range(k0, k1 + 1):
                smp = self.sample(n, self.clock.frame_to_ms(k), mode)
                errs.append(smp["err_ms"]); oor += (not smp["in_range"])
            rep[n] = {"max_query_err_ms": float(np.max(errs)), "mean_query_err_ms": float(np.mean(errs)), "out_of_range_frames": int(oor),
                      "sample_period_ms": float(np.median(np.diff(tm))), "clock_offset_ms": s.clock_offset_ms, "skew_ppm": s.skew_ppm}
        return rep

    def reconstruction_error(self, name: str, col: str) -> dict:
        """Leave-one-out linear-interpolation error against held-out real samples."""
        s = self.streams[name]; tm = s.master_t(); v = s.cols[col]
        pred = v[:-2] + (v[2:] - v[:-2]) * (tm[1:-1] - tm[:-2]) / (tm[2:] - tm[:-2])      # interpolate each interior sample from its neighbours
        e = np.abs(pred - v[1:-1])
        return {"mae": float(e.mean()), "p95": float(np.percentile(e, 95)), "max": float(e.max())}


# ====================================================================== branching state machine
class BranchError(Exception): pass


class BranchGraph:
    """Scenario graph: nodes (time segment + narration + optional decision gate); gate options point at next nodes and carry an `effect` tag
    that the domain simulator turns into a different telemetry future. Validation guarantees no dangling edges and a reachable terminal."""

    def __init__(self, scenario: dict):
        self.s = scenario; self.nodes = {n["id"]: n for n in scenario["nodes"]}; self.start = scenario["start"]
        self.validate()

    def validate(self):
        if self.start not in self.nodes: raise BranchError(f"start node {self.start} missing")
        for n in self.nodes.values():
            g = n.get("gate")
            if g:
                ids = set()
                for o in g["options"]:
                    if o["next"] not in self.nodes: raise BranchError(f"{n['id']}: option {o['id']} -> unknown node {o['next']}")
                    if o["id"] in ids: raise BranchError(f"{n['id']}: duplicate option id {o['id']}")
                    ids.add(o["id"])
            elif not n.get("terminal"):
                if n.get("next") not in self.nodes: raise BranchError(f"{n['id']}: no gate, not terminal, bad 'next'")
        seen, stack = set(), [self.start]
        while stack:
            x = stack.pop()
            if x in seen: continue
            seen.add(x); nd = self.nodes[x]
            stack += [o["next"] for o in nd["gate"]["options"]] if nd.get("gate") else ([nd["next"]] if nd.get("next") else [])
        if not any(self.nodes[i].get("terminal") for i in seen): raise BranchError("no terminal node reachable")
        if seen != set(self.nodes): raise BranchError(f"unreachable nodes: {set(self.nodes) - seen}")

    def advance(self, node_id: str, option_id: str | None = None) -> dict:
        n = self.nodes[node_id]
        if n.get("gate"):
            opts = {o["id"]: o for o in n["gate"]["options"]}
            if option_id not in opts: raise BranchError(f"option {option_id!r} not available at {node_id}")
            o = opts[option_id]; return {"next": o["next"], "effect": o.get("effect"), "feedback": o.get("feedback", ""), "score": o.get("score", 0)}
        if n.get("terminal"): raise BranchError("terminal node")
        return {"next": n["next"], "effect": None, "feedback": "", "score": 0}

    def resolve_path(self, decisions: list[str]) -> dict:
        """Walk the graph from `start` applying `decisions` (option ids at successive gates). Returns visited nodes, effects, total score."""
        cur, visited, effects, score = self.start, [self.start], [], 0
        it = iter(decisions)
        while True:
            n = self.nodes[cur]
            if n.get("terminal"): break
            if n.get("gate"):
                try: opt = next(it)
                except StopIteration: break
            else: opt = None
            a = self.advance(cur, opt)
            if a["effect"]: effects.append({"node": cur, "effect": a["effect"], "t_ms": n.get("gate", {}).get("t_ms", n.get("t_end_ms"))})
            score += a["score"]; cur = a["next"]; visited.append(cur)
        return {"nodes": visited, "effects": effects, "score": score, "at": cur, "done": bool(self.nodes[cur].get("terminal"))}


def load_scenario(path) -> dict:
    return json.load(open(path, encoding="utf-8"))
