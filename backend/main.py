"""V2X-TwinStream API.  Run:  uvicorn backend.main:app --reload   (then open http://127.0.0.1:8000/)

Synchronized multi-stream endpoints (all deterministic: same track + same decisions -> bit-identical bytes):
  GET  /api/tracks                            list of tracks
  GET  /api/tracks/{id}                       scenario graph (option effects / scores are withheld until a decision is made)
  GET  /api/tracks/{id}/path?decisions=a,b    branching state machine: visited nodes, feedback, score, outcome metrics
  GET  /api/tracks/{id}/timeline?decisions=&k0=&k1=&step=   columnar, frame-aligned telemetry (all streams on the master clock)
  GET  /api/tracks/{id}/frame/{k}?decisions=  one aligned frame (video frame k -> integer ms -> every stream)
  GET  /api/tracks/{id}/scrub?decisions=&t_ms=  telemetry scrubbing: aligned frame at an arbitrary millisecond
  GET  /api/tracks/{id}/digest|alignment      determinism digest / alignment report
  GET  /api/tracks/track3_av_can/lidar/{k}    real Velodyne frame (BEV) when KITTI is present
  GET  /api/audit                             consequence-aware IDS benchmark summary
  GET  /api/ctf/{video,frame.png,manifest,artifacts/{name}}  POST /api/ctf/{extract,verify,submit}   Track 5
"""
from __future__ import annotations
import functools, io, json, pathlib
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from . import tracks as T, stego_manifest as SM
from .sync_engine import SyncEngine, BranchError

ROOT = pathlib.Path(__file__).resolve().parents[1]
app = FastAPI(title="V2X-TwinStream", version="1.0.0")
BUILDERS = {"cav": T.build_cav, "ev": T.build_ev, "av": T.build_av, "ctf": T.build_ctf}


def _decisions(s: str) -> list[str]: return [x for x in s.split(",") if x]


@functools.lru_cache(maxsize=64)
def _world(track_id: str, effect: str | None):
    sc = T.scenario(track_id); streams, outcome = BUILDERS[sc["domain"]](sc, effect)
    eng = SyncEngine(sc["fps"]); [eng.add(s) for s in streams]; return eng, outcome


def _resolve(track_id: str, decisions: str):
    try: sc = T.scenario(track_id)
    except KeyError: raise HTTPException(404, f"unknown track {track_id}")
    try: path = T.graph(track_id).resolve_path(_decisions(decisions))
    except (BranchError, KeyError) as e: raise HTTPException(400, str(e))
    eff = T._effect_of(sc, path)
    try: eng, out = _world(track_id, eff)
    except FileNotFoundError as e: raise HTTPException(503, str(e))
    return sc, path, eff, eng, out


def _public(sc: dict) -> dict:
    s = json.loads(json.dumps(sc))
    for n in s["nodes"]:
        if n.get("gate"):
            for o in n["gate"]["options"]: [o.pop(k, None) for k in ("effect", "feedback", "score")]
    return s


@app.get("/api/health")
def health(): return {"ok": True, "kitti_available": T._kitti_ok(), "tracks": len(T.list_tracks())}


@app.get("/api/tracks")
def tracks(): return T.list_tracks()


@app.get("/api/tracks/{track_id}")
def track(track_id: str):
    try: sc = T.scenario(track_id)
    except KeyError: raise HTTPException(404, "unknown track")
    return _public(sc)


@app.get("/api/tracks/{track_id}/path")
def path(track_id: str, decisions: str = ""):
    sc, p, eff, eng, out = _resolve(track_id, decisions)
    g = T.graph(track_id); fb = []; cur = g.start
    for d in _decisions(decisions):
        a = g.advance(cur, d); fb.append({"node": cur, "option": d, "feedback": a["feedback"], "score": a["score"]}); cur = a["next"]
    return {"nodes": p["nodes"], "done": p["done"], "at": p["at"], "score": p["score"], "feedback": fb, "effect": eff,
            "outcome": {k: v for k, v in out.items() if not isinstance(v, np.ndarray)}}


def _n_frames(sc): return int(sc["duration_ms"] * sc["fps"] // 1000)


@app.get("/api/tracks/{track_id}/timeline")
def timeline(track_id: str, decisions: str = "", k0: int = 0, k1: int | None = None, step: int = Query(1, ge=1, le=300)):
    sc, p, eff, eng, out = _resolve(track_id, decisions); k1 = _n_frames(sc) if k1 is None else k1
    if not (0 <= k0 <= k1): raise HTTPException(400, "bad frame range")
    cols = eng.columns(k0, k1, step); cols["digest"] = eng.digest(k0, k1)
    return JSONResponse(cols)


@app.get("/api/tracks/{track_id}/frame/{k}")
def frame(track_id: str, k: int, decisions: str = "", mode: str = "interp"):
    sc, p, eff, eng, out = _resolve(track_id, decisions)
    if k < 0 or k > _n_frames(sc): raise HTTPException(400, "frame out of range")
    return eng.frame(k, mode)


@app.get("/api/tracks/{track_id}/scrub")
def scrub(track_id: str, t_ms: int, decisions: str = "", mode: str = "interp"):
    sc, p, eff, eng, out = _resolve(track_id, decisions); k = eng.clock.ms_to_frame(t_ms); f = eng.frame(k, mode)
    f["requested_ms"] = t_ms; f["frame_snap_error_ms"] = f["t_ms"] - t_ms; return f


@app.get("/api/tracks/{track_id}/digest")
def digest(track_id: str, decisions: str = "", k0: int = 0, k1: int | None = None):
    sc, p, eff, eng, out = _resolve(track_id, decisions); k1 = _n_frames(sc) if k1 is None else k1
    return {"track": track_id, "decisions": _decisions(decisions), "k0": k0, "k1": k1, "sha256": eng.digest(k0, k1)}


@app.get("/api/tracks/{track_id}/alignment")
def alignment(track_id: str, decisions: str = ""):
    sc, p, eff, eng, out = _resolve(track_id, decisions); return eng.alignment_report(0, _n_frames(sc))


@app.get("/api/tracks/track3_av_can/lidar/{k}")
def lidar(k: int):
    if not T._kitti_ok(): raise HTTPException(503, "KITTI not downloaded: run python scripts/fetch_data.py kitti")
    return T.lidar_frame(k)


@app.get("/api/audit")
def audit():
    p = ROOT / "results" / "audit_summary.json"
    if not p.exists(): raise HTTPException(404, "run python benchmarks/run_sync_and_ids_eval.py first")
    return json.load(open(p))


# ------------------------------------------------------------------ Track 5 (CTF)
CTF_VIDEO = ROOT / "data" / "ctf" / "incident.mp4"


def _ctf_video() -> pathlib.Path:
    if not CTF_VIDEO.exists():
        CTF_VIDEO.parent.mkdir(parents=True, exist_ok=True); SM.write_mp4(CTF_VIDEO, T.ctf_frames(), 30, crf=28)
    return CTF_VIDEO


@app.get("/api/ctf/video")
def ctf_video(): return FileResponse(_ctf_video(), media_type="video/mp4", filename="incident.mp4")


@app.get("/api/ctf/frame.png")
def ctf_frame_png():
    from PIL import Image
    buf = io.BytesIO(); Image.fromarray(T.ctf_frames(1)[0]).save(buf, "PNG"); return Response(buf.getvalue(), media_type="image/png")


@app.get("/api/ctf/manifest")
def ctf_manifest():
    a = T.ctf_artifacts(); return {"files": a["per_file"], "algorithm": "sha256(canonical_json({file: sha256})) -> embedded in every video frame"}


@app.get("/api/ctf/artifacts/{name}")
def ctf_artifact(name: str):
    a = T.ctf_artifacts()["files"]
    if name not in a: raise HTTPException(404, "no such artifact")
    return Response(a[name], media_type="application/octet-stream", headers={"Content-Disposition": f"attachment; filename={name}"})


@app.post("/api/ctf/extract")
def ctf_extract():
    fr = SM.read_mp4(_ctf_video(), every=10, limit=6); e = SM.extract_hash(fr); return {"decode": e}


class Hash(BaseModel): hash: str


class Flag(BaseModel): flag: str


@app.post("/api/ctf/verify")
def ctf_verify(b: Hash):
    a = T.ctf_artifacts(); ok = b.hash.strip().lower() == a["manifest_hash"]
    return {"match": ok, "message": "Hash matches the manifest of telemetry.csv + capture.pcap: the artifacts belong to this footage." if ok else "No match."}


@app.post("/api/ctf/submit")
def ctf_submit(b: Flag):
    ok = b.flag.strip() == T.ctf_artifacts()["flag"]
    return {"correct": ok, "message": "Correct." if ok else "Not the forged frame's flag. Hint: TWIN{first 16 hex of sha256(compact JSON of the rogue-host SetChargingProfile)}"}


app.mount("/", StaticFiles(directory=str(ROOT / "frontend"), html=True), name="frontend")
