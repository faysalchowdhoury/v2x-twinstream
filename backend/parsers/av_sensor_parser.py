"""AV perception / CAN adapter for a REAL KITTI raw drive (2011_09_26_drive_0001): OXTS GNSS/IMU stream, Velodyne HDL-64 frames,
camera/LiDAR timestamps and 3-D object tracklets. KITTI is CC BY-NC-SA 3.0 and is downloaded at run time (scripts/fetch_data.py kitti);
nothing derived from it is committed.

CAN-like signals (speed, longitudinal acceleration, yaw rate, steering angle) are DERIVED from the real OXTS measurements (KITTI does not
publish the vehicle CAN bus); the derivation (kinematic bicycle model, wheelbase 2.71 m) is explicit in `can_stream`.
"""
from __future__ import annotations
import math, pathlib, xml.etree.ElementTree as ET
from datetime import datetime
import numpy as np
from ..streams import Stream

ROOT = pathlib.Path(__file__).resolve().parents[2] / "data" / "raw" / "kitti" / "2011_09_26" / "2011_09_26_drive_0001_sync"
WHEELBASE = 2.71
OXTS_FIELDS = ["lat", "lon", "alt", "roll", "pitch", "yaw", "vn", "ve", "vf", "vl", "vu", "ax", "ay", "az", "af", "al", "au", "wx", "wy", "wz",
               "wf", "wl", "wu", "pos_acc", "vel_acc", "navstat", "numsats", "posmode", "velmode", "orimode"]
SENSOR_Z = -1.73                 # ground plane height in the Velodyne frame (sensor mounted 1.73 m above ground)


def available() -> bool:
    return (ROOT / "oxts" / "timestamps.txt").exists()


def _ts(path) -> np.ndarray:
    out = []
    for line in open(path):
        line = line.strip()
        if not line: continue
        d, t = line.split(" "); sec, frac = t.split(".")
        dt = datetime.strptime(f"{d} {sec}", "%Y-%m-%d %H:%M:%S")
        out.append(int(dt.replace(tzinfo=None).timestamp() * 1000) + int(frac[:3]))
    return np.array(out, dtype=np.int64)


def oxts_stream() -> Stream:
    t = _ts(ROOT / "oxts" / "timestamps.txt"); n = len(t)
    A = np.array([[float(x) for x in open(ROOT / "oxts" / "data" / f"{i:010d}.txt").read().split()] for i in range(n)])
    cols = {k: A[:, j] for j, k in enumerate(OXTS_FIELDS)}
    return Stream("gnss_imu:kitti0001", "can", t, cols, {"dataset": "KITTI raw 2011_09_26_drive_0001 (OXTS)", "rate_hz": 10})


def can_stream(ox: Stream) -> Stream:
    """CAN-like frames derived from real OXTS: wheel-speed (m/s), steering angle (rad, bicycle model), long. acceleration, yaw rate."""
    v = np.maximum(ox.cols["vf"], 0.1); wz = ox.cols["wz"]
    steer = np.arctan(WHEELBASE * wz / v)
    cols = {"speed_mps": ox.cols["vf"].copy(), "steer_rad": steer, "accel_long": ox.cols["af"].copy(), "yaw_rate": wz.copy(),
            "brake_cmd_mps2": np.minimum(ox.cols["af"], 0.0)}
    return Stream("can:kitti0001", "can", ox.t_ms.copy(), cols, {"derived_from": "OXTS", "wheelbase_m": WHEELBASE, "rate_hz": 10})


def load_points(i: int) -> np.ndarray:
    return np.fromfile(ROOT / "velodyne_points" / "data" / f"{i:010d}.bin", dtype=np.float32).reshape(-1, 4)


def frame_features(pts: np.ndarray) -> np.ndarray:
    """Per-frame scalar features an ML IDS would see: point count, range stats, forward-corridor density, intensity stats."""
    r = np.linalg.norm(pts[:, :3], axis=1)
    fwd = (pts[:, 0] > 0) & (pts[:, 0] < 30) & (np.abs(pts[:, 1]) < 2.0) & (pts[:, 2] > SENSOR_Z + 0.3)
    return np.array([len(pts), r.mean(), r.std(), fwd.sum(), pts[:, 3].mean(), pts[:, 3].std(), (pts[:, 2] > SENSOR_Z + 0.3).mean()])


def lidar_stream() -> Stream:
    t = _ts(ROOT / "velodyne_points" / "timestamps.txt"); n = len(t)
    F = np.array([frame_features(load_points(i)) for i in range(n)])
    names = ["n_points", "r_mean", "r_std", "fwd_corridor_pts", "inten_mean", "inten_std", "above_ground_frac"]
    return Stream("lidar:kitti0001", "lidar", t, {k: F[:, j] for j, k in enumerate(names)}, {"dataset": "KITTI raw (Velodyne HDL-64E)", "rate_hz": 10})


def camera_timestamps() -> np.ndarray:
    return _ts(ROOT / "image_02" / "timestamps.txt")


def tracklets() -> list[dict]:
    """Real annotated objects: dict(type, h, w, l, first_frame, poses[(tx,ty,tz,rz)]) in the Velodyne frame."""
    root = ET.parse(ROOT / "tracklet_labels.xml").getroot(); out = []
    for it in root.find("tracklets").findall("item"):
        poses = [tuple(float(p.find(k).text) for k in ("tx", "ty", "tz", "rz")) for p in it.find("poses").findall("item")]
        out.append({"type": it.find("objectType").text, "h": float(it.find("h").text), "w": float(it.find("w").text), "l": float(it.find("l").text),
                    "first_frame": int(it.find("first_frame").text), "poses": poses})
    return out


# ---------------------------------------------------------------- attack: LiDAR ghost-object injection
def ghost_points(dist: float, lateral: float = 0.0, l: float = 4.5, w: float = 1.8, h: float = 1.5, spacing: float = 0.12) -> np.ndarray:
    """Points on the sensor-facing surfaces (rear face + near side + top) of a car-sized box at (dist, lateral); deterministic lattice."""
    x0 = dist; pts = []
    ys = np.arange(-w / 2, w / 2 + 1e-6, spacing); zs = np.arange(0, h + 1e-6, spacing); xs = np.arange(0, l + 1e-6, spacing)
    for y in ys:
        for z in zs: pts.append((x0, lateral + y, SENSOR_Z + z))                          # rear face (towards sensor)
    for x in xs:
        for y in ys: pts.append((x0 + x, lateral + y, SENSOR_Z + h))                      # roof
        for z in zs: pts.append((x0 + x, lateral + (-w / 2 if lateral >= 0 else w / 2), SENSOR_Z + z))   # side facing the ego lane
    P = np.array(pts, dtype=np.float32)
    return np.c_[P, np.full(len(P), 0.3, np.float32)]


def inject_ghost(pts: np.ndarray, dist: float, lateral: float = 0.0, **kw) -> tuple[np.ndarray, np.ndarray]:
    g = ghost_points(dist, lateral, **kw)
    return np.vstack([pts, g]), g


# ---------------------------------------------------------------- physics: line-of-sight (ray-casting) consistency
def _sph(p: np.ndarray):
    r = np.linalg.norm(p[:, :3], axis=1)
    az = np.degrees(np.arctan2(p[:, 1], p[:, 0])); el = np.degrees(np.arcsin(np.clip(p[:, 2] / np.maximum(r, 1e-6), -1, 1)))
    return r, az, el


def behind_fraction(cloud: np.ndarray, obj_pts: np.ndarray, az_tol: float = 0.35, el_tol: float = 0.9, gap: float = 1.5) -> float:
    """Fraction of the object's rays on which the scan ALSO contains returns clearly BEHIND the object. A real opaque object occludes
    the space behind it, so real objects give ~0; points injected into a recorded scan leave the true returns behind them (violation)."""
    ro, ao, eo = _sph(obj_pts); rc, ac, ec = _sph(cloud)
    order = np.argsort(ac); ac_s, rc_s, ec_s = ac[order], rc[order], ec[order]
    # an object ray (az, el) is violated if some other return along (az±tol, el±tol) lies > gap metres farther than the object
    viol = 0; n = 0
    for k in range(0, len(ro), 3):
        lo, hi = np.searchsorted(ac_s, [ao[k] - az_tol, ao[k] + az_tol])
        seg_r, seg_e = rc_s[lo:hi], ec_s[lo:hi]
        n += 1
        if len(seg_r) and np.any((np.abs(seg_e - eo[k]) < el_tol) & (seg_r > ro[k] + gap)): viol += 1
    return viol / max(n, 1)


def object_points(pts: np.ndarray, tr: dict, frame: int, margin: float = 0.3) -> np.ndarray:
    """Returns of the cloud that fall inside a real tracklet box (used to calibrate the check on REAL objects)."""
    k = frame - tr["first_frame"]
    if k < 0 or k >= len(tr["poses"]): return np.zeros((0, 4), np.float32)
    tx, ty, tz, rz = tr["poses"][k]
    c, s = math.cos(-rz), math.sin(-rz)
    dx, dy = pts[:, 0] - tx, pts[:, 1] - ty
    lx, ly = c * dx - s * dy, s * dx + c * dy
    m = (np.abs(lx) < tr["l"] / 2 + margin) & (np.abs(ly) < tr["w"] / 2 + margin) & (pts[:, 2] > tz + 0.25) & (pts[:, 2] < tz + tr["h"] + margin)
    return pts[m]
