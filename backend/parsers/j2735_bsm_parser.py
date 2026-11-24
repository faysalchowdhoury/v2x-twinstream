"""SAE J2735 Basic Safety Message (BSM Part I) adapter for real NGSIM Peachtree Street trajectories.

NGSIM (US DOT) gives, at 10 Hz per vehicle: global X/Y (Georgia West State Plane, US ft), speed (ft/s), acceleration (ft/s^2), lane, lateral
position, vehicle size, and the measured space headway to the preceding vehicle. We convert these to the physical units/encodings of the
J2735 BSM Part I core data set:

  msgCnt (0..127) | id (4 octets) | secMark (ms in minute, 0..59999) | lat, long (1e-7 deg) | elev (0.1 m) | speed (0.02 m/s, 13 bit)
  heading (0.0125 deg, 0..28799) | accelSet: long, lat (0.01 m/s^2), vert, yaw (0.01 deg/s) | brakes | size (cm)

Fields NGSIM cannot provide (GNSS positional accuracy, steering-wheel angle, vertical accel.) are encoded with the standard "unavailable"
values rather than invented.
"""
from __future__ import annotations
import hashlib
import numpy as np
import pandas as pd
from pyproj import Transformer
from ..streams import Stream

FT = 0.3048
_T = Transformer.from_crs("EPSG:2240", "EPSG:4326", always_xy=True)       # NAD83 / Georgia West (ftUS) -> WGS84
SPEED_UNAVAILABLE, ANGLE_UNAVAILABLE, ACC_UNAVAILABLE = 8191, 127, 2001
BSM_HZ = 10.0


def load_ngsim(path) -> pd.DataFrame:
    d = pd.read_csv(path).sort_values(["vehicle_id", "frame_id"]).reset_index(drop=True)
    return d


def _heading_deg(x, y, v):
    """Course over ground (deg clockwise from north) from consecutive positions; held when (almost) stationary."""
    dx, dy = np.gradient(x), np.gradient(y)
    h = (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0
    slow = v < 0.5
    for i in range(len(h)):
        if slow[i]:
            h[i] = h[i - 1] if i > 0 else (h[i + 1:][~slow[i + 1:]][0] if (~slow[i + 1:]).any() else 0.0)
    return h


def vehicle_stream(df: pd.DataFrame, vid: int, f0: int | None = None, f1: int | None = None) -> Stream:
    """One vehicle's BSM-style stream. `cols` holds both the physical truth (SI units) and the J2735 encoded fields."""
    v = df[df.vehicle_id == vid]
    if f0 is not None: v = v[v.frame_id >= f0]
    if f1 is not None: v = v[v.frame_id <= f1]
    v = v.sort_values("frame_id")
    gx, gy = v.global_x.to_numpy() * FT, v.global_y.to_numpy() * FT        # metres in the state-plane frame
    lon, lat = _T.transform(v.global_x.to_numpy(), v.global_y.to_numpy())
    spd = v.v_vel.to_numpy() * FT
    acc = v.v_acc.to_numpy() * FT
    hd = _heading_deg(gx, gy, spd)
    yaw_rate = np.gradient(np.unwrap(np.radians(hd))) * BSM_HZ               # rad/s
    t = (v.global_time.to_numpy()).astype(np.int64)
    t = np.round(t / 100.0).astype(np.int64) * 100                           # NGSIM global_time is on a 100 ms grid (some rows jitter)
    cols = dict(
        x_m=gx, y_m=gy, lat=lat, lon=lon, speed_mps=spd, accel_long=acc, heading_deg=hd, yaw_rate=yaw_rate,
        lane_id=v.lane_id.to_numpy().astype(float), local_x_m=v.local_x.to_numpy() * FT, local_y_m=v.local_y.to_numpy() * FT,
        length_m=v.v_length.to_numpy() * FT, width_m=v.v_width.to_numpy() * FT,
        radar_gap_m=v.space_headway.to_numpy() * FT, preceding=v.preceding.to_numpy().astype(float), frame_id=v.frame_id.to_numpy().astype(float))
    ok = np.concatenate([[True], np.diff(t) > 0])
    s = Stream(f"bsm:{vid}", "bsm", t[ok], {k: a[ok] for k, a in cols.items()}, {"vehicle_id": int(vid), "dataset": "NGSIM Peachtree (US DOT)", "rate_hz": BSM_HZ})
    return s


def bsm_message(s: Stream, i: int) -> dict:
    """Encode sample i as a J2735 BSM (dict mirroring the ASN.1 BasicSafetyMessage / BSMcoreData)."""
    c = s.cols; vid = int(s.meta["vehicle_id"])
    ident = hashlib.sha256(f"veh{vid}".encode()).hexdigest()[:8]            # 4-octet temporary ID (pseudonym)
    return {
        "messageId": 20, "value": {"coreData": {
            "msgCnt": int(i % 128), "id": ident, "secMark": int(s.t_ms[i] % 60000),
            "lat": int(round(c["lat"][i] * 1e7)), "long": int(round(c["lon"][i] * 1e7)), "elev": 0,
            "accuracy": {"semiMajor": 255, "semiMinor": 255, "orientation": 65535},     # unavailable in NGSIM
            "transmission": "forwardGears",
            "speed": int(round(min(c["speed_mps"][i], 163.8) / 0.02)),
            "heading": int(round(c["heading_deg"][i] / 0.0125)) % 28800,
            "angle": ANGLE_UNAVAILABLE,
            "accelSet": {"long": int(round(np.clip(c["accel_long"][i], -20, 20) / 0.01)), "lat": ACC_UNAVAILABLE, "vert": -127,
                         "yaw": int(round(np.degrees(c["yaw_rate"][i]) / 0.01))},
            "brakes": {"wheelBrakes": "unavailable", "traction": "unavailable", "abs": "unavailable"},
            "size": {"width": int(round(c["width_m"][i] * 100)), "length": int(round(c["length_m"][i] * 100))}}}}


def decode_core(msg: dict) -> dict:
    """Inverse of bsm_message for the fields the verifiers use (SI units)."""
    c = msg["value"]["coreData"]
    return {"lat": c["lat"] * 1e-7, "lon": c["long"] * 1e-7, "speed_mps": c["speed"] * 0.02, "heading_deg": c["heading"] * 0.0125,
            "accel_long": c["accelSet"]["long"] * 0.01, "yaw_rate_dps": c["accelSet"]["yaw"] * 0.01, "secMark": c["secMark"]}


def leader_follower_pairs(df: pd.DataFrame, min_common_frames: int = 300, top: int = 12, min_moving: float = 0.0) -> list[tuple[int, int, int, int]]:
    """Real leader/follower pairs (NGSIM 'preceding' field) inside ONE road section with a continuous overlap:
    (follower, leader, first_frame, last_frame). `min_moving` = minimum fraction of frames with follower speed > 3 m/s (free-flowing traffic)."""
    out = []
    d = df[df.preceding > 0]
    g = d.groupby(["vehicle_id", "preceding", "section_id"]).agg(f0=("frame_id", "min"), f1=("frame_id", "max"), n=("frame_id", "count"),
                                                                  mov=("v_vel", lambda x: float(np.mean(x * FT > 3.0)))).reset_index()
    for r in g.itertuples():
        if r.n >= min_common_frames and (r.f1 - r.f0 + 1) == r.n and r.mov >= min_moving:
            out.append((int(r.vehicle_id), int(r.preceding), int(r.f0), int(r.f1)))
    out.sort(key=lambda p: -(p[3] - p[2]))
    return out[:top]
