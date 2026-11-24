"""Download the REAL public datasets used by V2X-TwinStream into data/raw (nothing is generated or simulated here).

  ngsim   NGSIM Peachtree Street trajectories, US DOT open-data API (public domain), subset by section/frames.
  ev      Electric vehicle charging dataset, 35,000 sessions, 12 residential locations in Norway (Zenodo 13896176, CC-BY 4.0).
  kitti   KITTI raw drive 2011_09_26_drive_0001: OXTS (GNSS/IMU/CAN-like), Velodyne frames, tracklets (CC BY-NC-SA 3.0, downloaded at run
          time only; never redistributed by this repository).
  veremi  (optional, large) not downloaded automatically; see README for the Zenodo record.

usage: python scripts/fetch_data.py [ngsim] [ev] [kitti]
"""
import io, json, sys, zipfile, pathlib
import requests

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"; RAW.mkdir(parents=True, exist_ok=True)
SOC = "https://data.transportation.gov/resource/8ect-6jqj.json"
KITTI = "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0001/2011_09_26_drive_0001_sync.zip"
KITTI_TRK = "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0001/2011_09_26_drive_0001_tracklets.zip"
KITTI_CAL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_calib.zip"
ZEN_EV = "https://zenodo.org/api/records/13896176/files/Dataset1_charging_reports.csv/content"


def ngsim(section=None, f0=0, f1=20000):
    """US DOT NGSIM Peachtree Street (Atlanta), one section, frames f0..f1 (10 Hz)."""
    import pandas as pd
    rows, off = [], 0
    while True:
        r = requests.get(SOC, params={"$where": f"location='peachtree'" + (f" AND section_id='{section}'" if section else "") + f" AND frame_id>={f0} AND frame_id<={f1}", "$order": "vehicle_id,frame_id",
                                      "$limit": 50000, "$offset": off}, timeout=120)
        r.raise_for_status(); part = r.json()
        rows += part
        if len(part) < 50000: break
        off += 50000
    df = pd.DataFrame(rows)
    num = ["vehicle_id", "frame_id", "global_time", "local_x", "local_y", "global_x", "global_y", "v_length", "v_width", "v_class", "v_vel", "v_acc",
           "lane_id", "preceding", "following", "space_headway", "time_headway", "section_id", "direction", "movement"]
    for c in num:
        if c in df: df[c] = pd.to_numeric(df[c], errors="coerce")
    out = RAW / "ngsim_peachtree.csv"; df.to_csv(out, index=False)
    print("NGSIM rows", len(df), "vehicles", df.vehicle_id.nunique(), "->", out)


def ev():
    r = requests.get(ZEN_EV, timeout=300); r.raise_for_status()
    out = RAW / "ev_norway_sessions.csv"; out.write_bytes(r.content); print("EV sessions file", len(r.content) // 1024, "KB ->", out)


def kitti():
    for name, url in (("sync", KITTI), ("tracklets", KITTI_TRK), ("calib", KITTI_CAL)):
        dst = RAW / "kitti"; dst.mkdir(exist_ok=True)
        print("downloading KITTI", name, "...", flush=True)
        r = requests.get(url, timeout=600, stream=True); r.raise_for_status()
        z = zipfile.ZipFile(io.BytesIO(r.content))
        # keep what the AV track needs: OXTS, Velodyne, timestamps, tracklets, calibration (images are skipped to save space)
        for n in z.namelist():
            if name == "sync" and n.lower().endswith((".png", ".jpg")): continue
            z.extract(n, dst)
        print("  extracted", len(z.namelist()), "entries")


if __name__ == "__main__":
    what = sys.argv[1:] or ["ngsim", "ev"]
    for w in what:
        {"ngsim": ngsim, "ev": ev, "kitti": kitti}[w]()
