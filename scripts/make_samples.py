"""Build the small committed data samples (public-domain NGSIM rows + CC-BY Norway sessions of one location) from data/raw.
KITTI (CC BY-NC-SA) is never copied into the repository."""
import json, pathlib, sys
import pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[1]
CASES = [(85, 60, 709, 1036), (1311, 1293, 7315, 7561)]           # scenario pairs (follower, leader, f0, f1); see scenarios/track1, track4
pairs = json.load(open(ROOT / "data" / "cav_pairs.json"))
df = pd.read_csv(ROOT / "data" / "raw" / "ngsim_peachtree.csv")
parts = []
for f, l, a, b in CASES:
    for vid in (f, l): parts.append(df[(df.vehicle_id == vid) & (df.frame_id >= a) & (df.frame_id <= b)])
out = pd.concat(parts).drop_duplicates(["vehicle_id", "frame_id"])
out.to_csv(ROOT / "data" / "samples" / "ngsim_cases.csv", index=False)
ev = pd.read_csv(ROOT / "data" / "raw" / "ev_norway_sessions.csv", sep=";", decimal=",")
ev[ev.location == "OSL_S"].to_csv(ROOT / "data" / "samples" / "ev_norway_OSL_S.csv", sep=";", decimal=",", index=False)
print("ngsim rows", len(out), "ev rows", int((ev.location == "OSL_S").sum()))
