"""Common time-series container used by every parser and by the sync engine.

All timestamps are integer milliseconds on the dataset's own clock ("t_ms"); the sync engine maps them onto one master clock.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Stream:
    name: str                       # e.g. "bsm:105", "ocpp:ASK_18266", "can:kitti0001"
    kind: str                       # "bsm" | "ocpp" | "can" | "lidar" | "video"
    t_ms: np.ndarray                # int64, strictly increasing
    cols: dict[str, np.ndarray]     # float/int arrays, same length as t_ms
    meta: dict = field(default_factory=dict)
    clock_offset_ms: float = 0.0    # device-clock offset to the master clock (added to t_ms)
    skew_ppm: float = 0.0           # device-clock drift relative to master, parts per million

    def __post_init__(self):
        self.t_ms = np.asarray(self.t_ms, dtype=np.int64)
        for k, v in self.cols.items():
            assert len(v) == len(self.t_ms), f"column {k} length mismatch"

    def __len__(self):
        return len(self.t_ms)

    def master_t(self) -> np.ndarray:
        """Timestamps on the master clock (float ms)."""
        t0 = float(self.t_ms[0])
        return (self.t_ms - t0) * (1.0 + self.skew_ppm * 1e-6) + t0 + self.clock_offset_ms

    @property
    def duration_ms(self) -> int:
        return int(self.t_ms[-1] - self.t_ms[0])

    def slice_ms(self, a: float, b: float) -> "Stream":
        m = (self.master_t() >= a) & (self.master_t() <= b)
        return Stream(self.name, self.kind, self.t_ms[m], {k: v[m] for k, v in self.cols.items()}, dict(self.meta), self.clock_offset_ms, self.skew_ppm)
