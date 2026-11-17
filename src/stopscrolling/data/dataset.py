"""Load extracted features for training.

    from stopscrolling.data.dataset import load_dataset
    ds = load_dataset()                 # reads data/processed/dataset.csv
    ds.X            # (n_frames, n_features) float32, NaN = missing
    ds.attentive    # (n_frames,) 1 = paying attention to work, 0 = not
    ds.labels       # (n_frames,) label names, e.g. "paper", "phone"
    ds.groups       # (n_frames,) clip ids - split train/test by clip, never by frame
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from stopscrolling.config import PROCESSED_DIR
from stopscrolling.features.extractor import FEATURE_NAMES


@dataclass
class Dataset:
    X: np.ndarray
    attentive: np.ndarray
    labels: np.ndarray
    groups: np.ndarray
    t: np.ndarray
    feature_names: list[str]

    def __len__(self) -> int:
        return len(self.X)


def load_dataset(path: Path | None = None) -> Dataset:
    path = Path(path) if path else PROCESSED_DIR / "dataset.csv"
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append(r)
    X = np.array([[float(r[n]) for n in FEATURE_NAMES] for r in rows], dtype=np.float32).reshape(-1, len(FEATURE_NAMES))
    return Dataset(
        X=X,
        attentive=np.array([int(r["attentive"]) for r in rows], dtype=np.int8),
        labels=np.array([r["label"] for r in rows]),
        groups=np.array([r["clip_id"] for r in rows]),
        t=np.array([float(r["t"]) for r in rows], dtype=np.float32),
        feature_names=list(FEATURE_NAMES),
    )
