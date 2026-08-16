"""Dataset loading and group-aware splitting for GazeCom and HMR.

Both datasets ship as tab-separated CSVs with columns::

    X_coord    Y_coord    Confidence    Pattern

with one row per gaze sample and no explicit timestamp column - both source
papers report a fixed sampling rate per dataset (GazeCom: 250 Hz, HMR: 200 Hz),
so sample index directly implies time via ``t = i / fs``.

GazeCom stores one file per (participant, video) pair, e.g.
``DATASET/data_gazecom/beach/AAF_beach.csv``. HMR stores one directory per
participant with one file per eye, e.g. ``DATASET/data_hmr/user_1/eye_0.csv``.

The label column ``Pattern`` uses single-character codes shared by both
datasets: F(ixation), S(accade), P(smooth pursuit), B(link).

Splitting: both source papers report an 80/10/10 train/val/test split without
specifying whether it's over individual samples or over whole recordings.
Splitting by sample risks temporal leakage (adjacent samples from the same
recording landing in different splits), so `group_train_val_test_split`
groups by *recording* (participant x video for GazeCom, participant for HMR)
by default. `sample_train_val_test_split` is kept for anyone who wants to try
matching the paper's numbers more closely despite the leakage risk.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "DATASET"
GAZECOM_ROOT = DATASET_ROOT / "data_gazecom"
HMR_ROOT = DATASET_ROOT / "data_hmr"

GAZECOM_FS_HZ = 250.0
HMR_FS_HZ = 200.0

LABEL_COLUMNS = ["X_coord", "Y_coord", "Confidence", "Pattern"]
CLASSES_4 = ["F", "S", "P", "B"]
CLASS_NAMES_4 = {"F": "fixation", "S": "saccade", "P": "smooth_pursuit", "B": "blink"}


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RecordingMeta:
    """Identifies a single continuous recording (one subject, one session/video, one eye)."""

    dataset: str  # "gazecom" | "hmr"
    recording_id: str  # unique key, e.g. "AAF_beach" or "user_1_eye_0"
    subject: str  # participant id
    group: str  # split-grouping key: recording_id for gazecom, subject for hmr
    fs_hz: float
    video: str | None = None  # gazecom only
    eye: str | None = None  # hmr only ("eye_0" | "eye_1")


def _read_raw_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", names=LABEL_COLUMNS, header=0, dtype={"Pattern": "string"})
    df = df.rename(columns={"X_coord": "x", "Y_coord": "y", "Confidence": "confidence", "Pattern": "label"})
    df["label"] = df["label"].astype(str)
    return df.reset_index(drop=True)


def load_gazecom_recording(path: Path) -> tuple[pd.DataFrame, RecordingMeta]:
    """Load one GazeCom CSV file, e.g. DATASET/data_gazecom/beach/AAF_beach.csv."""
    path = Path(path)
    video = path.parent.name
    subject = path.stem[: -(len(video) + 1)] if path.stem.endswith(f"_{video}") else path.stem
    recording_id = path.stem
    df = _read_raw_csv(path)
    meta = RecordingMeta(
        dataset="gazecom", recording_id=recording_id, subject=subject, group=recording_id,
        fs_hz=GAZECOM_FS_HZ, video=video,
    )
    return df, meta


def load_hmr_recording(path: Path) -> tuple[pd.DataFrame, RecordingMeta]:
    """Load one HMR CSV file, e.g. DATASET/data_hmr/user_1/eye_0.csv."""
    path = Path(path)
    subject = path.parent.name
    eye = path.stem
    recording_id = f"{subject}_{eye}"
    df = _read_raw_csv(path)
    meta = RecordingMeta(dataset="hmr", recording_id=recording_id, subject=subject, group=subject, fs_hz=HMR_FS_HZ, eye=eye)
    return df, meta


def iterate_gazecom(root: Path = GAZECOM_ROOT) -> Iterator[tuple[pd.DataFrame, RecordingMeta]]:
    """Yield (df, meta) for every GazeCom recording under `root`."""
    for csv_path in sorted(root.glob("*/*.csv")):
        yield load_gazecom_recording(csv_path)


def iterate_hmr(root: Path = HMR_ROOT) -> Iterator[tuple[pd.DataFrame, RecordingMeta]]:
    """Yield (df, meta) for every HMR recording (one per eye) under `root`."""
    for csv_path in sorted(root.glob("user_*/eye_*.csv")):
        yield load_hmr_recording(csv_path)


def load_hmr_binocular(user_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, RecordingMeta]:
    """Load both eyes for one HMR user, truncated to the common length (row counts can
    differ by a handful of samples; no finer-grained alignment signal is available)."""
    user_dir = Path(user_dir)
    df0, meta0 = load_hmr_recording(user_dir / "eye_0.csv")
    df1, meta1 = load_hmr_recording(user_dir / "eye_1.csv")
    n = min(len(df0), len(df1))
    meta = RecordingMeta(dataset="hmr", recording_id=f"{meta0.subject}_binocular", subject=meta0.subject, group=meta0.subject, fs_hz=HMR_FS_HZ)
    return df0.iloc[:n].reset_index(drop=True), df1.iloc[:n].reset_index(drop=True), meta


def iterate_hmr_binocular(root: Path = HMR_ROOT) -> Iterator[tuple[pd.DataFrame, pd.DataFrame, RecordingMeta]]:
    for user_dir in sorted(root.glob("user_*")):
        if user_dir.is_dir():
            yield load_hmr_binocular(user_dir)


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GroupSplit:
    train_groups: list[str]
    val_groups: list[str]
    test_groups: list[str]


def group_train_val_test_split(groups: list[str], train: float = 0.8, val: float = 0.1, test: float = 0.1, seed: int = 42) -> GroupSplit:
    """Split a list of group ids (recordings or subjects) into train/val/test.

    Uses a simple shuffled partition at the group level (not sklearn's
    GroupShuffleSplit) to keep exact proportions deterministic for a fixed
    group list.
    """
    if not np.isclose(train + val + test, 1.0):
        raise ValueError(f"train+val+test must sum to 1.0, got {train + val + test}")
    unique_groups = sorted(set(groups))
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_groups)
    n = len(shuffled)
    n_train = int(round(n * train))
    n_val = int(round(n * val))
    n_test = n - n_train - n_val

    # Guard against a rounded-to-zero split with very few groups (e.g. a small
    # --limit smoke test): steal one group from the largest split.
    counts = {"train": n_train, "val": n_val, "test": n_test}
    fractions = {"train": train, "val": val, "test": test}
    n_positive_fractions = sum(1 for f in fractions.values() if f > 0)
    if n >= n_positive_fractions:
        for name in ("val", "test"):
            if fractions[name] > 0 and counts[name] == 0:
                donor = max(counts, key=lambda k: counts[k])
                counts[donor] -= 1
                counts[name] += 1
    n_train, n_val = counts["train"], counts["val"]

    train_groups = list(shuffled[:n_train])
    val_groups = list(shuffled[n_train : n_train + n_val])
    test_groups = list(shuffled[n_train + n_val :])
    return GroupSplit(train_groups=train_groups, val_groups=val_groups, test_groups=test_groups)


def sample_train_val_test_split(n_samples: int, train: float = 0.8, val: float = 0.1, test: float = 0.1, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random per-sample split (ignores grouping); can leak adjacent-in-time samples across splits."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_samples)
    n_train = int(round(n_samples * train))
    n_val = int(round(n_samples * val))
    train_idx, val_idx, test_idx = idx[:n_train], idx[n_train : n_train + n_val], idx[n_train + n_val :]
    train_mask, val_mask, test_mask = (np.zeros(n_samples, dtype=bool) for _ in range(3))
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    return train_mask, val_mask, test_mask


def loso_folds(groups: pd.Series | np.ndarray | list[str]) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """Leave-One-Subject/Recording-Out folds: list of (train_idx, test_idx, held_out_group)."""
    groups = np.asarray(groups)
    unique_groups = sorted(set(groups))
    folds = []
    for g in unique_groups:
        test_idx = np.where(groups == g)[0]
        train_idx = np.where(groups != g)[0]
        folds.append((train_idx, test_idx, g))
    return folds


def group_kfold_splits(X_len: int, groups: np.ndarray, n_splits: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    """Thin wrapper around sklearn's GroupKFold for consistency with the rest of the module."""
    gkf = GroupKFold(n_splits=n_splits)
    return list(gkf.split(np.zeros(X_len), groups=groups))
