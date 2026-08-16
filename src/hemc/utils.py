"""Small shared utilities: on-disk feature caching and global RNG seeding."""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

CACHE_ROOT = Path(__file__).resolve().parents[2] / "cache"
DEFAULT_SEED = 42


def cache_path(*parts: str, ext: str = "parquet") -> Path:
    """Build a path under cache/<parts...>.<ext>, creating parent dirs."""
    path = CACHE_ROOT.joinpath(*parts[:-1], f"{parts[-1]}.{ext}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_or_compute(path: Path, compute_fn):
    """Return the cached parquet DataFrame at `path` if present, else compute, save, return."""
    if path.exists():
        return pd.read_parquet(path)
    df = compute_fn()
    df.to_parquet(path)
    return df


def set_global_seed(seed: int = DEFAULT_SEED) -> None:
    """Seed python, numpy and (if importable) torch RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
