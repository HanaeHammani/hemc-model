"""Point-wise kinematics, multi-scale window aggregation, and the EMCCF / EMCCF++ feature sets.

Pipeline: raw (x, y) trace -> point-wise kinematic signals (velocity,
direction, dispersion, ...) -> each signal aggregated over multi-scale
centered windows -> one feature vector per sample.

Window scheme (EMCCF paper): n_i = 1 + 2^i, i=1..7 -> spans [3, 5, 9, 17, 33,
65, 129] samples, centered on each sample (documented assumption: the paper
does not spell out the exact aggregation formula for "the velocity/direction
within a window" -- see README "Documented Assumptions & Reproduction Notes").

- EMCCF (Wang et al., 2024): 2 signals (velocity, direction) x 7 windows x 1
  stat each = 14-dim feature vector (mean speed, circular mean direction).
- EMCCF++ (HEMC extension): velocity, smoothed velocity, acceleration,
  directional change, dispersion (5 stats each: mean/std/max/P25/P90) plus
  direction (2 circular stats: circular mean, resultant length /
  "directional consistency" -- linear max/percentile of an angle is not
  physically meaningful).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

WINDOW_SCALES = [1 + 2**i for i in range(1, 8)]  # [3, 5, 9, 17, 33, 65, 129]


# --------------------------------------------------------------------------
# Point-wise kinematic signals
# --------------------------------------------------------------------------


def instantaneous_velocity(x: np.ndarray, y: np.ndarray, fs_hz: float) -> np.ndarray:
    """Speed magnitude via central-difference velocity, shape (T,)."""
    vx, vy = instantaneous_velocity_xy(x, y, fs_hz)
    return np.hypot(vx, vy)


def instantaneous_velocity_xy(x: np.ndarray, y: np.ndarray, fs_hz: float) -> tuple[np.ndarray, np.ndarray]:
    """Signed velocity components (vx, vy), each shape (T,)."""
    dt = 1.0 / fs_hz
    return np.gradient(x, dt), np.gradient(y, dt)


def smoothed_velocity(velocity: np.ndarray, window_length: int = 7, polyorder: int = 2) -> np.ndarray:
    """Savitzky-Golay smoothed velocity. Falls back to the raw signal if too short."""
    n = len(velocity)
    wl = min(window_length, n - (1 - n % 2))  # ensure wl <= n and odd
    if wl % 2 == 0:
        wl -= 1
    if wl <= polyorder or wl < 3:
        return velocity.copy()
    return savgol_filter(velocity, window_length=wl, polyorder=polyorder)


def instantaneous_acceleration(velocity: np.ndarray, fs_hz: float) -> np.ndarray:
    """First-order acceleration: central difference of the velocity magnitude signal."""
    dt = 1.0 / fs_hz
    return np.gradient(velocity, dt)


def instantaneous_direction(x: np.ndarray, y: np.ndarray, fs_hz: float) -> np.ndarray:
    """Direction angle in degrees, atan2(vy, vx), wrapped to (-180, 180]."""
    vx, vy = instantaneous_velocity_xy(x, y, fs_hz)
    return np.degrees(np.arctan2(vy, vx))


def directional_change(direction_deg: np.ndarray) -> np.ndarray:
    """Frame-to-frame signed angular difference, wrapped to (-180, 180], shape (T,)."""
    diff = np.diff(direction_deg, prepend=direction_deg[0])
    return (diff + 180.0) % 360.0 - 180.0


def local_dispersion(x: np.ndarray, y: np.ndarray, window: int = 5) -> np.ndarray:
    """I-DT-style local spatial dispersion: (max-min span in x) + (max-min span in y)
    over a small centered window of `window` samples, shape (T,)."""
    n = len(x)
    if window < 1:
        raise ValueError("window must be >= 1")
    half = window // 2
    x_pad = np.pad(x, (half, half), mode="edge")
    y_pad = np.pad(y, (half, half), mode="edge")
    x_windows = np.lib.stride_tricks.sliding_window_view(x_pad, window)
    y_windows = np.lib.stride_tricks.sliding_window_view(y_pad, window)
    span_x = x_windows.max(axis=-1) - x_windows.min(axis=-1)
    span_y = y_windows.max(axis=-1) - y_windows.min(axis=-1)
    return (span_x + span_y)[:n]


def compute_point_wise_signals(x: np.ndarray, y: np.ndarray, fs_hz: float, dispersion_window: int = 5) -> dict[str, np.ndarray]:
    """Compute all point-wise kinematic signals used by both EMCCF and EMCCF++ feature sets."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    velocity = instantaneous_velocity(x, y, fs_hz)
    direction = instantaneous_direction(x, y, fs_hz)
    return {
        "velocity": velocity,
        "velocity_smooth": smoothed_velocity(velocity),
        "acceleration": instantaneous_acceleration(velocity, fs_hz),
        "direction": direction,
        "directional_change": directional_change(direction),
        "dispersion": local_dispersion(x, y, window=dispersion_window),
    }


# --------------------------------------------------------------------------
# Multi-scale centered-window aggregation
# --------------------------------------------------------------------------


def _centered_windows(signal: np.ndarray, window: int) -> np.ndarray:
    """Return shape (T, window): for every sample t, the `window` values centered on t
    (edge-reflect padded so every sample gets a full-length window)."""
    if window % 2 == 0:
        raise ValueError(f"window size must be odd (centered), got {window}")
    n = len(signal)
    half = window // 2
    padded = np.pad(signal, (half, half), mode="reflect")
    return np.lib.stride_tricks.sliding_window_view(padded, window)[:n]


def aggregate_linear(signal: np.ndarray, window_scales: list[int] = WINDOW_SCALES) -> dict[int, dict[str, np.ndarray]]:
    """For each window scale, {mean, std, max, p25, p90} over the centered window."""
    out: dict[int, dict[str, np.ndarray]] = {}
    for w in window_scales:
        windows = _centered_windows(signal, w)
        out[w] = {
            "mean": windows.mean(axis=1),
            "std": windows.std(axis=1),
            "max": windows.max(axis=1),
            "p25": np.percentile(windows, 25, axis=1),
            "p90": np.percentile(windows, 90, axis=1),
        }
    return out


def aggregate_circular(direction_deg: np.ndarray, window_scales: list[int] = WINDOW_SCALES) -> dict[int, dict[str, np.ndarray]]:
    """For each window scale: circular mean direction + resultant length R in [0, 1]
    ("directional consistency" -- R=1 all samples point the same way, R->0 scattered)."""
    rad = np.radians(direction_deg)
    sin_r, cos_r = np.sin(rad), np.cos(rad)
    out: dict[int, dict[str, np.ndarray]] = {}
    for w in window_scales:
        sin_w = _centered_windows(sin_r, w).mean(axis=1)
        cos_w = _centered_windows(cos_r, w).mean(axis=1)
        out[w] = {"circmean": np.degrees(np.arctan2(sin_w, cos_w)), "consistency": np.hypot(sin_w, cos_w)}
    return out


def stack_features(aggregated: dict[str, dict[int, dict[str, np.ndarray]]]) -> tuple[np.ndarray, list[str]]:
    """Flatten {signal_name: {window: {stat: (T,)}}} into a (T, D) matrix + deterministic column names."""
    columns: list[str] = []
    arrays: list[np.ndarray] = []
    for signal_name in sorted(aggregated.keys()):
        by_window = aggregated[signal_name]
        for w in sorted(by_window.keys()):
            for stat_name, values in by_window[w].items():
                columns.append(f"{signal_name}_w{w}_{stat_name}")
                arrays.append(values)
    return np.column_stack(arrays), columns


# --------------------------------------------------------------------------
# EMCCF (Wang et al., 2024) -- 14-dim: velocity + direction, 7 windows, 1 stat each
# --------------------------------------------------------------------------

N_WINDOWS = 7
assert len(WINDOW_SCALES) == N_WINDOWS


def extract_emccf_features(x: np.ndarray, y: np.ndarray, fs_hz: float) -> tuple[np.ndarray, list[str]]:
    """Return (features (T, 2w), column_names) for one recording's raw (x, y) trace."""
    velocity = instantaneous_velocity(x, y, fs_hz)
    direction = instantaneous_direction(x, y, fs_hz)
    vel_agg = aggregate_linear(velocity, WINDOW_SCALES)
    dir_agg = aggregate_circular(direction, WINDOW_SCALES)

    columns: list[str] = []
    arrays: list[np.ndarray] = []
    for w in WINDOW_SCALES:
        columns.append(f"velocity_w{w}")
        arrays.append(vel_agg[w]["mean"])
    for w in WINDOW_SCALES:
        columns.append(f"direction_w{w}")
        arrays.append(dir_agg[w]["circmean"])
    return np.column_stack(arrays), columns


def extract_emccf_features_df(df: pd.DataFrame, fs_hz: float) -> pd.DataFrame:
    """Convenience wrapper: df must have 'x', 'y' columns; returns a features-only DataFrame."""
    feats, columns = extract_emccf_features(df["x"].to_numpy(), df["y"].to_numpy(), fs_hz)
    return pd.DataFrame(feats, columns=columns, index=df.index)


# --------------------------------------------------------------------------
# EMCCF++ (HEMC extension) -- richer kinematic feature set
# --------------------------------------------------------------------------

LINEAR_SIGNALS = ["velocity", "velocity_smooth", "acceleration", "directional_change", "dispersion"]
CIRCULAR_SIGNALS = ["direction"]


def extract_emccfpp_features(x: np.ndarray, y: np.ndarray, fs_hz: float, window_scales: list[int] = WINDOW_SCALES) -> tuple[np.ndarray, list[str]]:
    """Return (features (T, D), column_names) for one recording's raw (x, y) trace."""
    signals = compute_point_wise_signals(x, y, fs_hz)

    aggregated_linear = {name: aggregate_linear(signals[name], window_scales) for name in LINEAR_SIGNALS}
    linear_feats, linear_columns = stack_features(aggregated_linear)

    aggregated_circular = {name: aggregate_circular(signals[name], window_scales) for name in CIRCULAR_SIGNALS}
    circular_feats, circular_columns = stack_features(aggregated_circular)

    features = np.column_stack([linear_feats, circular_feats])
    return features, linear_columns + circular_columns


def extract_emccfpp_features_df(df: pd.DataFrame, fs_hz: float) -> pd.DataFrame:
    """Convenience wrapper: df must have 'x', 'y' columns; returns a features-only DataFrame."""
    feats, columns = extract_emccfpp_features(df["x"].to_numpy(), df["y"].to_numpy(), fs_hz)
    return pd.DataFrame(feats, columns=columns, index=df.index)


def feature_group_of(column_name: str) -> str:
    """Map a feature column name back to its originating signal, for importance-analysis grouping."""
    for name in LINEAR_SIGNALS + CIRCULAR_SIGNALS:
        if column_name.startswith(f"{name}_w"):
            return name
    raise ValueError(f"Unrecognized column name: {column_name}")


# --------------------------------------------------------------------------
# Auxiliary ocular-signal features (eye-tracker-provided, dataset-optional)
# --------------------------------------------------------------------------
# Not part of the original EMCCF/EMCCF++ formulation, which only ever sees
# (x, y). Some eye trackers (e.g. Pupil Labs Neon, used by the RTC dataset)
# additionally expose pupil diameter and eyelid aperture per sample. The
# reference paper's annotation pipeline (Sec 2.2.2, Step 0) identifies
# exactly these two signals as the discriminative cue for blinks -- there
# used for an unsupervised k-means pre-annotation step, not fed to the
# classifier. Here they're turned into actual classifier features, reusing
# the same multi-scale mean/std/max/P25/P90 aggregation recipe as EMCCF++
# (Sec 3.1.1) so they compose naturally with the rest of the feature vector.


def extract_auxiliary_signal_features(
    signals: dict[str, np.ndarray],
    fs_hz: float,
    window_scales: list[int] = WINDOW_SCALES,
    with_rate: set[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Multi-scale {mean, std, max, p25, p90} aggregation (EMCCF++ recipe) applied to
    arbitrary named point-wise signals. For names listed in `with_rate`, the signal's
    first-order rate of change (central difference, same treatment as `acceleration` for
    velocity in EMCCF++) is aggregated too, e.g. to capture how fast an eyelid is
    closing/opening rather than just how open it is at each scale."""
    with_rate = with_rate or set()
    dt = 1.0 / fs_hz
    expanded: dict[str, np.ndarray] = {}
    for name, sig in signals.items():
        sig = np.asarray(sig, dtype=np.float64)
        expanded[name] = sig
        if name in with_rate:
            expanded[f"{name}_rate"] = np.gradient(sig, dt)
    aggregated = {name: aggregate_linear(sig, window_scales) for name, sig in expanded.items()}
    return stack_features(aggregated)


def extract_auxiliary_signal_features_df(
    df: pd.DataFrame,
    signal_columns: list[str],
    fs_hz: float,
    window_scales: list[int] = WINDOW_SCALES,
    with_rate: set[str] | None = None,
) -> pd.DataFrame:
    """Convenience wrapper: `df` must have all of `signal_columns`; returns a features-only
    DataFrame aligned on `df.index`, ready to `pd.concat(..., axis=1)` onto EMCCF++ features."""
    signals = {col: df[col].to_numpy() for col in signal_columns}
    feats, columns = extract_auxiliary_signal_features(signals, fs_hz, window_scales, with_rate)
    return pd.DataFrame(feats, columns=columns, index=df.index)
