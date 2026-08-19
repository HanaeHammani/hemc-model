"""
Pipeline: raw (x, y) -> point-wise kinematic signals (velocity,
direction, dispersion, ...) -> each signal aggregated over multi-scale
centered windows -> one feature vector per sample

Window scheme : n_i = 1 + 2^i, i=1..7 -> spans [3, 5, 9, 17, 33,
65, 129] samples, centered on each sample 

- EMCCF++ : velocity, smoothed velocity, acceleration,
  directional change, dispersion (5 stats each: mean/std/max/P25/P90) plus
  direction (2 circular stats: circular mean, resultant length /
  "directional consistency" )
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
    vx, vy = instantaneous_velocity_xy(x, y, fs_hz)
    return np.hypot(vx, vy)


def instantaneous_velocity_xy(x: np.ndarray, y: np.ndarray, fs_hz: float) -> tuple[np.ndarray, np.ndarray]:
    dt = 1.0 / fs_hz
    return np.gradient(x, dt), np.gradient(y, dt)


def smoothed_velocity(velocity: np.ndarray, window_length: int = 7, polyorder: int = 2) -> np.ndarray:
    n = len(velocity)
    wl = min(window_length, n - (1 - n % 2))  # ensure wl <= n and odd
    if wl % 2 == 0:
        wl -= 1
    if wl <= polyorder or wl < 3:
        return velocity.copy()
    return savgol_filter(velocity, window_length=wl, polyorder=polyorder)


def instantaneous_acceleration(velocity: np.ndarray, fs_hz: float) -> np.ndarray:
    dt = 1.0 / fs_hz
    return np.gradient(velocity, dt)


def instantaneous_direction(x: np.ndarray, y: np.ndarray, fs_hz: float) -> np.ndarray:
    vx, vy = instantaneous_velocity_xy(x, y, fs_hz)
    return np.degrees(np.arctan2(vy, vx))


def directional_change(direction_deg: np.ndarray) -> np.ndarray:
    diff = np.diff(direction_deg, prepend=direction_deg[0])
    return (diff + 180.0) % 360.0 - 180.0


def local_dispersion(x: np.ndarray, y: np.ndarray, window: int = 5) -> np.ndarray:
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
    if window % 2 == 0:
        raise ValueError(f"window size must be odd (centered), got {window}")
    n = len(signal)
    half = window // 2
    padded = np.pad(signal, (half, half), mode="reflect")
    return np.lib.stride_tricks.sliding_window_view(padded, window)[:n]


def aggregate_linear(signal: np.ndarray, window_scales: list[int] = WINDOW_SCALES) -> dict[int, dict[str, np.ndarray]]:
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
    rad = np.radians(direction_deg)
    sin_r, cos_r = np.sin(rad), np.cos(rad)
    out: dict[int, dict[str, np.ndarray]] = {}
    for w in window_scales:
        sin_w = _centered_windows(sin_r, w).mean(axis=1)
        cos_w = _centered_windows(cos_r, w).mean(axis=1)
        out[w] = {"circmean": np.degrees(np.arctan2(sin_w, cos_w)), "consistency": np.hypot(sin_w, cos_w)}
    return out


def stack_features(aggregated: dict[str, dict[int, dict[str, np.ndarray]]]) -> tuple[np.ndarray, list[str]]:
    columns: list[str] = []
    arrays: list[np.ndarray] = []
    for signal_name in sorted(aggregated.keys()):
        by_window = aggregated[signal_name]
        for w in sorted(by_window.keys()):
            for stat_name, values in by_window[w].items():
                columns.append(f"{signal_name}_w{w}_{stat_name}")
                arrays.append(values)
    return np.column_stack(arrays), columns



N_WINDOWS = 7
assert len(WINDOW_SCALES) == N_WINDOWS


def extract_emccf_features(x: np.ndarray, y: np.ndarray, fs_hz: float) -> tuple[np.ndarray, list[str]]:
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
    feats, columns = extract_emccf_features(df["x"].to_numpy(), df["y"].to_numpy(), fs_hz)
    return pd.DataFrame(feats, columns=columns, index=df.index)



LINEAR_SIGNALS = ["velocity", "velocity_smooth", "acceleration", "directional_change", "dispersion"]
CIRCULAR_SIGNALS = ["direction"]


def extract_emccfpp_features(x: np.ndarray, y: np.ndarray, fs_hz: float, window_scales: list[int] = WINDOW_SCALES) -> tuple[np.ndarray, list[str]]:
    signals = compute_point_wise_signals(x, y, fs_hz)

    aggregated_linear = {name: aggregate_linear(signals[name], window_scales) for name in LINEAR_SIGNALS}
    linear_feats, linear_columns = stack_features(aggregated_linear)

    aggregated_circular = {name: aggregate_circular(signals[name], window_scales) for name in CIRCULAR_SIGNALS}
    circular_feats, circular_columns = stack_features(aggregated_circular)

    features = np.column_stack([linear_feats, circular_feats])
    return features, linear_columns + circular_columns


def extract_emccfpp_features_df(df: pd.DataFrame, fs_hz: float) -> pd.DataFrame:
    feats, columns = extract_emccfpp_features(df["x"].to_numpy(), df["y"].to_numpy(), fs_hz)
    return pd.DataFrame(feats, columns=columns, index=df.index)


def feature_group_of(column_name: str) -> str:
    for name in LINEAR_SIGNALS + CIRCULAR_SIGNALS:
        if column_name.startswith(f"{name}_w"):
            return name
    raise ValueError(f"Unrecognized column name: {column_name}")



def extract_auxiliary_signal_features(
    signals: dict[str, np.ndarray],
    fs_hz: float,
    window_scales: list[int] = WINDOW_SCALES,
    with_rate: set[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
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
    
    signals = {col: df[col].to_numpy() for col in signal_columns}
    feats, columns = extract_auxiliary_signal_features(signals, fs_hz, window_scales, with_rate)
    return pd.DataFrame(feats, columns=columns, index=df.index)
