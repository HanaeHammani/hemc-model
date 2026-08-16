"""Stage 2: microsaccade vs. drift classification within Stage-1-identified fixations.

Pipeline: `extract_windows` slices overlapping time windows out of fixation
segments -> `build_stage2_labels` assigns each fixation sample a MS/DRIFT
pseudo-label (no expert-validated ground truth is available on the public
datasets used here -- see README "Documented Assumptions") -> `HardNegativeMiner`
selects the drift windows that most resemble microsaccades, to counter the
~63:1 class imbalance, before training the ResNet-TS classifier (models.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import groupby

import numpy as np
from scipy.signal import find_peaks
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from hemc.features import instantaneous_velocity, instantaneous_velocity_xy

MICROSACCADE = "MS"
DRIFT = "DRIFT"
NOT_APPLICABLE = ""


# ==========================================================================
# Windowing (HEMC article Section 3.2.1: overlapping windows, stride = 0.6 * window_size)
# ==========================================================================


def build_channels_monocular(x: np.ndarray, y: np.ndarray, fs_hz: float) -> tuple[np.ndarray, list[str]]:
    """Columns: x, y, vx, vy, v, |acceleration| -- shape (T, 6).

    Matches the 6-channel window construction of the article's reference
    Stage 2 notebook (`to_6_channels`): position, per-axis velocity, speed
    magnitude, and absolute acceleration of the speed signal.
    """
    vx, vy = instantaneous_velocity_xy(x, y, fs_hz)
    v = np.hypot(vx, vy)
    acc = np.abs(np.gradient(v) * fs_hz)
    return np.column_stack([x, y, vx, vy, v, acc]), ["x", "y", "vx", "vy", "v", "acc"]


def build_channels_binocular(x_l: np.ndarray, y_l: np.ndarray, x_r: np.ndarray, y_r: np.ndarray, fs_hz: float) -> tuple[np.ndarray, list[str]]:
    """Per-eye + mean-eye channels: x,y,vx,vy,v,acc for left, right, and mean -- shape (T, 18)."""
    left, _ = build_channels_monocular(x_l, y_l, fs_hz)
    right, _ = build_channels_monocular(x_r, y_r, fs_hz)
    x_m, y_m = (x_l + x_r) / 2.0, (y_l + y_r) / 2.0
    mean_eye, _ = build_channels_monocular(x_m, y_m, fs_hz)
    base = ["x", "y", "vx", "vy", "v", "acc"]
    columns = [f"{c}_l" for c in base] + [f"{c}_r" for c in base] + [f"{c}_mean" for c in base]
    return np.column_stack([left, right, mean_eye]), columns


def _true_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous (start, end) index ranges [start, end) where mask is True."""
    segments, idx = [], 0
    for value, group in groupby(mask):
        length = sum(1 for _ in group)
        if value:
            segments.append((idx, idx + length))
        idx += length
    return segments


def extract_windows(channels: np.ndarray, center_labels: np.ndarray, fixation_mask: np.ndarray, window_size: int, stride_fraction: float = 0.6) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract centered, overlapping windows fully contained within one contiguous fixation segment.

    Returns (X (N, window_size, C), y (N,) center-sample labels, center_indices (N,)).
    """
    half = window_size // 2
    stride = max(int(round(window_size * stride_fraction)), 1)
    X_list, y_list, centers = [], [], []
    for start, end in _true_segments(np.asarray(fixation_mask, dtype=bool)):
        seg_len = end - start
        if seg_len < window_size:
            continue
        first_center, last_center = start + half, end - 1 - half
        for center in range(first_center, last_center + 1, stride):
            w_start, w_end = center - half, center + half + 1
            X_list.append(channels[w_start:w_end])
            y_list.append(center_labels[center])
            centers.append(center)
    if not X_list:
        n_channels = channels.shape[1]
        return np.empty((0, window_size, n_channels)), np.empty((0,), dtype=object), np.empty((0,), dtype=int)
    return np.stack(X_list), np.array(y_list, dtype=object), np.array(centers, dtype=int)


# ==========================================================================
# Microsaccade / drift pseudo-labeling
# ==========================================================================
# GazeCom and HMR only carry the 4 macro classes (F/S/P/B); neither has
# expert-validated microsaccade/drift labels (those came from the private
# surgical dataset in the HEMC article). We regenerate labels using the
# *same* semi-automated methodology the article defines (Section 2.2.2, Step
# 2): a correlation-based detector following Sheynikhovich et al. (2018). A
# second, independent detector (Engbert & Kliegl 2003) is provided as a
# cross-check; `agreement_report` computes Cohen's kappa between the two as a
# lightweight substitute for human inter-annotator agreement.


def _fixation_segments(is_fixation: np.ndarray, min_length: int = 5) -> list[tuple[int, int]]:
    segments, idx = [], 0
    for value, group in groupby(is_fixation):
        length = sum(1 for _ in group)
        if value and length >= min_length:
            segments.append((idx, idx + length))
        idx += length
    return segments


def detect_microsaccades_engbert_kliegl(x: np.ndarray, y: np.ndarray, fs_hz: float, lam: float = 6.0, min_duration_ms: float = 6.0, max_duration_ms: float = 100.0) -> np.ndarray:
    """Engbert & Kliegl (2003) adaptive elliptical velocity threshold. Returns boolean mask, shape (T,)."""
    vx, vy = instantaneous_velocity_xy(x, y, fs_hz)
    sigma_x = np.sqrt(max(np.median(vx**2) - np.median(vx) ** 2, 1e-12))
    sigma_y = np.sqrt(max(np.median(vy**2) - np.median(vy) ** 2, 1e-12))
    ellipse = (vx / (lam * sigma_x)) ** 2 + (vy / (lam * sigma_y)) ** 2
    above = ellipse > 1.0

    min_len = max(int(round(min_duration_ms / 1000.0 * fs_hz)), 1)
    max_len = max(int(round(max_duration_ms / 1000.0 * fs_hz)), min_len)

    mask, idx = np.zeros(len(x), dtype=bool), 0
    for value, group in groupby(above):
        length = sum(1 for _ in group)
        if value and min_len <= length <= max_len:
            mask[idx : idx + length] = True
        idx += length
    return mask


def detect_microsaccades_sheynikhovich(
    x: np.ndarray, y: np.ndarray, fs_hz: float, corr_thr: float = 0.7, min_duration_ms: float = 6.0, max_duration_ms: float = 30.0,
    profile_half_window_ms: float = 15.0, peak_mad_k: float = 3.0, peak_mad_k_seed: float = 6.0,
    template_top_fraction: float = 0.3, min_template_seeds: int = 3,
) -> np.ndarray:
    """Correlation-based microsaccade detector (Sheynikhovich et al., 2018 style). Boolean mask, shape (T,)."""
    n = len(x)
    velocity = instantaneous_velocity(x, y, fs_hz)
    half_window = max(int(round(profile_half_window_ms / 1000.0 * fs_hz)), 1)
    min_isi = 2 * half_window

    median_v = np.median(velocity)
    mad_v = np.median(np.abs(velocity - median_v)) + 1e-12
    threshold = median_v + peak_mad_k * mad_v

    if n < 2 * half_window + 1:
        return np.zeros(n, dtype=bool)

    peak_idx, _ = find_peaks(velocity, height=threshold, distance=max(min_isi, 1))
    peak_idx = peak_idx[(peak_idx >= half_window) & (peak_idx < n - half_window)]
    if len(peak_idx) == 0:
        return np.zeros(n, dtype=bool)

    profiles = np.stack([velocity[p - half_window : p + half_window + 1] for p in peak_idx])
    profile_ranges = profiles.max(axis=1) - profiles.min(axis=1)
    profile_ranges[profile_ranges == 0] = 1.0
    norm_profiles = (profiles - profiles.min(axis=1, keepdims=True)) / profile_ranges[:, None]

    # Template from high-confidence "seed" peaks only (stricter secondary threshold):
    # mixing in borderline candidates dilutes the template shape. Falls back to the
    # top-K candidates by amplitude if too few seeds clear the stricter threshold.
    seed_threshold = median_v + peak_mad_k_seed * mad_v
    seed_selector = velocity[peak_idx] >= seed_threshold
    if seed_selector.sum() >= min_template_seeds:
        template = norm_profiles[seed_selector].mean(axis=0)
    else:
        n_template = min(max(int(round(len(peak_idx) * template_top_fraction)), min_template_seeds), len(peak_idx))
        top_by_amplitude = np.argsort(velocity[peak_idx])[::-1][:n_template]
        template = norm_profiles[top_by_amplitude].mean(axis=0)
    template_centered = template - template.mean()
    template_norm = np.linalg.norm(template_centered) + 1e-12

    min_len = max(int(round(min_duration_ms / 1000.0 * fs_hz)), 1)
    max_len = max(int(round(max_duration_ms / 1000.0 * fs_hz)), min_len)

    mask = np.zeros(n, dtype=bool)
    for p, profile in zip(peak_idx, norm_profiles):
        prof_centered = profile - profile.mean()
        corr = float(np.dot(prof_centered, template_centered) / (np.linalg.norm(prof_centered) + 1e-12) / template_norm)
        if corr <= corr_thr:
            continue
        peak_amp = velocity[p]
        half_amp = 0.5 * peak_amp
        left = p
        while left > 0 and velocity[left - 1] > half_amp and (p - left) < max_len:
            left -= 1
        right = p
        while right < n - 1 and velocity[right + 1] > half_amp and (right - p) < max_len:
            right += 1
        length = right - left + 1
        if length < min_len:
            deficit = min_len - length
            left = max(0, left - deficit // 2)
            right = min(n - 1, right + (deficit - deficit // 2))
        mask[left : right + 1] = True
    return mask


def build_stage2_labels(x: np.ndarray, y: np.ndarray, fs_hz: float, is_fixation: np.ndarray, method: str = "sheynikhovich", min_fixation_length: int = 5, **detector_kwargs) -> np.ndarray:
    """Label every fixation sample as MICROSACCADE or DRIFT; non-fixation samples get "" (not applicable).

    Detection runs independently per contiguous fixation segment (a
    microsaccade profile from one fixation should not leak into the next).
    """
    detector = {"sheynikhovich": detect_microsaccades_sheynikhovich, "engbert_kliegl": detect_microsaccades_engbert_kliegl}[method]
    labels = np.full(len(x), NOT_APPLICABLE, dtype=object)
    for start, end in _fixation_segments(np.asarray(is_fixation, dtype=bool), min_length=min_fixation_length):
        seg_mask = detector(x[start:end], y[start:end], fs_hz, **detector_kwargs)
        labels[start:end] = np.where(seg_mask, MICROSACCADE, DRIFT)
    return labels


def agreement_report(labels_a: np.ndarray, labels_b: np.ndarray) -> dict:
    """Cohen's kappa + confusion matrix between two detectors' labels, restricted to positions
    where both labels are applicable (i.e. within fixation for both)."""
    valid = (labels_a != NOT_APPLICABLE) & (labels_b != NOT_APPLICABLE)
    a, b = labels_a[valid], labels_b[valid]
    if len(a) == 0:
        return {"kappa": float("nan"), "n_compared": 0, "confusion": None}
    kappa = cohen_kappa_score(a, b, labels=[MICROSACCADE, DRIFT])
    cm = confusion_matrix(a, b, labels=[MICROSACCADE, DRIFT])
    return {"kappa": kappa, "n_compared": int(len(a)), "confusion": cm}


# ==========================================================================
# Hard negative mining (HEMC article Section 3.2.2)
# ==========================================================================
# Microsaccades are rare vs. drift (~63:1). A preliminary model is trained on
# an 8:1 drift:microsaccade set, used to score all drift windows for
# similarity to microsaccades. Drift windows in the 10th-50th percentile of
# scores (descending -- excluding the top 10%, likely near-mislabeled
# outliers, and anything below the 50th percentile, trivially easy to
# separate) are the "hard negatives". The final training set combines all
# microsaccade windows with sampled hard negatives at a 4:1 ratio.


@dataclass
class HardNegativeMiningResult:
    X_hard_negatives: np.ndarray  # (n_selected, window_size, n_channels)
    selected_indices: np.ndarray  # indices into the original X_drift array
    scores: np.ndarray  # microsaccade-similarity score for every drift window
    preliminary_model: RandomForestClassifier


def _flatten(X: np.ndarray) -> np.ndarray:
    return X.reshape(X.shape[0], -1)


class HardNegativeMiner:
    def __init__(self, ratio_preliminary: float = 8.0, percentile_low: float = 10.0, percentile_high: float = 50.0, ratio_final: float = 4.0, random_state: int = 0, n_estimators: int = 200):
        self.ratio_preliminary = ratio_preliminary
        self.percentile_low = percentile_low
        self.percentile_high = percentile_high
        self.ratio_final = ratio_final
        self.random_state = random_state
        self.n_estimators = n_estimators

    def mine(self, X_ms: np.ndarray, X_drift: np.ndarray) -> HardNegativeMiningResult:
        rng = np.random.default_rng(self.random_state)
        n_ms = len(X_ms)
        if n_ms == 0:
            raise ValueError("No microsaccade windows available for hard negative mining")

        n_prelim_neg = min(len(X_drift), max(int(round(n_ms * self.ratio_preliminary)), 1))
        prelim_idx = rng.choice(len(X_drift), size=n_prelim_neg, replace=False)

        X_prelim = np.concatenate([_flatten(X_ms), _flatten(X_drift[prelim_idx])], axis=0)
        y_prelim = np.concatenate([np.ones(n_ms), np.zeros(n_prelim_neg)])

        prelim_model = RandomForestClassifier(n_estimators=self.n_estimators, class_weight="balanced", random_state=self.random_state, n_jobs=-1)
        prelim_model.fit(X_prelim, y_prelim)

        scores = prelim_model.predict_proba(_flatten(X_drift))[:, list(prelim_model.classes_).index(1.0)]
        order = np.argsort(-scores)  # descending: most microsaccade-like first
        n_total = len(order)
        lo = int(round(self.percentile_low / 100.0 * n_total))
        hi = int(round(self.percentile_high / 100.0 * n_total))
        band_idx = order[lo:hi] if hi > lo else order[lo : lo + 1]

        n_final_neg = max(int(round(n_ms * self.ratio_final)), 1)
        replace = len(band_idx) < n_final_neg
        selected = rng.choice(band_idx, size=min(n_final_neg, len(band_idx)) if not replace else n_final_neg, replace=replace)

        return HardNegativeMiningResult(X_hard_negatives=X_drift[selected], selected_indices=selected, scores=scores, preliminary_model=prelim_model)
