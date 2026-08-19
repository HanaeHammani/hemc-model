import numpy as np

from hemc.stage2 import (
    agreement_report,
    build_stage2_labels,
    detect_microsaccades_engbert_kliegl,
    detect_microsaccades_sheynikhovich,
)


def _synthetic_fixation_with_microsaccades(fs_hz=200.0, n=1000, spike_centers=(100, 300, 500, 700, 900), seed=0):
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.normal(0, 0.0003, n))
    y = np.cumsum(rng.normal(0, 0.0003, n))

    # inject Gaussian-bump "microsaccade" excursions: smooth, bell-shaped velocity profile
    half_width = 4
    t_local = np.arange(-half_width, half_width + 1)
    bump_shape = np.exp(-0.5 * (t_local / 1.5) ** 2)
    step = np.cumsum(bump_shape) / np.cumsum(bump_shape)[-1] * 0.03  # ~0.03 unit excursion

    for center in spike_centers:
        lo, hi = center - half_width, center + half_width + 1
        if 0 <= lo and hi <= n:
            x[lo:hi] += step
            x[hi:] += step[-1]

    return x, y


def test_engbert_kliegl_flags_injected_spikes():
    fs_hz = 200.0
    spike_centers = [100, 300, 500, 700, 900]
    x, y = _synthetic_fixation_with_microsaccades(fs_hz=fs_hz, spike_centers=spike_centers)

    mask = detect_microsaccades_engbert_kliegl(x, y, fs_hz)
    assert mask.dtype == bool
    assert len(mask) == len(x)

    for center in spike_centers:
        window = mask[max(0, center - 6) : center + 6]
        assert window.any(), f"expected a detection near injected spike at {center}"

    # detections should be a small minority of samples, not everything
    assert mask.mean() < 0.15


def test_sheynikhovich_flags_injected_spikes():
    fs_hz = 200.0
    spike_centers = [100, 300, 500, 700, 900]
    x, y = _synthetic_fixation_with_microsaccades(fs_hz=fs_hz, spike_centers=spike_centers)

    mask = detect_microsaccades_sheynikhovich(x, y, fs_hz)
    assert mask.dtype == bool
    assert len(mask) == len(x)

    for center in spike_centers:
        window = mask[max(0, center - 8) : center + 8]
        assert window.any(), f"expected a detection near injected spike at {center}"

    assert mask.mean() < 0.15


def test_pure_noise_has_few_false_positives():
    fs_hz = 200.0
    x, y = _synthetic_fixation_with_microsaccades(fs_hz=fs_hz, spike_centers=())  # no injected spikes
    mask_ek = detect_microsaccades_engbert_kliegl(x, y, fs_hz)
    mask_sh = detect_microsaccades_sheynikhovich(x, y, fs_hz)
    assert mask_ek.mean() < 0.1
    assert mask_sh.mean() < 0.1


def test_build_stage2_labels_and_agreement():
    fs_hz = 200.0
    n = 1000
    x, y = _synthetic_fixation_with_microsaccades(fs_hz=fs_hz, n=n)
    is_fixation = np.ones(n, dtype=bool)

    labels_a = build_stage2_labels(x, y, fs_hz, is_fixation, method="engbert_kliegl")
    labels_b = build_stage2_labels(x, y, fs_hz, is_fixation, method="sheynikhovich")

    assert set(np.unique(labels_a)) <= {"MS", "DRIFT"}
    assert set(np.unique(labels_b)) <= {"MS", "DRIFT"}

    report = agreement_report(labels_a, labels_b)
    assert report["n_compared"] == n
    assert -1.0 <= report["kappa"] <= 1.0
