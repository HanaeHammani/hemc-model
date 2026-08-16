import numpy as np

from hemc.features import (
    compute_point_wise_signals,
    directional_change,
    instantaneous_direction,
    instantaneous_velocity,
    local_dispersion,
)


def test_constant_velocity_straight_line():
    fs = 100.0
    n = 200
    t = np.arange(n) / fs
    vx0, vy0 = 3.0, -4.0  # 3-4-5 triangle, speed = 5
    x = vx0 * t
    y = vy0 * t

    velocity = instantaneous_velocity(x, y, fs)
    assert np.allclose(velocity, 5.0, atol=1e-6)

    direction = instantaneous_direction(x, y, fs)
    expected_dir = np.degrees(np.arctan2(vy0, vx0))
    assert np.allclose(direction, expected_dir, atol=1e-4)

    change = directional_change(direction)
    assert np.allclose(change, 0.0, atol=1e-4)


def test_circular_arc_constant_speed():
    fs = 200.0
    n = 400
    t = np.arange(n) / fs
    r, w = 2.0, 3.0  # rad/s
    x = r * np.cos(w * t)
    y = r * np.sin(w * t)

    velocity = instantaneous_velocity(x, y, fs)
    expected_speed = r * w
    # exclude the very first/last few samples where edge effects are largest
    assert np.allclose(velocity[5:-5], expected_speed, rtol=0.02)


def test_local_dispersion_zero_for_constant_position():
    x = np.full(50, 1.23)
    y = np.full(50, -4.56)
    dispersion = local_dispersion(x, y, window=5)
    assert np.allclose(dispersion, 0.0)
    assert len(dispersion) == 50


def test_local_dispersion_detects_spread():
    n = 30
    x = np.zeros(n)
    y = np.zeros(n)
    x[15] = 10.0  # single spike
    dispersion = local_dispersion(x, y, window=5)
    assert dispersion[15] > 0
    assert dispersion[0] == 0.0  # far from the spike


def test_compute_point_wise_signals_shapes():
    fs = 250.0
    n = 500
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.normal(0, 0.01, n))
    y = np.cumsum(rng.normal(0, 0.01, n))
    signals = compute_point_wise_signals(x, y, fs)
    expected_keys = {"velocity", "velocity_smooth", "acceleration", "direction", "directional_change", "dispersion"}
    assert set(signals.keys()) == expected_keys
    for arr in signals.values():
        assert arr.shape == (n,)
        assert np.all(np.isfinite(arr))
