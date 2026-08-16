import numpy as np

from hemc.features import WINDOW_SCALES, aggregate_circular, aggregate_linear, stack_features


def test_aggregate_linear_shapes_and_constant_signal():
    n = 300
    signal = np.full(n, 7.0)
    agg = aggregate_linear(signal)
    assert set(agg.keys()) == set(WINDOW_SCALES)
    for w in WINDOW_SCALES:
        stats = agg[w]
        assert set(stats.keys()) == {"mean", "std", "max", "p25", "p90"}
        for name, values in stats.items():
            assert values.shape == (n,)
            if name == "std":
                assert np.allclose(values, 0.0)
            else:
                assert np.allclose(values, 7.0)


def test_aggregate_linear_edge_padding_no_nans():
    n = 50
    rng = np.random.default_rng(1)
    signal = rng.normal(size=n)
    agg = aggregate_linear(signal, window_scales=[3, 9, 33])
    for w, stats in agg.items():
        for values in stats.values():
            assert values.shape == (n,)
            assert np.all(np.isfinite(values))


def test_aggregate_circular_constant_direction():
    n = 100
    direction = np.full(n, 45.0)
    agg = aggregate_circular(direction, window_scales=[3, 9])
    for w, stats in agg.items():
        assert np.allclose(stats["circmean"], 45.0, atol=1e-6)
        assert np.allclose(stats["consistency"], 1.0, atol=1e-6)


def test_aggregate_circular_wraparound():
    # directions alternating near +179 / -179 should average to ~180 (or -180), not ~0
    n = 40
    direction = np.where(np.arange(n) % 2 == 0, 179.0, -179.0)
    agg = aggregate_circular(direction, window_scales=[5])
    circmean = agg[5]["circmean"]
    assert np.all(np.abs(np.abs(circmean) - 180.0) < 5.0)


def test_stack_features_column_count():
    n = 60
    signal_a = np.linspace(0, 1, n)
    signal_b = np.linspace(1, 0, n)
    aggregated = {
        "a": aggregate_linear(signal_a, window_scales=[3, 5]),
        "b": aggregate_linear(signal_b, window_scales=[3, 5]),
    }
    feats, columns = stack_features(aggregated)
    # 2 signals x 2 windows x 5 stats = 20 columns
    assert feats.shape == (n, 20)
    assert len(columns) == 20
    assert len(set(columns)) == 20  # all unique
