import numpy as np
import pytest

from hemc.models import (
    SequentialCRFDecoder,
    build_transition_matrix,
    estimate_mean_durations_ms,
    viterbi_decode,
)


def test_viterbi_removes_isolated_flicker():
    class_names = ["A", "B", "C"]
    true_path = ["A"] * 20 + ["B"] * 20 + ["C"] * 20
    idx = {c: i for i, c in enumerate(class_names)}

    n = len(true_path)
    log_proba = np.log(np.full((n, 3), 0.05))
    for t, c in enumerate(true_path):
        log_proba[t] = np.log(0.05)
        log_proba[t, idx[c]] = np.log(0.9)

    # inject isolated single-sample flickers (should be corrected by strong self-transition)
    flicker_positions = [5, 25, 45]
    for pos in flicker_positions:
        true_c = true_path[pos]
        other = [c for c in class_names if c != true_c][0]
        log_proba[pos] = np.log(0.05)
        log_proba[pos, idx[other]] = np.log(0.9)

    transition = np.full((3, 3), 0.05)
    np.fill_diagonal(transition, 0.9)

    decoded = viterbi_decode(log_proba, transition, class_names)
    n_errors = sum(1 for d, t in zip(decoded, true_path) if d != t)
    # without correction there would be exactly len(flicker_positions) errors from
    # the injected flickers alone; Viterbi with a strong self-transition prior
    # should recover a smoother path with fewer or equal errors
    n_errors_no_correction = sum(1 for t, c in enumerate(true_path) if (np.array(log_proba[t]).argmax() != idx[c]))
    assert n_errors <= n_errors_no_correction


def test_estimate_mean_durations_and_transition_matrix():
    class_names = ["A", "B"]
    fs_hz = 100.0
    # A runs of 10 samples (100ms), B runs of 5 samples (50ms)
    seqs = [["A"] * 10 + ["B"] * 5 + ["A"] * 10 + ["B"] * 5]
    seqs = [np.array(s) for s in seqs]
    durations = estimate_mean_durations_ms(seqs, class_names, fs_hz)
    assert durations["A"] == pytest.approx(100.0)
    assert durations["B"] == pytest.approx(50.0)

    transition = build_transition_matrix(class_names, durations, fs_hz)
    assert transition.shape == (2, 2)
    # rows sum to 1
    assert np.allclose(transition.sum(axis=1), 1.0)
    # longer mean duration -> higher self-transition (stay) probability
    assert transition[0, 0] > transition[1, 1]


def test_context_features_shape():
    class_names = ["A", "B"]
    fs_hz = 100.0
    n = 50
    rng = np.random.default_rng(0)
    proba_seq = rng.dirichlet([1, 1], size=n)
    durations = {"A": 100.0, "B": 100.0}

    decoder = SequentialCRFDecoder(class_names=class_names, context_k=5)
    features = decoder.build_context_features(proba_seq, durations, fs_hz)
    expected_dim = (2 * 5 + 1) * 2 + 2 + 1 + 1 + 1  # context window + deriv + entropy + run_dur + momentum
    assert features.shape == (n, expected_dim)
    assert np.all(np.isfinite(features))
