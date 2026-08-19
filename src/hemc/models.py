from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from itertools import groupby

import numpy as np
import pandas as pd
import torch
from torch import nn
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import KBinsDiscretizer
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", message="Bins whose width are too small.*", category=UserWarning)

_EPS = 1e-12


# ==========================================================================
# Cascade Forest 
# ==========================================================================
def _make_base_learners(n_estimators: int, n_classes: int, random_state: int, n_jobs: int) -> dict:
    return {
        "rf": RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=n_jobs),
        "erf": ExtraTreesClassifier(n_estimators=n_estimators, random_state=random_state + 1, n_jobs=n_jobs),
        "xgb": XGBClassifier(
            n_estimators=n_estimators, random_state=random_state + 2, n_jobs=n_jobs,
            objective="multi:softprob" if n_classes > 2 else "binary:logistic",
            eval_metric="mlogloss" if n_classes > 2 else "logloss", verbosity=0,
        ),
        "lgb": LGBMClassifier(n_estimators=n_estimators, random_state=random_state + 3, n_jobs=n_jobs, verbosity=-1),
    }


@dataclass
class _Layer:
    learners: dict = field(default_factory=dict)
    class_vector_discretizer: KBinsDiscretizer | None = None  # None for layer 0


class CascadeForestClassifier:
    
    def __init__(self, n_estimators_per_forest: int = 100, n_bins: int = 32, max_layers: int = 20,
                 cv_folds: int = 5, tol: float = 1e-4, random_state: int = 0, n_jobs: int = -1, verbose: bool = True):
        self.n_estimators_per_forest = n_estimators_per_forest
        self.n_bins = n_bins
        self.max_layers = max_layers
        self.cv_folds = cv_folds
        self.tol = tol
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

        self.classes_: np.ndarray | None = None
        self._orig_discretizer: KBinsDiscretizer | None = None
        self._layers: list[_Layer] = []
        self.score_history_: list[float] = []
        self.n_layers_: int = 0

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[CascadeForest] {msg}")

    def _bin_fit_transform(self, X: np.ndarray) -> tuple[np.ndarray, KBinsDiscretizer]:
        disc = KBinsDiscretizer(n_bins=self.n_bins, encode="ordinal", strategy="quantile", subsample=None)
        return disc.fit_transform(X).astype(np.float32), disc

    def _layer_class_vectors_oof(self, learners: dict, X: np.ndarray, y_idx: np.ndarray) -> np.ndarray:
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        vectors = [cross_val_predict(learner, X, y_idx, cv=cv, method="predict_proba", n_jobs=self.n_jobs) for learner in learners.values()]
        return np.concatenate(vectors, axis=1)

    def _layer_class_vectors_infer(self, learners: dict, X: np.ndarray) -> np.ndarray:
        return np.concatenate([learner.predict_proba(X) for learner in learners.values()], axis=1)

    def _averaged_proba(self, learners: dict, X: np.ndarray) -> np.ndarray:
        return np.stack([learner.predict_proba(X) for learner in learners.values()], axis=0).mean(axis=0)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> "CascadeForestClassifier":
        self.classes_ = np.unique(y_train)
        n_classes = len(self.classes_)
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_train_idx = np.array([class_to_idx[v] for v in y_train])
        y_val_idx = np.array([class_to_idx[v] for v in y_val])

        X_train_binned, self._orig_discretizer = self._bin_fit_transform(X_train)
        X_val_binned = self._orig_discretizer.transform(X_val).astype(np.float32)

        cur_train_input, cur_val_input = X_train_binned, X_val_binned
        self._layers, self.score_history_ = [], []
        best_score, best_n_layers = -np.inf, 0

        for layer_idx in range(self.max_layers):
            learners = _make_base_learners(self.n_estimators_per_forest, n_classes, self.random_state + layer_idx * 10, self.n_jobs)
            train_class_vectors = self._layer_class_vectors_oof(learners, cur_train_input, y_train_idx)
            for learner in learners.values():
                learner.fit(cur_train_input, y_train_idx)
            val_class_vectors = self._layer_class_vectors_infer(learners, cur_val_input)

            val_proba_avg = self._averaged_proba(learners, cur_val_input)
            val_pred = self.classes_[val_proba_avg.argmax(axis=1)]
            score = f1_score(self.classes_[y_val_idx], val_pred, average="macro", zero_division=0)
            self.score_history_.append(score)
            self._log(f"layer {layer_idx + 1}: val macro-F1 = {score:.4f}")

            layer = _Layer(learners=learners)
            self._layers.append(layer)

            if score > best_score + self.tol:
                best_score, best_n_layers = score, layer_idx + 1
            else:
                self._log(f"no improvement after layer {layer_idx + 1}, stopping at {best_n_layers} layers")
                break

            class_vec_disc = KBinsDiscretizer(n_bins=self.n_bins, encode="ordinal", strategy="quantile", subsample=None)
            train_class_vectors_binned = class_vec_disc.fit_transform(train_class_vectors).astype(np.float32)
            val_class_vectors_binned = class_vec_disc.transform(val_class_vectors).astype(np.float32)
            layer.class_vector_discretizer = class_vec_disc

            cur_train_input = np.concatenate([X_train_binned, train_class_vectors_binned], axis=1)
            cur_val_input = np.concatenate([X_val_binned, val_class_vectors_binned], axis=1)

        self._layers = self._layers[:best_n_layers]
        self.n_layers_ = best_n_layers
        self._log(f"final model: {self.n_layers_} layers, best val macro-F1 = {best_score:.4f}")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._orig_discretizer is None:
            raise RuntimeError("CascadeForestClassifier must be fit before predict_proba")
        X_binned = self._orig_discretizer.transform(X).astype(np.float32)
        cur_input = X_binned
        proba = None
        for layer in self._layers:
            class_vectors = self._layer_class_vectors_infer(layer.learners, cur_input)
            proba = self._averaged_proba(layer.learners, cur_input)
            if layer.class_vector_discretizer is not None:
                class_vectors_binned = layer.class_vector_discretizer.transform(class_vectors).astype(np.float32)
                cur_input = np.concatenate([X_binned, class_vectors_binned], axis=1)
        return proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[proba.argmax(axis=1)]


# ==========================================================================
# Sequential CRF-Viterbi decoder 
# ==========================================================================
def estimate_mean_durations_ms(label_seqs: list[np.ndarray], class_names: list[str], fs_hz: float) -> dict[str, float]:
    run_lengths: dict[str, list[int]] = {c: [] for c in class_names}
    for seq in label_seqs:
        for label, group in groupby(seq):
            length = sum(1 for _ in group)
            if str(label) in run_lengths:
                run_lengths[str(label)].append(length)
    out = {}
    for c in class_names:
        lengths = run_lengths[c]
        mean_samples = float(np.mean(lengths)) if lengths else 1.0
        out[c] = max(mean_samples / fs_hz * 1000.0, 1.0 / fs_hz * 1000.0 * 2)  # floor at 2 samples worth of ms
    return out


def estimate_transition_counts(label_seqs: list[np.ndarray], class_names: list[str]) -> pd.DataFrame:
    idx = {c: i for i, c in enumerate(class_names)}
    counts = np.zeros((len(class_names), len(class_names)))
    for seq in label_seqs:
        events = [str(label) for label, _ in groupby(seq)]
        for a, b in zip(events[:-1], events[1:]):
            if a in idx and b in idx:
                counts[idx[a], idx[b]] += 1
    return pd.DataFrame(counts, index=class_names, columns=class_names)


def build_transition_matrix(class_names: list[str], mean_durations_ms: dict[str, float], fs_hz: float, transition_counts: pd.DataFrame | None = None) -> np.ndarray:
    """transition matrix: P(stay|i) = (D_i*f/1000 - 1) / (D_i*f/1000)
    """
    n = len(class_names)
    mat = np.zeros((n, n))
    for i, c in enumerate(class_names):
        d_i_f_1000 = mean_durations_ms[c] * fs_hz / 1000.0
        p_stay = min(max((d_i_f_1000 - 1.0) / d_i_f_1000, 0.0), 1.0 - _EPS)
        mat[i, i] = p_stay
        remaining = 1.0 - p_stay
        others = [j for j in range(n) if j != i]
        if transition_counts is not None:
            row_others = np.array([transition_counts.loc[c].to_numpy()[j] for j in others])
            weights = row_others / row_others.sum() if row_others.sum() > 0 else np.ones(len(others)) / len(others)
        else:
            weights = np.ones(len(others)) / len(others)
        for j, w in zip(others, weights):
            mat[i, j] = remaining * w
    return mat


class SequentialCRFDecoder:

    def __init__(self, class_names: list[str], context_k: int = 5, momentum_window: int = 5, random_state: int = 0, max_iter: int = 3000):
        self.class_names = class_names
        self.context_k = context_k
        self.momentum_window = momentum_window
        self.random_state = random_state
        self.max_iter = max_iter
        self.clf: LogisticRegression | None = None

    def build_context_features(self, proba_seq: np.ndarray, mean_durations_ms: dict[str, float], fs_hz: float) -> np.ndarray:
        T, C = proba_seq.shape
        k = self.context_k

        padded = np.pad(proba_seq, ((k, k), (0, 0)), mode="edge")
        context_window = np.lib.stride_tricks.sliding_window_view(padded, (2 * k + 1, C))[:, 0, :, :]
        context_flat = context_window.reshape(T, (2 * k + 1) * C)

        deriv = np.diff(proba_seq, axis=0, prepend=proba_seq[:1])

        p_clipped = np.clip(proba_seq, _EPS, 1.0)
        entropy = -(p_clipped * np.log(p_clipped)).sum(axis=1) / np.log(C)

        argmax_idx = proba_seq.argmax(axis=1)
        mean_durations_samples = np.array([mean_durations_ms[c] * fs_hz / 1000.0 for c in self.class_names])
        run_duration_norm = self._run_duration_normalized(argmax_idx, mean_durations_samples)

        switches = np.concatenate([[0], (argmax_idx[1:] != argmax_idx[:-1]).astype(float)])
        momentum = pd.Series(switches).rolling(self.momentum_window, min_periods=1).sum().to_numpy()

        return np.column_stack([context_flat, deriv, entropy, run_duration_norm, momentum])

    @staticmethod
    def _run_duration_normalized(argmax_idx: np.ndarray, mean_durations_samples: np.ndarray) -> np.ndarray:
        T = len(argmax_idx)
        out = np.zeros(T)
        run_start = 0
        for t in range(1, T + 1):
            if t == T or argmax_idx[t] != argmax_idx[run_start]:
                cls_idx = argmax_idx[run_start]
                run_len = t - run_start
                out[run_start:t] = np.arange(1, run_len + 1) / mean_durations_samples[cls_idx]
                run_start = t
        return out

    def fit(self, X_context: np.ndarray, y: np.ndarray) -> "SequentialCRFDecoder":
        self.clf = LogisticRegression(solver="saga", class_weight="balanced", max_iter=self.max_iter, random_state=self.random_state)
        self.clf.fit(X_context, y)
        return self

    def predict_proba(self, X_context: np.ndarray) -> np.ndarray:
        if self.clf is None:
            raise RuntimeError("SequentialCRFDecoder must be fit before predict_proba")
        proba = self.clf.predict_proba(X_context)
        col_order = [list(self.clf.classes_).index(c) for c in self.class_names]
        return proba[:, col_order]


def viterbi_decode(log_proba: np.ndarray, transition_matrix: np.ndarray, class_names: list[str], start_prior: np.ndarray | None = None) -> list[str]:
    T, C = log_proba.shape
    log_trans = np.log(np.clip(transition_matrix, _EPS, 1.0))
    if start_prior is None:
        start_prior = np.full(C, 1.0 / C)
    log_start = np.log(np.clip(start_prior, _EPS, 1.0))

    dp = np.zeros((T, C))
    backpointer = np.zeros((T, C), dtype=int)
    dp[0] = log_start + log_proba[0]

    for t in range(1, T):
        scores = dp[t - 1][:, None] + log_trans  # (C_prev, C_cur)
        backpointer[t] = scores.argmax(axis=0)
        dp[t] = scores.max(axis=0) + log_proba[t]

    path_idx = np.zeros(T, dtype=int)
    path_idx[-1] = dp[-1].argmax()
    for t in range(T - 2, -1, -1):
        path_idx[t] = backpointer[t + 1, path_idx[t + 1]]
    return [class_names[i] for i in path_idx]


# ==========================================================================
# ResNet-TS 
# ==========================================================================
class SEBlock1D(nn.Module):

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc = nn.Sequential(nn.Linear(channels, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, channels), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = x.mean(dim=-1)
        scale = self.fc(pooled).unsqueeze(-1)
        return x * scale


class ResidualSEBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=8, padding="same")
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=5, padding="same")
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding="same")
        self.bn3 = nn.BatchNorm1d(out_channels)
        self.se = SEBlock1D(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = (
            nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=1), nn.BatchNorm1d(out_channels))
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.se(out)
        return self.relu(out + residual)


class ResNetTS(nn.Module):
    def __init__(self, in_channels: int, n_classes: int = 2, base_filters: int = 64):
        super().__init__()
        filters = [base_filters, base_filters * 2, base_filters * 2]
        blocks, prev = [], in_channels
        for f in filters:
            blocks.append(ResidualSEBlock1D(prev, f))
            prev = f
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Linear(prev, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.blocks(x)
        pooled = out.mean(dim=-1)
        return self.head(pooled)
