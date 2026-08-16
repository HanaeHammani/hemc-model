"""Evaluation metrics: point-wise (per-sample) and event-wise (segmentation-aware)."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import groupby

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


# ==========================================================================
# Point-wise (per-sample) metrics
# ==========================================================================


def pointwise_report(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> pd.DataFrame:
    """Per-class precision/recall/F1 + macro row, as a tidy DataFrame."""
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=class_names, zero_division=0)
    df = pd.DataFrame({"precision": precision, "recall": recall, "f1": f1, "support": support}, index=class_names)
    macro = df[["precision", "recall", "f1"]].mean()
    df.loc["macro_avg"] = [*macro.to_numpy(), df["support"].sum()]
    return df


def pointwise_confusion(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    return pd.DataFrame(cm, index=[f"true_{c}" for c in class_names], columns=[f"pred_{c}" for c in class_names])


# ==========================================================================
# Event-wise (segmentation-aware) metrics
# ==========================================================================
# Point-wise classifiers tend to fragment physiologically continuous events
# into many short, implausible predicted segments ("temporal
# over-segmentation"). This converts label sequences into contiguous events
# and evaluates at that level: events are matched via greedy
# highest-IoU-first one-to-one matching (threshold 0.5), and an
# "oversegmentation ratio" (predicted events / true events per class)
# quantifies fragmentation independently of raw classification accuracy.


@dataclass(frozen=True)
class Event:
    label: str
    start: int  # inclusive
    end: int  # exclusive


def labels_to_events(labels: np.ndarray) -> list[Event]:
    """Group a per-sample label sequence into contiguous (label, start, end) events."""
    events, idx = [], 0
    for label, group in groupby(labels):
        length = sum(1 for _ in group)
        events.append(Event(label=str(label), start=idx, end=idx + length))
        idx += length
    return events


def _iou(a: Event, b: Event) -> float:
    overlap = max(0, min(a.end, b.end) - max(a.start, b.start))
    union = (a.end - a.start) + (b.end - b.start) - overlap
    return overlap / union if union > 0 else 0.0


def match_events(pred_events: list[Event], true_events: list[Event], class_name: str, iou_thr: float = 0.5):
    """Greedy highest-IoU-first one-to-one matching restricted to one class.

    Returns (n_matched, n_pred_of_class, n_true_of_class).
    """
    preds = [e for e in pred_events if e.label == class_name]
    trues = [e for e in true_events if e.label == class_name]
    pairs = [(_iou(p, t), i, j) for i, p in enumerate(preds) for j, t in enumerate(trues) if _iou(p, t) >= iou_thr]
    pairs.sort(key=lambda x: x[0], reverse=True)
    matched_pred: set[int] = set()
    matched_true: set[int] = set()
    n_matched = 0
    for _, i, j in pairs:
        if i in matched_pred or j in matched_true:
            continue
        matched_pred.add(i)
        matched_true.add(j)
        n_matched += 1
    return n_matched, len(preds), len(trues)


def event_wise_report(pred_labels: np.ndarray, true_labels: np.ndarray, class_names: list[str], iou_thr: float = 0.5) -> pd.DataFrame:
    """Per-class event precision/recall/F1 (IoU-matched) + oversegmentation ratio."""
    pred_events = labels_to_events(pred_labels)
    true_events = labels_to_events(true_labels)

    rows = []
    for class_name in class_names:
        n_matched, n_pred, n_true = match_events(pred_events, true_events, class_name, iou_thr)
        precision = n_matched / n_pred if n_pred > 0 else 0.0
        recall = n_matched / n_true if n_true > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        oversegmentation = n_pred / n_true if n_true > 0 else np.nan
        rows.append({
            "class": class_name, "n_true_events": n_true, "n_pred_events": n_pred,
            "event_precision": precision, "event_recall": recall, "event_f1": f1,
            "oversegmentation_ratio": oversegmentation,
        })
    return pd.DataFrame(rows).set_index("class")
