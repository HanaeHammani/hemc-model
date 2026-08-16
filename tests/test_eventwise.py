import numpy as np

from hemc.eval import Event, event_wise_report, labels_to_events, match_events


def test_labels_to_events_basic():
    labels = np.array(list("AAABBBBCCA"))
    events = labels_to_events(labels)
    assert events == [
        Event(label="A", start=0, end=3),
        Event(label="B", start=3, end=7),
        Event(label="C", start=7, end=9),
        Event(label="A", start=9, end=10),
    ]


def test_match_events_perfect_match():
    pred = [Event("F", 0, 10), Event("S", 10, 15)]
    true = [Event("F", 0, 10), Event("S", 10, 15)]
    n_matched, n_pred, n_true = match_events(pred, true, "F")
    assert (n_matched, n_pred, n_true) == (1, 1, 1)


def test_oversegmentation_ratio_detects_fragmentation():
    # true: one 30-sample fixation event
    true_labels = np.array(["F"] * 30)
    # pred: the same span fragmented into 3 short fixation events separated by
    # single-sample saccade blips (classic over-segmentation artifact)
    pred_labels = np.array(
        ["F"] * 9 + ["S"] * 1 + ["F"] * 9 + ["S"] * 1 + ["F"] * 10
    )
    report = event_wise_report(pred_labels, true_labels, class_names=["F", "S"], iou_thr=0.5)
    assert report.loc["F", "n_true_events"] == 1
    assert report.loc["F", "n_pred_events"] == 3
    assert report.loc["F", "oversegmentation_ratio"] == 3.0
    # the largest fragment (10 samples) still has IoU 10/30 = 0.33 < 0.5 threshold
    # against the single 30-sample true event, so event-level recall should be 0
    assert report.loc["F", "event_recall"] == 0.0


def test_event_wise_report_perfect_prediction():
    labels = np.array(list("FFFFSSSPPPPBB"))
    report = event_wise_report(labels, labels, class_names=["F", "S", "P", "B"], iou_thr=0.5)
    assert (report["oversegmentation_ratio"] == 1.0).all()
    assert (report["event_f1"] == 1.0).all()
