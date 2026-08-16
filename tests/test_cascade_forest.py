import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from hemc.models import CascadeForestClassifier


def test_cascade_forest_fits_separable_data():
    X, y_idx = make_classification(
        n_samples=800,
        n_features=12,
        n_informative=8,
        n_redundant=0,
        n_classes=4,
        n_clusters_per_class=1,
        class_sep=2.5,
        random_state=0,
    )
    class_names = np.array(["F", "S", "P", "B"])
    y = class_names[y_idx]

    X_train, X_rest, y_train, y_rest = train_test_split(X, y, test_size=0.4, random_state=0, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_rest, y_rest, test_size=0.5, random_state=0, stratify=y_rest)

    clf = CascadeForestClassifier(n_estimators_per_forest=10, max_layers=3, cv_folds=3, random_state=0, verbose=False)
    clf.fit(X_train, y_train, X_val, y_val)

    assert clf.n_layers_ >= 1
    assert len(clf.score_history_) >= clf.n_layers_

    y_pred = clf.predict(X_test)
    accuracy = (y_pred == y_test).mean()
    assert accuracy > 0.85


def test_cascade_forest_predict_proba_sums_to_one():
    X, y_idx = make_classification(
        n_samples=300, n_features=8, n_informative=6, n_classes=3, n_clusters_per_class=1, random_state=1
    )
    class_names = np.array(["A", "B", "C"])
    y = class_names[y_idx]
    X_train, X_val = X[:200], X[200:]
    y_train, y_val = y[:200], y[200:]

    clf = CascadeForestClassifier(n_estimators_per_forest=10, max_layers=2, cv_folds=3, random_state=0, verbose=False)
    clf.fit(X_train, y_train, X_val, y_val)
    proba = clf.predict_proba(X_val)
    assert proba.shape == (len(X_val), 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
