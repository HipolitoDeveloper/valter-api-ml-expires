import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier

def build_classifier() -> Pipeline:
    clf = Pipeline(steps=[
        ("scaler", StandardScaler(with_mean=False)),
        ("logreg", LogisticRegression(max_iter=500, class_weight="balanced"))
    ])
    return clf

def build_baseline(constant: int) -> DummyClassifier:
    return DummyClassifier(strategy="constant", constant=int(constant))


def predict_proba_positive(model, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] == 2:
            return proba[:, 1]
        if proba.ndim == 2 and proba.shape[1] == 1:
            only_class = None
            if hasattr(model, "classes_") and len(getattr(model, "classes_")) == 1:
                only_class = int(model.classes_[0])
            if only_class is not None:
                return np.full((proba.shape[0],), 1.0 if only_class == 1 else 0.0, dtype=float)
            return np.zeros((proba.shape[0],), dtype=float)

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-scores))

    preds = model.predict(X)
    return preds.astype(float)
