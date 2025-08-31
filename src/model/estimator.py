import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def build_classifier() -> Pipeline:
    clf = Pipeline(steps=[
        ("scaler", StandardScaler(with_mean=False)),
        ("logreg", LogisticRegression(max_iter=500, class_weight="balanced"))
    ])
    return clf

def predict_proba(model, X) -> np.ndarray:
    return model.predict_proba(X)[:, 1]
