from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.utils.class_weight import compute_sample_weight
import joblib
from models.common import artifacts_dir, thresholds_path

RANDOM_STATE = 42

def load_diabetes_csv() -> pd.DataFrame:
    # Expect a local file in data/diabetes.csv
    path = os.path.join(os.path.dirname(__file__), "..", "data", "diabetes.csv")
    if not os.path.exists(path):
        raise FileNotFoundError("No se encuentra data/diabetes.csv. Usa scripts/download_data.py o agrega el archivo manualmente.")
    return pd.read_csv(path)

def fit_and_choose_threshold(X_train, y_train, X_val, y_val, pipe) -> float:
    # Busca umbral que maximiza J sobre validación
    proba_val = pipe.predict_proba(X_val)[:,1]
    fpr, tpr, thr = roc_curve(y_val, proba_val)
    J = tpr - fpr
    j_best = np.argmax(J)
    tau = float(thr[j_best])
    return tau

def main():
    df = load_diabetes_csv()
    # Robust 'Outcome' (case-insensitive)
    _cols_map = {c.lower(): c for c in df.columns}
    if 'outcome' not in _cols_map:
        raise KeyError("No se encontró la columna 'Outcome' en diabetes.csv")
    y = df[_cols_map['outcome']].astype(int)
    X = df.drop(columns=[_cols_map['outcome']])

    num_cols = X.columns.tolist()

    pre = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler())
    ])

    model = LogisticRegression(max_iter=200, class_weight="balanced", random_state=RANDOM_STATE)
    pipe = Pipeline([
        ("pre", pre),
        ("clf", model)
    ])

    param_dist = {
        "clf__C": np.logspace(-3, 3, 20),
        "clf__penalty": ["l2"],
        "clf__solver": ["liblinear","lbfgs"]
    }

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=RANDOM_STATE)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(pipe, param_distributions=param_dist, n_iter=30, scoring="roc_auc", cv=cv, n_jobs=-1, random_state=RANDOM_STATE, verbose=0)
    search.fit(X_train, y_train)

    best = search.best_estimator_
    auc_val = roc_auc_score(y_val, best.predict_proba(X_val)[:,1])
    tau = fit_and_choose_threshold(X_train, y_train, X_val, y_val, best)

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ad = artifacts_dir(root)
    joblib.dump(best, os.path.join(ad, "diabetes_pipeline.joblib"))

    # save threshold
    with open(thresholds_path(root), "w") as f:
        json.dump({"diabetes_threshold": float(tau)}, f)

    os.makedirs(os.path.join(root, "reports"), exist_ok=True)
    with open(os.path.join(root, "reports", "diabetes_metrics.txt"), "w") as f:
        f.write(f"AUC_val: {auc_val:.4f}\nOptimalThreshold(Youden): {tau:.4f}\n")

if __name__ == "__main__":
    main()
