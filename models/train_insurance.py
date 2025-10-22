from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import joblib

RANDOM_STATE = 42

def load_insurance_csv() -> pd.DataFrame:
    path = os.path.join(os.path.dirname(__file__), "..", "data", "insurance.csv")
    if not os.path.exists(path):
        raise FileNotFoundError("No se encuentra data/insurance.csv. Usa scripts/download_data.py o agrega el archivo manualmente.")
    return pd.read_csv(path)

def main():
    df = load_insurance_csv()
    y = df["charges"].astype(float)
    X = df.drop(columns=["charges"])

    num_cols = ["age","bmi","children"]
    cat_cols = ["sex","smoker","region"]

    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols)
    ])

    # Ridge + poly optional (degree 1-2) via RandomizedSearch
    pipe = Pipeline([
        ("pre", pre),
        ("poly", PolynomialFeatures(include_bias=False)),
        ("reg", Ridge(random_state=RANDOM_STATE))
    ])

    param_dist = {
        "poly__degree": [1,2],
        "reg__alpha": np.logspace(-3, 3, 20)
    }

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(pipe, param_distributions=param_dist, n_iter=30, scoring="neg_root_mean_squared_error", cv=cv, n_jobs=-1, random_state=RANDOM_STATE, verbose=0)
    search.fit(X_train, y_train)

    best = search.best_estimator_
    y_pred = best.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ad = os.path.join(root, "artifacts"); os.makedirs(ad, exist_ok=True)
    joblib.dump(best, os.path.join(ad, "insurance_pipeline.joblib"))

    os.makedirs(os.path.join(root, "reports"), exist_ok=True)
    with open(os.path.join(root, "reports", "insurance_metrics.txt"), "w") as f:
        f.write(f"RMSE_test: {rmse:.2f}\nR2_test: {r2:.4f}\n")

if __name__ == "__main__":
    main()
