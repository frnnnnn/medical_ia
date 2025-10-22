from __future__ import annotations
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
import joblib

RANDOM_STATE = 42

def rf_diabetes(df: pd.DataFrame, out_dir: str):
    _cols_map = {c.lower(): c for c in df.columns}
    if 'outcome' not in _cols_map:
        raise KeyError("No se encontró la columna 'Outcome' en diabetes.csv")
    y = df[_cols_map['outcome']].astype(int)
    X = df.drop(columns=[_cols_map['outcome']])
    num_cols = X.columns.tolist()

    pre = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])

    rf = Pipeline([
        ("pre", pre),
        ("rf", RandomForestClassifier(
            n_estimators=300, max_depth=None, random_state=RANDOM_STATE, class_weight="balanced"))
    ])
    rf.fit(X, y)

    imp = rf.named_steps["rf"].feature_importances_
    names = num_cols
    imp_df = pd.DataFrame({"feature": names, "importance": imp}).sort_values("importance", ascending=False)
    imp_df.to_csv(os.path.join(out_dir, "diabetes_rf_importances.csv"), index=False)

def rf_insurance(df: pd.DataFrame, out_dir: str):
    y = df[[c for c in df.columns if c.lower()=='charges'][0]].astype(float)
    X = df.drop(columns=[[c for c in df.columns if c.lower()=='charges'][0]])

    num_cols = ["age","bmi","children"]
    cat_cols = ["sex","smoker","region"]

    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols)
    ])
    rf = Pipeline([
        ("pre", pre),
        ("rf", RandomForestRegressor(n_estimators=400, random_state=RANDOM_STATE))
    ])
    rf.fit(X, y)

    # Get feature names after OHE
    oh: OneHotEncoder = rf.named_steps["pre"].named_transformers_["cat"].named_steps["oh"]
    cat_names = oh.get_feature_names_out(cat_cols).tolist()
    all_names = num_cols + cat_names

    imp = rf.named_steps["rf"].feature_importances_
    imp_df = pd.DataFrame({"feature": all_names, "importance": imp}).sort_values("importance", ascending=False)
    imp_df.to_csv(os.path.join(out_dir, "insurance_rf_importances.csv"), index=False)

def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_dir = os.path.join(root, "reports")
    os.makedirs(out_dir, exist_ok=True)

    dia = os.path.join(root, "data", "diabetes.csv")
    ins = os.path.join(root, "data", "insurance.csv")
    if not (os.path.exists(dia) and os.path.exists(ins)):
        print("Coloca los datasets en data/ o ejecuta scripts/download_data.py")
        return

    df_d = pd.read_csv(dia)
    df_i = pd.read_csv(ins)

    rf_diabetes(df_d, out_dir)
    rf_insurance(df_i, out_dir)

if __name__ == "__main__":
    main()
