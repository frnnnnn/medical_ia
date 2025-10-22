
import os, joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, Ridge

def train_synthetic(root: str):
    # Diabetes synthetic
    rng = np.random.default_rng(42)
    n = 400
    Xd = pd.DataFrame({
        "Pregnancies": rng.integers(0, 10, n),
        "Glucose": rng.normal(120, 30, n).clip(50, 220),
        "BloodPressure": rng.normal(70, 12, n).clip(30, 120),
        "SkinThickness": rng.normal(20, 8, n).clip(0, 60),
        "Insulin": rng.normal(100, 50, n).clip(0, 400),
        "BMI": rng.normal(30, 6, n).clip(15, 55),
        "DiabetesPedigreeFunction": rng.uniform(0.05, 2.5, n),
        "Age": rng.integers(21, 80, n),
    })
    logit = (0.02*(Xd["Glucose"]-120) + 0.03*(Xd["BMI"]-30) + 0.01*(Xd["Age"]-50) + rng.normal(0,0.5,n))
    yd = (logit > 0.0).astype(int)

    pre_d = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    pipe_d = Pipeline([("pre", pre_d), ("clf", LogisticRegression(max_iter=200, class_weight="balanced"))])
    pipe_d.fit(Xd, yd)

    # Insurance synthetic
    ni = 500
    sex = rng.choice(["male","female"], size=ni)
    smoker = rng.choice(["yes","no"], size=ni, p=[0.2,0.8])
    region = rng.choice(["southwest","southeast","northwest","northeast"], size=ni)
    age = rng.integers(18, 65, size=ni)
    bmi = rng.normal(30, 6, size=ni).clip(18, 55)
    children = rng.integers(0, 4, size=ni)

    base = 1500 + 250*children + 300*(age/10) + 200*(bmi-25)
    smoke_add = np.where(smoker=="yes", 10000 + 200*(bmi-25), 0)
    yi = base + smoke_add + rng.normal(0, 1000, size=ni)

    Xi = pd.DataFrame({"age":age,"sex":sex,"bmi":bmi,"children":children,"smoker":smoker,"region":region})
    pre_i = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), ["age","bmi","children"]),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), ["sex","smoker","region"])
    ])
    pipe_i = Pipeline([("pre", pre_i), ("poly", PolynomialFeatures(include_bias=False, degree=1)), ("reg", Ridge(alpha=1.0))])
    pipe_i.fit(Xi, yi)

    artifacts = os.path.join(root, "artifacts")
    os.makedirs(artifacts, exist_ok=True)
    joblib.dump(pipe_d, os.path.join(artifacts, "diabetes_pipeline.joblib"))
    joblib.dump(pipe_i, os.path.join(artifacts, "insurance_pipeline.joblib"))
