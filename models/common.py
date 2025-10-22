import os, json, joblib
from typing import List

def artifacts_dir(root: str) -> str:
    d = os.path.join(root, "artifacts")
    os.makedirs(d, exist_ok=True)
    return d

def thresholds_path(root: str) -> str:
    return os.path.join(artifacts_dir(root), "thresholds.json")

def current_threshold(root: str) -> float:
    path = thresholds_path(root)
    if os.path.exists(path):
        return float(json.load(open(path)).get("diabetes_threshold", 0.5))
    return 0.5

def set_threshold(root: str, tau: float) -> None:
    path = thresholds_path(root)
    data = {"diabetes_threshold": float(tau)}
    with open(path, "w") as f:
        json.dump(data, f)

def ensure_artifacts(root: str):
    """Ensure pipelines exist. If Kaggle data is missing, train quick synthetic
    models so the app keeps working (documented in README)."""
    ad = artifacts_dir(root)
    # If models not present, try to train quickly (will rely on local data if present)
    di_path = os.path.join(ad, "diabetes_pipeline.joblib")
    in_path = os.path.join(ad, "insurance_pipeline.joblib")
    if not os.path.exists(di_path) or not os.path.exists(in_path):
        try:
            from .train_diabetes import main as train_diabetes
            from .train_insurance import main as train_insurance
            train_insurance()
            train_diabetes()
        except Exception as e:
            print("WARNING: primary training failed:", e)
            # synthetic fallback
            try:
                from .synthetic_fallback import train_synthetic
                train_synthetic(root)
            except Exception as e2:
                print("FATAL: synthetic fallback failed:", e2)

def load_diabetes_pipeline(root: str):
    return joblib.load(os.path.join(artifacts_dir(root), "diabetes_pipeline.joblib"))

def load_insurance_pipeline(root: str):
    return joblib.load(os.path.join(artifacts_dir(root), "insurance_pipeline.joblib"))

def feature_names_diabetes() -> List[str]:
    return ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]

def feature_names_insurance() -> List[str]:
    return ["age","sex","bmi","children","smoker","region"]
