from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import numpy as np
import pandas as pd
import os
from models.common import (
    ensure_artifacts, load_diabetes_pipeline, load_insurance_pipeline,
    current_threshold, set_threshold, feature_names_diabetes, feature_names_insurance
)

APP_DIR = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(APP_DIR, ".."))

app = FastAPI(title="Medical Insurance & Diabetes ML – FastAPI")
app.mount("/static", StaticFiles(directory=os.path.join(APP_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(APP_DIR, "templates"))

def render(request: Request, **ctx):
    base = {
        "request": request,
        "threshold": current_threshold(ROOT),
        "diabetes_features": feature_names_diabetes(),
        "insurance_features": feature_names_insurance(),
    }
    base.update(ctx)
    return templates.TemplateResponse("index.html", base)

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    try:
        ensure_artifacts(ROOT)
        return render(request)
    except Exception as e:
        return render(request, error=str(e))

@app.post("/predict/diabetes", response_class=HTMLResponse)
def predict_diabetes(request: Request,
                     Pregnancies: float = Form(...),
                     Glucose: float = Form(...),
                     BloodPressure: float = Form(...),
                     SkinThickness: float = Form(...),
                     Insulin: float = Form(...),
                     BMI: float = Form(...),
                     DiabetesPedigreeFunction: float = Form(...),
                     Age: float = Form(...),
                     threshold: float = Form(...)):
    try:
        ensure_artifacts(ROOT)
        pipe = load_diabetes_pipeline(ROOT)
        X = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]], dtype=float)
        proba = float(pipe.predict_proba(X)[0,1])
        y = int(proba >= float(threshold))
        set_threshold(ROOT, float(threshold))
        return render(request, diabetes_result={"proba": proba, "class": y})
    except Exception as e:
        return render(request, error=f"Diabetes: {e}")



@app.post("/predict/insurance", response_class=HTMLResponse)
def predict_insurance(request: Request,
                      age: float = Form(...),
                      sex: str = Form(...),
                      bmi: float = Form(...),
                      children: float = Form(...),
                      smoker: str = Form(...),
                      region: str = Form(...)):
    try:
        ensure_artifacts(ROOT)
        pipe = load_insurance_pipeline(ROOT)
        
        # 👇 construimos un DataFrame con nombres de columnas
        X = pd.DataFrame([{
            "age": age,
            "sex": sex,
            "bmi": bmi,
            "children": children,
            "smoker": smoker,
            "region": region
        }])
        
        yhat = float(pipe.predict(X)[0])
        return render(request, insurance_result={"charges": yhat})
    except Exception as e:
        return render(request, error=f"Insurance: {e}")
