import json, joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

meta = json.load(open("models/meta.json", encoding="utf-8"))
scaler = joblib.load("models/scaler.pkl")
models = {n: joblib.load(f"models/{n}.pkl") for n in ["LogReg", "RandomForest", "XGBoost"]}

def prepare(data: dict) -> pd.DataFrame:
    row = pd.DataFrame([data])[meta["raw_columns"]]
    row[meta["num_cols"]] = row[meta["num_cols"]].apply(pd.to_numeric)
    row = pd.get_dummies(row)
    row = row.reindex(columns=meta["columns"], fill_value=0).astype(float)
    row[meta["num_cols"]] = scaler.transform(row[meta["num_cols"]])
    return row

def risk_level(p, t):
    return "High" if p >= t else ("Medium" if p >= t / 2 else "Low")

def predict(data: dict) -> dict:
    x = prepare(data)
    out = {}
    for name, model in models.items():
        p = float(model.predict_proba(x)[0, 1])
        t = meta["thresholds"][name]
        out[name] = {
            "probability": round(p, 4),
            "predicted_attrition": "Yes" if p >= t else "No",
            "risk_level": risk_level(p, t),
        }
    return out

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def home():
    return {"status": "ok"}

@app.get("/fields")
def fields():
    return {"raw_columns": meta["raw_columns"], "categories": meta["categories"]}

@app.post("/predict")
def predict_endpoint(data: dict):
    return predict(data)
