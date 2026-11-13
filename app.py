import json      
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

meta = json.load(open("models/meta.json", encoding="utf-8"))
scaler = joblib.load("models/scaler.pkl")
models = {n: joblib.load(f"models/{n}.pkl") for n in ["LogReg", "RandomForest", "XGBoost"]}

# الداتا الأصلية: للقيم الافتراضية ولحدود الخطورة
raw_df = pd.read_csv("models/WA_Fn-UseC_-HR-Employee-Attrition.csv", encoding="utf-8-sig")
raw_df = raw_df[meta["raw_columns"]]

# القيمة الافتراضية لكل حقل: الوسيط للأرقام والأكتر تكرارا للنصوص
defaults = {}
for c in meta["raw_columns"]:
    col = raw_df[c]
    if pd.api.types.is_numeric_dtype(col):
        defaults[c] = float(col.median())
    else:
        defaults[c] = col.mode()[0]


def prepare(data) -> pd.DataFrame:
    if isinstance(data, dict):
        given = {k: v for k, v in data.items() if v is not None and v != ""}
        data = pd.DataFrame([{**defaults, **given}])
    row = data[meta["raw_columns"]].copy()
    row[meta["num_cols"]] = row[meta["num_cols"]].apply(pd.to_numeric)
    row = pd.get_dummies(row)
    row = row.reindex(columns=meta["columns"], fill_value=0).astype(float)
    row[meta["num_cols"]] = scaler.transform(row[meta["num_cols"]])
    return row


# حدود الخطورة بالمئين لكل موديل على الـ 1470 موظف
all_x = prepare(raw_df)
cuts = {}
for name, model in models.items():
    probs = model.predict_proba(all_x)[:, 1]
    cuts[name] = {
        "medium": float(np.percentile(probs, 60)),
        "high": float(np.percentile(probs, 84)),
    }
del all_x


def risk_level(p, name):
    c = cuts[name]
    if p >= c["high"]:
        return "High"
    if p >= c["medium"]:
        return "Medium"
    return "Low"


def predict(data: dict) -> dict:
    x = prepare(data)
    out = {}
    for name, model in models.items():
        p = float(model.predict_proba(x)[0, 1])
        t = meta["thresholds"][name]
        out[name] = {
            "probability": round(p, 4),
            "predicted_attrition": "Yes" if p >= t else "No",
            "risk_level": risk_level(p, name),
        }
    return out


app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/")
def home():
    return {"status": "ok"}


@app.get("/fields")
def fields():
    return {"raw_columns": meta["raw_columns"], "categories": meta["categories"]}


@app.post("/predict")
def predict_endpoint(data: dict):
    return predict(data)
