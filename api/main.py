# api/main.py
import uvicorn
from fastapi import FastAPI, Query
from pydantic import BaseModel
from pathlib import Path
import joblib
from src.features.build_features import load_processed, build_features

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Energy Forecast API")

# allow calls from Streamlit or any origin during demo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # change to your Streamlit origin for production
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class ForecastResponse(BaseModel):
    horizon: int
    forecast_mw: float

def load_artifact(h: int):
    p = Path(f"models/lgbm_h{h}.joblib")
    if not p.exists():
        raise FileNotFoundError(f"Model not found at {p}. Train it first.")
    return joblib.load(p)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/forecast", response_model=ForecastResponse)
def forecast(horizon: int = Query(24, enum=[24, 168])):
    art = load_artifact(horizon)
    model = art["model"]
    feat_names = art["features"]

    df = load_processed()
    X, y = build_features(df, horizon=horizon)
    x_last = X.iloc[[-1]].copy()

    for m in feat_names:
        if m not in x_last.columns:
            x_last[m] = 0.0
    x_last = x_last[feat_names]

    pred = float(model.predict(x_last)[0])
    return ForecastResponse(horizon=horizon, forecast_mw=pred)

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
