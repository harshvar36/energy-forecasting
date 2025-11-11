
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import argparse, joblib, pandas as pd, numpy as np
from src.features.build_features import load_processed, build_features

MODELDIR = Path("models")

def forecast(horizon: int):
    art_path = MODELDIR / f"lgbm_h{horizon}.joblib"
    if not art_path.exists():
        raise FileNotFoundError(f"Model not found at {art_path}. Train it first.")

    artifact = joblib.load(art_path)
    model = artifact["model"]
    feat_names = artifact["features"]

    df = load_processed()
    # Build features and keep the **last available row** for inference (time t)
    Xall, yall = build_features(df, horizon=horizon)
    x_last = Xall.iloc[[-1]].copy()

    # Ensure column alignment
    missing = [c for c in feat_names if c not in x_last.columns]
    if missing:
        for m in missing: x_last[m] = 0.0
    x_last = x_last[feat_names]

    pred = float(model.predict(x_last)[0])

    # the corresponding target index is t + horizon (we don't compute actual timestamp here)
    print(f"Forecast (t+{horizon}h): {pred:.3f} MW")
    return pred

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=24, choices=[24,168])
    args = ap.parse_args()
    forecast(args.horizon)
