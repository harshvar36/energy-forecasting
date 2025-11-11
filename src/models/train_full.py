
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import yaml, joblib
import pandas as pd
import lightgbm as lgb

from src.features.build_features import load_processed, build_features

MODELDIR = Path("models"); MODELDIR.mkdir(parents=True, exist_ok=True)

def load_tuned(h):
    p = Path(f"configs/lgbm_params_h{h}.yaml")
    if p.exists():
        with open(p, "r") as f:
            params = yaml.safe_load(f)
    else:
        params = {
            "n_estimators": 2500, "learning_rate": 0.03, "num_leaves": 64,
            "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0,
            "random_state": 42, "n_jobs": -1
        }
    # add required fixed fields if missing
    params.update({"objective": "regression", "verbosity": -1})
    return params

def train_and_save(horizon):
    df = load_processed()
    X, y = build_features(df, horizon=horizon)

    params = load_tuned(horizon)
    model = lgb.LGBMRegressor(**params)
    model.fit(X, y)

    # Save model + feature columns
    artifact = {
        "horizon": horizon,
        "features": list(X.columns),
        "model": model
    }
    out = MODELDIR / f"lgbm_h{horizon}.joblib"
    joblib.dump(artifact, out)
    print(f"Saved model to {out.resolve()} (features: {len(artifact['features'])})")

if __name__ == "__main__":
    for h in (24, 168):
        train_and_save(h)
