
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import optuna, yaml
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

from src.features.build_features import load_processed, build_features

CONFIGS = Path("configs")
CONFIGS.mkdir(parents=True, exist_ok=True)

def mape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-6, None))) * 100

def objective(trial, X, y):
    params = {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "random_state": 42,
        "n_estimators": trial.suggest_int("n_estimators", 800, 3000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 256, log=True),
        "max_depth": trial.suggest_int("max_depth", -1, 16),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
    }

    tscv = TimeSeriesSplit(n_splits=5)
    oof = np.zeros(len(y))
    for tr, va in tscv.split(X):
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        y_tr, y_va = y.iloc[tr], y.iloc[va]

        model = lgb.LGBMRegressor(**params, n_jobs=-1)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)],
        )
        pred = model.predict(X_va)
        oof[va] = pred

    rmse = mean_squared_error(y, oof, squared=False)
    trial.set_user_attr("rmse", rmse)
    trial.set_user_attr("mape", mape(y, oof))
    return rmse  # minimize RMSE

def tune(horizon: int, n_trials: int = 30):
    df = load_processed()
    X, y = build_features(df, horizon=horizon)

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda t: objective(t, X, y), n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    best["random_state"] = 42
    best["n_jobs"] = -1
    best["objective"] = "regression"
    best["metric"] = "rmse"
    best["verbosity"] = -1

    out = CONFIGS / f"lgbm_params_h{horizon}.yaml"
    with open(out, "w") as f:
        yaml.safe_dump(best, f)
    print(f"Best RMSE: {study.best_value:.3f}")
    print(f"Saved best params to {out.resolve()}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--trials", type=int, default=30)
    args = ap.parse_args()
    tune(args.horizon, args.trials)
