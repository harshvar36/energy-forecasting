
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import yaml, os
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

import lightgbm as lgb

from src.features.build_features import load_processed, build_features

def mape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-6, None))) * 100

def wape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100

def seasonal_naive(y, horizon, period):
    
    pred = pd.Series(y).shift(period).shift(-horizon + period)
    return np.array(pred)

def load_cfg():
    with open("configs/config.yaml", "r") as f:
        return yaml.safe_load(f)

def run_backtest(horizon=24, n_splits=5):
    print(f"\n=== Backtest: LightGBM direct model | horizon={horizon}h ===")

    df = load_processed()
    X, y = build_features(df, horizon=horizon)

    # rolling/expanding split
    tscv = TimeSeriesSplit(n_splits=n_splits)
    oof = np.zeros(len(y))
    importances = []

    for fold, (tr, va) in enumerate(tscv.split(X), 1):
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        y_tr, y_va = y.iloc[tr], y.iloc[va]

        # --- Load tuned params if available ---
    param_path = Path(f"configs/lgbm_params_h{horizon}.yaml")
    if param_path.exists():
        with open(param_path, "r") as f:
            tuned = yaml.safe_load(f)
    else:
        tuned = {
            "n_estimators": 2500,
            "learning_rate": 0.03,
            "num_leaves": 64,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "random_state": 42,
            "n_jobs": -1,
        }

        model = lgb.LGBMRegressor(**tuned)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)],
        )
    
        pred = model.predict(X_va)
        oof[va] = pred
        importances.append(pd.Series(model.feature_importances_, index=X.columns))

        fold_rmse = mean_squared_error(y_va, pred, squared=False)
        fold_mape = mape(y_va, pred)
        print(f"  Fold {fold}: RMSE={fold_rmse:.2f} | MAPE={fold_mape:.2f}% "
              f"| best_iter={model.best_iteration_}")

    rmse = mean_squared_error(y, oof, squared=False)
    mape_val = mape(y, oof)
    wape_val = wape(y, oof)
    print(f"\nLightGBM OOF → RMSE={rmse:.2f} | MAPE={mape_val:.2f}% | WAPE={wape_val:.2f}%")

   
    period = 24 if horizon == 24 else 168
    base_pred = seasonal_naive(y, horizon=horizon, period=period)
    # align lengths (drop NaNs)
    valid = ~np.isnan(base_pred)
    base_rmse = mean_squared_error(y[valid], base_pred[valid], squared=False)
    base_mape = mape(y[valid], base_pred[valid])
    base_wape = wape(y[valid], base_pred[valid])
    print(f"Baseline (seasonal {period}h) → RMSE={base_rmse:.2f} | MAPE={base_mape:.2f}% | WAPE={base_wape:.2f}%")

    # Save reports
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({
        "y_true": y,
        "y_pred_lgbm": oof,
        "y_pred_baseline": base_pred
    }).to_csv(out_dir / f"oof_h{horizon}.csv", index=False)

    fi = pd.concat(importances, axis=1).mean(1).sort_values(ascending=False)
    fi.to_csv(out_dir / f"feature_importance_h{horizon}.csv", header=["importance"])

    print(f"Saved OOF predictions and feature importance to {out_dir.resolve()}")

if __name__ == "__main__":
    cfg = load_cfg()
    horizons = cfg.get("forecast_horizons", [24, 168])
    for h in horizons:
        run_backtest(horizon=int(h), n_splits=5)
