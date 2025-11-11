
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
from datetime import timedelta

from src.features.build_features import load_processed, build_features

OUT = Path("data/processed/plots")
OUT.mkdir(parents=True, exist_ok=True)

def train_for_shap(horizon=24, holdout_days=30):
    df = load_processed()
    
    X, y = build_features(df, horizon=horizon)

    
    rows_per_day = 24
    val_len = holdout_days * rows_per_day
    split = len(X) - val_len if len(X) > val_len + 1000 else int(len(X)*0.85)

    X_tr, X_va = X.iloc[:split], X.iloc[split:]
    y_tr, y_va = y.iloc[:split], y.iloc[split:]

    model = lgb.LGBMRegressor(
        n_estimators=2000, learning_rate=0.03,
        num_leaves=64, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, random_state=42, n_jobs=-1
    )
    model.fit(X_tr, y_tr,
              eval_set=[(X_va, y_va)],
              eval_metric="rmse",
              callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)])
    return model, X_va, y_va

def main():
    model, X_va, y_va = train_for_shap(horizon=24, holdout_days=30)

    # Use a small sample for speed
    sample = X_va.sample(min(2000, len(X_va)), random_state=42)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    # Summary (beeswarm)
    shap.summary_plot(shap_values, sample, show=False)
    import matplotlib.pyplot as plt
    plt.tight_layout()
    plt.savefig(OUT / "shap_summary_h24.png", dpi=130); plt.close()

    # Bar summary
    shap.summary_plot(shap_values, sample, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(OUT / "shap_bar_h24.png", dpi=130); plt.close()

    # Dependence plot for temperature if available
    for cand in ["temperature_2m_mean24","temperature_2m","apparent_temperature","fourier_d_0"]:
        if cand in sample.columns:
            shap.dependence_plot(cand, shap_values, sample, show=False)
            plt.tight_layout()
            plt.savefig(OUT / f"shap_depend_{cand}_h24.png", dpi=130); plt.close()
            break

    print(f"Saved SHAP plots in {OUT}")

if __name__ == "__main__":
    main()
