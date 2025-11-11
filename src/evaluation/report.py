
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT = Path("data/processed")
PLOTS = OUT / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

def mape(y, p): return np.mean(np.abs((y - p) / np.clip(np.abs(y), 1e-6, None))) * 100
def wape(y, p): return np.sum(np.abs(y - p)) / np.sum(np.abs(y)) * 100
def rmse(y, p): return np.sqrt(np.mean((y - p) ** 2))

def make_report(horizon=24):
    oof = pd.read_csv(OUT / f"oof_h{horizon}.csv")
    y = oof["y_true"].values
    p = oof["y_pred_lgbm"].values
    b = oof["y_pred_baseline"].values

    rows = []
    rows.append(["LightGBM", rmse(y, p), mape(y, p), wape(y, p)])
    valid = ~np.isnan(b)
    rows.append([f"Seasonal-{24 if horizon==24 else 168}h", rmse(y[valid], b[valid]), mape(y[valid], b[valid]), wape(y[valid], b[valid])])
    dfm = pd.DataFrame(rows, columns=["model", "rmse", "mape", "wape"])
    dfm.to_csv(OUT / f"metrics_h{horizon}.csv", index=False)

    # Plot actual vs preds
    plt.figure(figsize=(14,4))
    plt.plot(y, label="actual")
    plt.plot(p, label="lgbm", alpha=0.8)
    plt.plot(b, label="baseline", alpha=0.6)
    plt.title(f"OOF predictions (h={horizon}h)")
    plt.xlabel("time index"); plt.ylabel("load (MW)"); plt.legend(); plt.tight_layout()
    plt.savefig(PLOTS / f"oof_h{horizon}.png", dpi=130); plt.close()

    # Error by hour-of-day heatmap (needs timestamps → approximate index bucketing)
    err = pd.DataFrame({"pred": p, "true": y})
    err["ae"] = (err["true"] - err["pred"]).abs()
    # keep simple: rolling-mean error curve
    plt.figure(figsize=(14,3))
    err["ae"].rolling(168).mean().plot()
    plt.title("Rolling 7-day Mean Absolute Error")
    plt.tight_layout()
    plt.savefig(PLOTS / f"rolling_mae_h{horizon}.png", dpi=130); plt.close()

    
    try:
        fi = pd.read_csv(OUT / f"feature_importance_h{horizon}.csv")
        fi = fi.sort_values("importance", ascending=False).head(20)
        plt.figure(figsize=(8,6))
        plt.barh(fi.iloc[::-1,0], fi.iloc[::-1,1])
        plt.title(f"Top-20 Feature Importance (h={horizon})")
        plt.tight_layout()
        plt.savefig(PLOTS / f"feature_importance_h{horizon}.png", dpi=130); plt.close()
    except Exception as e:
        print("FI plot skipped:", e)

    print(f"Saved metrics to {OUT}\\metrics_h{horizon}.csv and plots to {PLOTS}")

if __name__ == "__main__":
    make_report(24)
