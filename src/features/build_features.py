
import numpy as np
import pandas as pd
from pathlib import Path

DATA_PROCESSED = Path("data/processed")

def _fourier(t, period, K=3):
    x = []
    for k in range(1, K + 1):
        x.append(np.sin(2 * np.pi * k * t / period))
        x.append(np.cos(2 * np.pi * k * t / period))
    return np.vstack(x).T  

def load_processed() -> pd.DataFrame:
    df = pd.read_csv(DATA_PROCESSED / "dataset_merged.csv", parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

def build_features(df: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, pd.Series]:
    """
    Direct-forecasting setup: predict load at t+horizon using info up to t.
    We align the target by shifting backward: y = load_mw.shift(-horizon).
    """
    df = df.copy()

    # lags of target
    for lag in [1, 24, 48, 72, 168]:
        df[f"lag_{lag}"] = df["load_mw"].shift(lag)

    # rolling stats on target
    roll_specs = [(3, "3h"), (24, "24h"), (168, "168h")]
    for w, _name in roll_specs:
        r = df["load_mw"].rolling(w)
        df[f"roll_mean_{w}"] = r.mean()
        df[f"roll_std_{w}"] = r.std()
        df[f"roll_min_{w}"] = r.min()
        df[f"roll_max_{w}"] = r.max()

    # weather rolling means (helps denoise)
    weather_cols = [
        "temperature_2m","apparent_temperature","relative_humidity_2m",
        "dew_point_2m","wind_speed_10m","precipitation","cloudcover"
    ]
    for c in weather_cols:
        if c in df.columns:
            df[f"{c}_lag1"] = df[c].shift(1)
            df[f"{c}_mean24"] = df[c].rolling(24).mean()

    # calendar (already present if you used prepare_dataset)
   # calendar (ensure datetime)
    if "timestamp_local" not in df.columns:
        df["timestamp_local"] = df["timestamp"]

    # convert both to datetime if needed
    df["timestamp_local"] = pd.to_datetime(df["timestamp_local"], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")


    df["hour"] = df["timestamp_local"].dt.hour
    df["dow"] = df["timestamp_local"].dt.dayofweek
    df["month"] = df["timestamp_local"].dt.month
    df["is_weekend"] = (df["dow"] >= 5).astype(int)

    # Fourier seasonality: daily (24) and weekly (24*7)
    t = np.arange(len(df))
    fourier_day = _fourier(t, period=24, K=3)
    fourier_wk = _fourier(t, period=24 * 7, K=3)
    for i in range(fourier_day.shape[1]):
        df[f"fourier_d_{i}"] = fourier_day[:, i]
    for i in range(fourier_wk.shape[1]):
        df[f"fourier_w_{i}"] = fourier_wk[:, i]

    # target (direct)
    y = df["load_mw"].shift(-horizon)

    # assemble feature list
    feature_cols = [c for c in df.columns if c not in
                    ["timestamp","timestamp_local","load_mw"]]

    X = df[feature_cols]

   
    valid = ~(X.isna().any(axis=1) | y.isna())
    X = X.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True)

    return X, y

if __name__ == "__main__":
    dfp = load_processed()
    X, y = build_features(dfp, horizon=24)
    print(X.shape, y.shape)
