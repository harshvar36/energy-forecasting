import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from src.utils.io import DATA_RAW, DATA_PROCESSED, ensure_parents

def load_config():
    with open("configs/config.yaml", "r") as f:
        return yaml.safe_load(f)

def read_load(region: str) -> pd.DataFrame:
    # Map region to filename; change if you used another file
    file_map = {
        "AEP": "AEP_hourly.csv",
        # add more if you switch zones
    }
    f = DATA_RAW / file_map.get(region, "AEP_hourly.csv")
    if not f.exists():
        raise FileNotFoundError(f"Load file not found at {f}")
    df = pd.read_csv(f, parse_dates=[0])
    df.columns = [c.lower().strip() for c in df.columns]
    time_col = "datetime" if "datetime" in df.columns else df.columns[0]
    load_col_candidates = [c for c in df.columns if c.endswith("_mw")]
    if not load_col_candidates:
        # fallback assume second column is load
        load_col = df.columns[1]
    else:
        load_col = load_col_candidates[0]

    df = df[[time_col, load_col]].rename(columns={time_col: "timestamp", load_col: "load_mw"})
    df = df.sort_values("timestamp").dropna()
    
    df = df.set_index("timestamp").resample("H").mean().interpolate(limit=2).reset_index()
    return df

def read_weather() -> pd.DataFrame:
    f = DATA_RAW / "weather_hourly.csv"
    if not f.exists():
        raise FileNotFoundError(f"Weather file not found at {f}. Run fetch_weather.py first.")
    df = pd.read_csv(f, parse_dates=["timestamp"])
    return df

def enrich_calendar(df: pd.DataFrame, tz: str) -> pd.DataFrame:
    df["timestamp_local"] = df["timestamp"].dt.tz_localize("UTC").dt.tz_convert(tz).dt.tz_localize(None)
    df["hour"] = df["timestamp_local"].dt.hour
    df["dow"] = df["timestamp_local"].dt.dayofweek
    df["month"] = df["timestamp_local"].dt.month
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    return df

def main():
    cfg = load_config()
    tz = cfg.get("timezone", "US/Eastern")

    load_df = read_load(cfg.get("region","AEP"))
    weather_df = read_weather()

   
    merged = pd.merge_asof(
        load_df.sort_values("timestamp"),
        weather_df.sort_values("timestamp"),
        on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta("30min")
    )

    # Basic cleaning
    merged = merged.dropna(subset=["load_mw"]).copy()
    # Clip impossible weather values if any
    for col in ["temperature_2m","apparent_temperature","relative_humidity_2m","dew_point_2m","wind_speed_10m","precipitation","cloudcover"]:
        if col in merged.columns:
            merged[col] = merged[col].astype(float)

    # Calendar features
    merged = enrich_calendar(merged, tz)

    # Save
    out_csv = DATA_PROCESSED / "dataset_merged.csv"
    out_parq = DATA_PROCESSED / "dataset_merged.parquet"
    ensure_parents(out_csv)
    merged.to_csv(out_csv, index=False)
    try:
        merged.to_parquet(out_parq, index=False)
    except Exception:
        pass

    print(f"Rows: {len(merged):,}")
    print(f"Saved merged dataset to:\n - {out_csv.resolve()}\n - {out_parq.resolve()} (if supported)")

if __name__ == "__main__":
    main()
