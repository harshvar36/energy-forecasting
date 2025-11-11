import pandas as pd
import requests
from datetime import datetime
import yaml
from pathlib import Path
from src.utils.io import DATA_RAW, ensure_parents

def load_config():
    with open("configs/config.yaml", "r") as f:
        return yaml.safe_load(f)

def infer_date_range_from_load(load_path: Path):
    df = pd.read_csv(load_path, parse_dates=[0])
    df.columns = [c.lower().strip() for c in df.columns]
    
    time_col = "datetime" if "datetime" in df.columns else df.columns[0]
    start = df[time_col].min().date().isoformat()
    end = df[time_col].max().date().isoformat()
    return start, end

def fetch_open_meteo(lat, lon, start, end) -> pd.DataFrame:
    url = (
        "https://archive-api.open-meteo.com/v1/era5?"
        f"latitude={lat}&longitude={lon}"
        f"&start_date={start}&end_date={end}"
        "&hourly=temperature_2m,relative_humidity_2m,dew_point_2m,"
        "apparent_temperature,precipitation,weathercode,wind_speed_10m,wind_direction_10m,cloudcover"
        "&timezone=auto"
    )
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    js = r.json()
    hourly = js["hourly"]
    df = pd.DataFrame(hourly)
    df.rename(columns={"time": "timestamp"}, inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

def main():
    cfg = load_config()
    zone_file = {"AEP":"AEP_hourly.csv"}.get(cfg.get("region","AEP"), "AEP_hourly.csv")
    load_path = DATA_RAW / zone_file

    if not load_path.exists():
        raise FileNotFoundError(f"Load file not found at {load_path}. Put your chosen *_hourly.csv there.")

    start, end = infer_date_range_from_load(load_path)

    lat = cfg.get("latitude")
    lon = cfg.get("longitude")

    print(f"Fetching weather for {start} → {end} at {lat},{lon} ...")
    dfw = fetch_open_meteo(lat, lon, start, end)
    out = DATA_RAW / "weather_hourly.csv"
    ensure_parents(out)
    dfw.to_csv(out, index=False)
    print(f"Saved weather to {out.resolve()}")

if __name__ == "__main__":
    main()
