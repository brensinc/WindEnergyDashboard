import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import inspect
from timezonefinder import TimezoneFinder
from zoneinfo import ZoneInfo
import gridstatus




# -----------------------------
# Open-Meteo fetchers
# -----------------------------
def fetch_open_meteo_archive(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    local_tz = get_tz_from_latlon(lat, lon)

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ["wind_speed_100m", "wind_direction_100m"],
        "wind_speed_unit": "ms",
        "timezone": str(local_tz.key),
        "start_date": start_date,
        "end_date": end_date,
    }
    r = requests.get(ARCHIVE_URL, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    hourly = data.get("hourly", {})
    time = hourly.get("time", [])
    ws100 = hourly.get("wind_speed_100m", [])
    wd100 = hourly.get("wind_direction_100m", [])

    if not time or not ws100:
        return pd.DataFrame(columns=["timestamp", "wind_speed_mps", "wind_direction_deg", "source", "hub_height_m"])

    ts = _localize_safely(time, local_tz)

    return pd.DataFrame({
        "timestamp": ts,
        "wind_speed_mps": ws100,
        "wind_direction_deg": wd100 if wd100 else [np.nan] * len(ws100),
        "source": "archive",
        "hub_height_m": REFERENCE_HUB_HEIGHT_M,
    })

def fetch_open_meteo_forecast(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    local_tz = get_tz_from_latlon(lat, lon)

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ["wind_speed_80m", "wind_speed_120m", "wind_direction_80m"],
        "wind_speed_unit": "ms",
        "timezone": str(local_tz.key),
        "start_date": start_date,
        "end_date": end_date,
    }
    r = requests.get(FORECAST_URL, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    hourly = data.get("hourly", {})
    time = hourly.get("time", [])
    ws80 = hourly.get("wind_speed_80m", [])
    ws120 = hourly.get("wind_speed_120m", [])
    wd80 = hourly.get("wind_direction_80m", [])

    if not time or (not ws80 and not ws120):
        raise ValueError("No hourly forecast wind data returned for this query.")

    ts = _localize_safely(time, local_tz)

    df = pd.DataFrame({
        "timestamp": ts,
        "wind_speed_80m_mps": ws80 if ws80 else [np.nan] * len(time),
        "wind_speed_120m_mps": ws120 if ws120 else [np.nan] * len(time),
        "wind_direction_deg": wd80 if wd80 else [np.nan] * len(time),
        "source": "forecast_api",
    })
    df["wind_speed_mps"] = _wind_100m_from_80_120(df)
    df["hub_height_m"] = REFERENCE_HUB_HEIGHT_M
    return df[["timestamp", "wind_speed_mps", "wind_direction_deg", "source", "hub_height_m"]]

def get_wind_data(lat: float, lon: float, date_start: str, date_end: str) -> pd.DataFrame:
    """
    Stitch archive (older, lagged) + forecast (recent + future) into one continuous series.
    """
    local_tz = get_tz_from_latlon(lat, lon)

    now_local = datetime.now(local_tz)
    stitch_cutoff = (now_local - pd.Timedelta(days=5)).date()  # archive lag

    start = pd.to_datetime(date_start).date()
    end = pd.to_datetime(date_end).date()

    overlap_days = 2
    overlap_start = (pd.Timestamp(stitch_cutoff) - pd.Timedelta(days=overlap_days)).date()

    archive_end = min(end, stitch_cutoff)
    archive_df = pd.DataFrame()
    if start <= archive_end:
        archive_df = fetch_open_meteo_archive(lat, lon, str(start), str(archive_end))

    forecast_start = max(start, overlap_start)
    forecast_df = pd.DataFrame()
    if forecast_start <= end:
        forecast_df = fetch_open_meteo_forecast(lat, lon, str(forecast_start), str(end))

    combined = pd.concat([archive_df, forecast_df], ignore_index=True)
    if combined.empty:
        return combined

    combined["timestamp"] = _ensure_tzaware(combined["timestamp"], local_tz)
    combined = combined.sort_values("timestamp")
    combined = combined.drop_duplicates(subset=["timestamp"], keep="last")  # prefer forecast on overlap
    combined = combined.reset_index(drop=True)
    return combined