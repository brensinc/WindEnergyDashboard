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
# Power curve and derived columns
# -----------------------------
def wind_to_power(wind_speed_mps, cut_in_mps=3.0, rated_speed_mps=12.0, cut_out_mps=25.0, rated_power_mw=3.0):
    ws = np.asarray(wind_speed_mps, dtype=float)
    power = np.zeros_like(ws, dtype=float)

    ramp = (ws >= cut_in_mps) & (ws < rated_speed_mps)
    power[ramp] = rated_power_mw * (ws[ramp] - cut_in_mps) / (rated_speed_mps - cut_in_mps)

    rated = (ws >= rated_speed_mps) & (ws <= cut_out_mps)
    power[rated] = rated_power_mw

    return power

def add_power_output(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    out = df.copy()
    out["power_mw"] = wind_to_power(
        out["wind_speed_mps"].values,
        cut_in_mps=p["cut_in_mps"],
        rated_speed_mps=p["rated_speed_mps"],
        cut_out_mps=p["cut_out_mps"],
        rated_power_mw=p["rated_power_mw"],
    )
    out["energy_mwh"] = out["power_mw"] * 1.0
    return out

def add_data_type(df: pd.DataFrame, local_tz: ZoneInfo) -> pd.DataFrame:
    out = df.copy()
    out["timestamp"] = _ensure_tzaware(out["timestamp"], local_tz)
    now_local = datetime.now(local_tz)
    out["data_type"] = np.where(out["timestamp"] <= now_local, "historical", "forecast")
    return out
