import requests
import pandas as pd
from datetime import datetime, timezone
import numpy as np


def get_wind_data(lat, lon, date_start, date_end):
    """
    Fetch historical + forecast wind data from Open-Meteo.

    Parameters
    ----------
    lat : float
        Latitude
    lon : float
        Longitude
    date_start : str
        Start date in YYYY-MM-DD format
    date_end : str
        End date in YYYY-MM-DD format

    Returns
    -------
    pd.DataFrame
        DataFrame with timestamp, wind speed, wind direction, and data type
    """

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": lat,
        "longitude": lon,
        # Hourly is usually better for dashboards than 15-min to start
        "hourly": [
            "wind_speed_80m",
            "wind_direction_80m"
        ],
        "wind_speed_unit": "ms",
        "start_date": date_start,
        "end_date": date_end,
        "timezone": "UTC"
    }

    response = requests.get(url, params=params)
    response.raise_for_status()  # fail fast if API errors

    data = response.json()

    # Parse into DataFrame
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(data["hourly"]["time"], utc=True),
        "wind_speed_mps": data["hourly"]["wind_speed_80m"],
        "wind_direction_deg": data["hourly"]["wind_direction_80m"],
    })

    return df

def add_data_type(df):
    now = datetime.now(timezone.utc)    
    df["data_type"] = df["timestamp"].apply(
        lambda t: "historical" if t <= now else "forecast"
    )
    return df


def wind_to_power(
    wind_speed_mps,
    cut_in=3.0,
    rated_speed=12.0,
    cut_out=25.0,
    rated_power_mw=3.0
):
    """
    Convert wind speed to power output using an idealized power curve.

    Parameters
    ----------
    wind_speed_mps : float or np.ndarray
        Wind speed in m/s
    cut_in : float
        Cut-in wind speed (m/s)
    rated_speed : float
        Rated wind speed (m/s)
    cut_out : float
        Cut-out wind speed (m/s)
    rated_power_mw : float
        Rated power (MW)

    Returns
    -------
    power_mw : float or np.ndarray
        Power output (MW)
    """

    ws = np.array(wind_speed_mps)

    power = np.zeros_like(ws, dtype=float)

    # Linear ramp region
    ramp_mask = (ws >= cut_in) & (ws < rated_speed)
    power[ramp_mask] = rated_power_mw * (
        (ws[ramp_mask] - cut_in) / (rated_speed - cut_in)
    )

    # Rated region
    rated_mask = (ws >= rated_speed) & (ws <= cut_out)
    power[rated_mask] = rated_power_mw

    # Outside operating range stays 0
    return power

def add_power_output(df, turbine_params):
    df = df.copy()
    df["power_mw"] = wind_to_power(
        df["wind_speed_mps"].values,
        cut_in=turbine_params["cut_in"],
        rated_speed=turbine_params["rated_speed"],
        cut_out=turbine_params["cut_out"],
        rated_power_mw=turbine_params["rated_power_mw"],
    )
    return df
