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
# Constants
# -----------------------------
PROJECTS_FILE = Path("projects.json")

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
REFERENCE_HUB_HEIGHT_M = 100

ISO_CHOICES = ["CAISO", "ERCOT", "ISONE", "MISO", "NYISO", "PJM", "SPP"]
LMP_MARKETS = ["DAY_AHEAD_HOURLY", "REAL_TIME_HOURLY", "REAL_TIME_5_MIN"]

PRICE_MODELS = ["Fixed", "Market"]

ISO_TZ = {
    "CAISO": "America/Los_Angeles",
    "PJM": "America/New_York",
    "NYISO": "America/New_York",
    "ISONE": "America/New_York",
    "MISO": "America/Chicago",
    "SPP": "America/Chicago",
    "ERCOT": "America/Chicago",
}

DEFAULT_PROJECT = {
    "project_id": "demo",
    "name": "Demo Wind Project",
    "latitude": 37.75,
    "longitude": -122.45,
    "hub_height_m": 100,
    "cut_in_mps": 3.0,
    "rated_speed_mps": 12.0,
    "cut_out_mps": 25.0,
    "rated_power_mw": 3.0,
    "pricing_model": "fixed",           # "fixed" or "market"
    "fixed_price_usd_mwh": 50.0,
    "iso_name": "CAISO",
    "lmp_market": "DAY_AHEAD_HOURLY",
    "location_type": "Hub",
}
