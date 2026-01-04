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



def add_prices_and_revenue(df: pd.DataFrame, project: dict, local_tz: ZoneInfo, include_prices: bool) -> pd.DataFrame:
    out = df.copy()
    out["timestamp"] = _ensure_tzaware(out["timestamp"], local_tz)

    # If user doesn't want prices, ensure columns exist and exit
    if not include_prices:
        out["price_usd_mwh"] = np.nan
        out["revenue_usd"] = np.nan
        return out

    # Fixed pricing
    if project.get("pricing_model") == "fixed":
        fixed = float(project.get("fixed_price_usd_mwh", 0.0) or 0.0)
        out["price_usd_mwh"] = fixed
        out["revenue_usd"] = out["energy_mwh"] * out["price_usd_mwh"]
        return out

    # Market pricing
    if project.get("pricing_model") != "market":
        out["price_usd_mwh"] = np.nan
        out["revenue_usd"] = np.nan
        return out

    iso_name = project.get("iso_name")
    market = project.get("lmp_market", "DAY_AHEAD_HOURLY")
    location_type = project.get("location_type", "Hub")

    start_date = str(out["timestamp"].min().date())
    end_date = str(out["timestamp"].max().date())

    prices = get_market_prices_gridstatus(
        iso_name=iso_name,
        start_date=start_date,
        end_date=end_date,
        market=market,
        location_type=location_type,
        local_tz=local_tz,
    )

    prices["timestamp"] = _ensure_tzaware(prices["timestamp"], local_tz)

    # Keep only the columns we need from prices to prevent collisions
    prices = prices[["timestamp", "price_usd_mwh"]].sort_values("timestamp")
    out = out.sort_values("timestamp")

    merged = pd.merge_asof(
        out,
        prices,
        on="timestamp",
        direction="backward",
        # optional, if you want to avoid matching across big gaps:
        # tolerance=pd.Timedelta("1H"),
    )

    # At this point, merged should have exactly one "price_usd_mwh"
    if "price_usd_mwh" not in merged.columns:
        # If you ever hit this, it means a suffix collision still happened
        # (e.g., out already had a price_usd_mwh column).
        # Resolve it explicitly:
        candidates = [c for c in merged.columns if "price_usd_mwh" in c]
        raise KeyError(f"Expected 'price_usd_mwh' after merge. Found: {candidates}")

    merged["price_usd_mwh"] = merged["price_usd_mwh"].ffill()
    merged["revenue_usd"] = merged["energy_mwh"] * merged["price_usd_mwh"]

    return merged
