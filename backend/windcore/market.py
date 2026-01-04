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
# gridstatus prices
# -----------------------------
def _date_chunks(start_date: str, end_date: str, freq="MS"):
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    boundaries = pd.date_range(start=start.normalize(), end=end.normalize() + pd.Timedelta(days=1), freq=freq)
    if len(boundaries) == 0 or boundaries[0] != start.normalize():
        boundaries = boundaries.insert(0, start.normalize())
    if boundaries[-1] < end.normalize() + pd.Timedelta(days=1):
        boundaries = boundaries.append(pd.DatetimeIndex([end.normalize() + pd.Timedelta(days=1)]))
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        yield a.date(), (b - pd.Timedelta(days=1)).date()

def get_market_prices_gridstatus(
    iso_name: str,
    start_date: str,
    end_date: str,
    market: str = "DAY_AHEAD_HOURLY",
    location_type: str = "Hub",
    local_tz: ZoneInfo = ZoneInfo("UTC"),
) -> pd.DataFrame:
    iso_class = getattr(gridstatus, iso_name, None)
    if iso_class is None:
        raise ValueError(f"Unsupported ISO class: {iso_name}")
    iso = iso_class()

    sig = inspect.signature(iso.get_lmp)
    base_kwargs = {
        "market": market,
    }
    if "location_type" in sig.parameters:
        base_kwargs["location_type"] = location_type

    # Pull in monthly chunks to reduce rate-limit issues
    dfs = []
    for s, e in _date_chunks(start_date, end_date, freq="MS"):
        kwargs = dict(base_kwargs)
        kwargs["start"] = pd.Timestamp(s)
        kwargs["end"] = pd.Timestamp(e) + pd.Timedelta(days=1)  # treat as exclusive
        df_chunk = iso.get_lmp(**kwargs)
        if df_chunk is not None and len(df_chunk):
            dfs.append(df_chunk)

    if not dfs:
        raise ValueError("No LMP data returned from gridstatus for this query.")

    df = pd.concat(dfs, ignore_index=True)

    time_col = next((c for c in ["Time", "time", "timestamp", "Datetime", "DATETIME"] if c in df.columns), None)
    if time_col is None:
        raise ValueError(f"Could not find a time column in LMP data. Columns: {list(df.columns)}")
    price_col = next((c for c in ["LMP", "lmp", "price", "Price", "LBMP"] if c in df.columns), None)
    if price_col is None:
        raise ValueError(f"Could not find a price column in LMP data. Columns: {list(df.columns)}")

    out = df.rename(columns={time_col: "timestamp", price_col: "price_usd_mwh"}).copy()

    iso_tz = ZoneInfo(ISO_TZ.get(iso_name, "UTC"))
    out["timestamp"] = pd.to_datetime(out["timestamp"])

    # Localize/convert to ISO tz first (handles DST issues), then convert to project tz
    if getattr(out["timestamp"].dt, "tz", None) is None:
        out["timestamp"] = out["timestamp"].dt.tz_localize(
            iso_tz, ambiguous="infer", nonexistent="shift_forward"
        )
    else:
        out["timestamp"] = out["timestamp"].dt.tz_convert(iso_tz)

    out["timestamp"] = out["timestamp"].dt.tz_convert(local_tz)

    out = (
        out.groupby("timestamp", as_index=False)["price_usd_mwh"]
        .mean()
        .sort_values("timestamp")
    )
    return out