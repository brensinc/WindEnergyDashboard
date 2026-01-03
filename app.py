import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import pydeck as pdk
import inspect
from timezonefinder import TimezoneFinder
from zoneinfo import ZoneInfo 
import gridstatus

# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="Wind Energy Dashboard", layout="wide")

PROJECTS_FILE = Path("projects.json")  # optional persistence

DEFAULT_PROJECT = {
    "project_id": "demo",
    "name": "Demo Wind Project",
    "latitude": 37.75,
    "longitude": -122.45,
    "hub_height_m": 100,  # reference only (fixed at 100m)
    "cut_in_mps": 3.0,
    "rated_speed_mps": 12.0,
    "cut_out_mps": 25.0,
    "rated_power_mw": 3.0,
}


# Pricing options
PRICE_MODELS = ["Fixed price ($/MWh)", "Market price (gridstatus)"]

# ISO choices (keep this explicit & stable)
ISO_CHOICES = ["CAISO", "ERCOT", "ISONE", "MISO", "NYISO", "PJM", "SPP"]

ISO_TZ = {
    "CAISO": "America/Los_Angeles",
    "PJM": "America/New_York",
    "NYISO": "America/New_York",
    "ISONE": "America/New_York",
    "MISO": "America/Chicago",
    "SPP": "America/Chicago",
    "ERCOT": "America/Chicago",
}


# Market choices (these strings are used by gridstatus across ISOs for LMP queries)
LMP_MARKETS = ["DAY_AHEAD_HOURLY", "REAL_TIME_HOURLY", "REAL_TIME_5_MIN"]

# Add API endpoint constants
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

REFERENCE_HUB_HEIGHT_M = 100  # fixed reference height


# -----------------------------
# Persistence helpers
# -----------------------------
def load_projects_from_disk() -> dict:
    if PROJECTS_FILE.exists():
        try:
            return json.loads(PROJECTS_FILE.read_text())
        except Exception:
            return {}
    return {}

def save_projects_to_disk(projects: dict) -> None:
    PROJECTS_FILE.write_text(json.dumps(projects, indent=2, sort_keys=True))

@st.cache_data(ttl=3600, show_spinner=False)
def get_market_prices_gridstatus(
    iso_name: str,
    start_date: str,
    end_date: str,
    market: str = "DAY_AHEAD_HOURLY",
    location_type: str = "Hub",
    local_tz_str: str = "UTC",
) -> pd.DataFrame:
    iso_class = getattr(gridstatus, iso_name, None)
    if iso_class is None:
        raise ValueError(f"Unsupported ISO class: {iso_name}")

    iso = iso_class()

    sig = inspect.signature(iso.get_lmp)
    kwargs = {
        "start": pd.Timestamp(start_date),
        "end": pd.Timestamp(end_date),
        "market": market,
    }
    if "location_type" in sig.parameters:
        kwargs["location_type"] = location_type

    # df = iso.get_lmp(**kwargs)

    dfs = []
    for s, e in _date_chunks(start_date, end_date, freq="MS"):  # month chunks
        chunk_kwargs = dict(kwargs)
        chunk_kwargs["start"] = pd.Timestamp(s)
        chunk_kwargs["end"] = pd.Timestamp(e) + pd.Timedelta(days=1)  # many APIs treat end as exclusive
        df_chunk = iso.get_lmp(**chunk_kwargs)
        if df_chunk is not None and len(df_chunk):
            dfs.append(df_chunk)

    if not dfs:
        raise ValueError("No LMP data returned from gridstatus for this query.")

    df = pd.concat(dfs, ignore_index=True)


    if df is None or len(df) == 0:
        raise ValueError("No LMP data returned from gridstatus for this query.")

    time_col = next((c for c in ["Time", "time", "timestamp", "Datetime", "DATETIME"] if c in df.columns), None)
    if time_col is None:
        raise ValueError(f"Could not find a time column in LMP data. Columns: {list(df.columns)}")

    price_col = next((c for c in ["LMP", "lmp", "price", "Price", "LBMP"] if c in df.columns), None)
    if price_col is None:
        raise ValueError(f"Could not find a price column in LMP data. Columns: {list(df.columns)}")

    out = df.rename(columns={time_col: "timestamp", price_col: "price_usd_mwh"}).copy()

    iso_tz = ZoneInfo(ISO_TZ.get(iso_name, "UTC"))
    local_tz = ZoneInfo(local_tz_str)

    out["timestamp"] = pd.to_datetime(out["timestamp"])
    # if out["timestamp"].dt.tz is None:
    #     out["timestamp"] = out["timestamp"].dt.tz_localize(iso_tz)
    if out["timestamp"].dt.tz is None:
        out["timestamp"] = out["timestamp"].dt.tz_localize(
            iso_tz,
            ambiguous="infer",
            nonexistent="shift_forward",
        )
    else:
        out["timestamp"] = out["timestamp"].dt.tz_convert(iso_tz)

    # Convert to *project local tz* for merging
    out["timestamp"] = out["timestamp"].dt.tz_convert(local_tz)

    out = (
        out.groupby("timestamp", as_index=False)["price_usd_mwh"]
        .mean()
        .sort_values("timestamp")
    )
    return out


def _wind_100m_from_80_120(df: pd.DataFrame,
                          v80_col: str = "wind_speed_80m_mps",
                          v120_col: str = "wind_speed_120m_mps",
                          alpha_default: float = 0.14,
                          alpha_min: float = 0.0,
                          alpha_max: float = 0.5) -> pd.Series:
    """
    Derive wind speed at 100m from 80m & 120m using a time-varying power-law exponent.
    v(z) = v(z1) * (z/z1)^alpha
    alpha = ln(v120/v80) / ln(120/80)
    """
    v80 = df[v80_col].astype(float)
    v120 = df[v120_col].astype(float)

    valid = (v80 > 0) & (v120 > 0)

    alpha = pd.Series(alpha_default, index=df.index, dtype=float)
    alpha.loc[valid] = np.log(v120[valid] / v80[valid]) / np.log(120.0 / 80.0)
    alpha = alpha.clip(alpha_min, alpha_max)

    v100 = v80 * (100.0 / 80.0) ** alpha

    # If v80 missing but v120 present, fallback:
    missing_v100 = v100.isna() & (v120 > 0)
    v100.loc[missing_v100] = v120[missing_v100] * (100.0 / 120.0) ** alpha_default

    return v100


def get_tz_from_latlon(lat: float, lon: float) -> ZoneInfo:
    _tf = TimezoneFinder()
    tzname = _tf.timezone_at(lat=lat, lng=lon)
    if tzname is None:
        return ZoneInfo("UTC")
    return ZoneInfo(tzname)


# def _ensure_tzaware(dt_series: pd.Series, tz: ZoneInfo) -> pd.Series:
#     """Ensure a datetime series is tz-aware in tz (localize if naive, convert if aware)."""
#     s = pd.to_datetime(dt_series)
#     if getattr(s.dt, "tz", None) is None:
#         return s.dt.tz_localize(tz)
#     return s.dt.tz_convert(tz)

def _ensure_tzaware(dt_like, tz: ZoneInfo):
    """
    Ensure datetime-like is tz-aware in tz.
    Works for: Series, DatetimeIndex, list/array of datetime-like.
    - If naive: localize to tz (handling DST safely)
    - If aware: convert to tz
    """
    x = pd.to_datetime(dt_like)

    # DatetimeIndex path
    if isinstance(x, pd.DatetimeIndex):
        if x.tz is None:
            return x.tz_localize(tz, ambiguous="infer", nonexistent="shift_forward")
        return x.tz_convert(tz)
    

    # Series path
    if isinstance(x, pd.Series):
        # if Series is naive
        if x.dt.tz is None:
            return x.dt.tz_localize(tz, ambiguous="infer", nonexistent="shift_forward")
        return x.dt.tz_convert(tz)
    
    # Fallback: convert to Series then return Series (rare)
    s = pd.Series(x)
    if s.dt.tz is None:
        return s.dt.tz_localize(tz, ambiguous="infer", nonexistent="shift_forward")
    return s.dt.tz_convert(tz)


# def _localize_safely(naive_dt: pd.Series, tz: ZoneInfo) -> pd.Series:
#     """
#     Localize naive datetimes to tz while handling DST transitions:
#       - ambiguous (fall back): choose the *first* occurrence (DST) or infer if possible
#       - nonexistent (spring forward): shift forward to the next valid time
#     """

#     s = pd.to_datetime(naive_dt)


#     return s.dt.tz_localize(
#         tz,
#         ambiguous="infer",          # tries to infer based on monotonic order
#         nonexistent="shift_forward" # moves 02:xx -> 03:xx on spring forward
#     )

def _localize_safely(naive_dt, tz: ZoneInfo):
    """
    Localize datetimes to tz while handling DST transitions.

    Works whether naive_dt is:
      - list of strings
      - Series
      - DatetimeIndex

    Handles:
      - ambiguous (fall back): try infer, fallback to first occurrence
      - nonexistent (spring forward): shift_forward
    """
    s = pd.to_datetime(naive_dt)

    def _localize_idx(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
        if idx.tz is not None:
            return idx.tz_convert(tz)
        # Try infer; if it fails (common around DST duplicates), pick first occurrence.
        try:
            return idx.tz_localize(tz, ambiguous="infer", nonexistent="shift_forward")
        except Exception:
            return idx.tz_localize(tz, ambiguous=True, nonexistent="shift_forward")  # choose first

    if isinstance(s, pd.DatetimeIndex):
        return _localize_idx(s)

    # Otherwise assume Series-like
    if getattr(s.dt, "tz", None) is not None:
        return s.dt.tz_convert(tz)

    try:
        return s.dt.tz_localize(tz, ambiguous="infer", nonexistent="shift_forward")
    except Exception:
        return s.dt.tz_localize(tz, ambiguous=True, nonexistent="shift_forward")  # choose first


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


def _date_range_in_tz(ts: pd.Series, tz: ZoneInfo) -> tuple[str, str]:
    """Return start/end dates (YYYY-MM-DD) in tz based on ts."""
    ts_local = _ensure_tzaware(ts, tz)
    return str(ts_local.min().date()), str(ts_local.max().date())



def _is_valid_range(start_date: str, end_date: str) -> bool:
    return pd.to_datetime(start_date).date() <= pd.to_datetime(end_date).date()


# def add_prices_and_revenue(df: pd.DataFrame, project: dict, local_tz: ZoneInfo) -> pd.DataFrame:
#     out = df.copy()
#     out["timestamp"] = _ensure_tzaware(out["timestamp"], local_tz)

#     if project.get("pricing_model") == "fixed":
#         fixed = float(project.get("fixed_price_usd_mwh", 0.0))
#         out["price_usd_mwh"] = fixed
#         out["revenue_usd"] = out["energy_mwh"] * out["price_usd_mwh"]
#         return out

#     if project.get("pricing_model") != "market":
#         out["price_usd_mwh"] = np.nan
#         out["revenue_usd"] = np.nan
#         return out

#     iso_name = project.get("iso_name")
#     market = project.get("lmp_market", "DAY_AHEAD_HOURLY")
#     location_type = project.get("location_type", "Hub")

#     # IMPORTANT: end_date from gridstatus is often exclusive -> add 1 day
#     start_date_local = out["timestamp"].min().date()
#     end_date_local_exclusive = (pd.Timestamp(out["timestamp"].max().date()) + pd.Timedelta(days=1)).date()

#     prices = get_market_prices_gridstatus(
#         iso_name=iso_name,
#         start_date=str(start_date_local),
#         end_date=str(end_date_local_exclusive),
#         market=market,
#         location_type=location_type,
#         local_tz_str=str(local_tz.key),
#     )

#     prices["timestamp"] = _ensure_tzaware(prices["timestamp"], local_tz)

#     st.write("Wind time range:", out["timestamp"].min(), out["timestamp"].max())
#     st.write("Price time range:", prices["timestamp"].min(), prices["timestamp"].max())
#     st.write("Price rows:", len(prices))

#     out = out.sort_values("timestamp")
#     prices = prices.sort_values("timestamp")

#     out = pd.merge_asof(
#         out,
#         prices,
#         on="timestamp",
#         direction="backward"    
#     )

#     # Optional: fill small gaps but don't smear across big holes
#     out["price_usd_mwh"] = out["price_usd_mwh"].ffill()

#     out["revenue_usd"] = out["energy_mwh"] * out["price_usd_mwh"]
#     out["revenue_usd"] = out["revenue_usd"].astype(float)

#     return out

def add_prices_and_revenue(
    df: pd.DataFrame,
    project: dict,
    local_tz: ZoneInfo,
    include_prices: bool = True,
) -> pd.DataFrame:
    out = df.copy()
    out["timestamp"] = _ensure_tzaware(out["timestamp"], local_tz)

    # If user doesn't want prices, skip everything and return NaNs
    if not include_prices:
        out["price_usd_mwh"] = np.nan
        out["revenue_usd"] = np.nan
        return out

    if project.get("pricing_model") == "fixed":
        fixed = float(project.get("fixed_price_usd_mwh", 0.0))
        out["price_usd_mwh"] = fixed
        out["revenue_usd"] = out["energy_mwh"] * out["price_usd_mwh"]
        return out

    if project.get("pricing_model") != "market":
        out["price_usd_mwh"] = np.nan
        out["revenue_usd"] = np.nan
        return out

    # --- existing market price logic below unchanged ---
    iso_name = project.get("iso_name")
    market = project.get("lmp_market", "DAY_AHEAD_HOURLY")
    location_type = project.get("location_type", "Hub")

    start_date_local = out["timestamp"].min().date()
    end_date_local_exclusive = (pd.Timestamp(out["timestamp"].max().date()) + pd.Timedelta(days=1)).date()

    prices = get_market_prices_gridstatus(
        iso_name=iso_name,
        start_date=str(start_date_local),
        end_date=str(end_date_local_exclusive),
        market=market,
        location_type=location_type,
        local_tz_str=str(local_tz.key),
    )

    prices["timestamp"] = _ensure_tzaware(prices["timestamp"], local_tz)

    out = out.sort_values("timestamp")
    prices = prices.sort_values("timestamp")

    out = pd.merge_asof(out, prices, on="timestamp", direction="backward")
    out["price_usd_mwh"] = out["price_usd_mwh"].ffill()

    out["revenue_usd"] = (out["energy_mwh"] * out["price_usd_mwh"]).astype(float)
    return out



@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_open_meteo_archive(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    local_tz = get_tz_from_latlon(lat, lon)

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ["wind_speed_100m", "wind_direction_100m"],
        "wind_speed_unit": "ms",
        "timezone": str(local_tz.key),   # IMPORTANT: pass tz name string
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

    # Open-Meteo returns "YYYY-MM-DDTHH:MM" strings in the timezone you request.
    # ts = pd.to_datetime(time).tz_localize(local_tz)
    ts = _localize_safely(time, local_tz)


    df = pd.DataFrame({
        "timestamp": ts,
        "wind_speed_mps": ws100,
        "wind_direction_deg": wd100 if wd100 else [np.nan] * len(ws100),
        "source": "archive",
        "hub_height_m": REFERENCE_HUB_HEIGHT_M,
    })
    return df


@st.cache_data(ttl=900, show_spinner=False)
def _fetch_open_meteo_forecast(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
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

    # ts = pd.to_datetime(time).tz_localize(local_tz)
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



def get_wind_data(lat: float, lon: float, date_start: str, date_end: str, hub_height_m: int = 100) -> pd.DataFrame:
    local_tz = get_tz_from_latlon(lat, lon)
   
   
    now_local = datetime.now(local_tz)
    stitch_cutoff = (now_local - pd.Timedelta(days=5)).date()  # archive lag


    start = pd.to_datetime(date_start).date()
    end = pd.to_datetime(date_end).date()

    overlap_days = 2
    overlap_start = (pd.Timestamp(stitch_cutoff) - pd.Timedelta(days=overlap_days)).date()



    archive_df = pd.DataFrame()
    archive_end = min(end, stitch_cutoff)
    if start <= archive_end:
        archive_df = _fetch_open_meteo_archive(lat, lon, str(start), str(archive_end))


    forecast_df = pd.DataFrame()
    forecast_start = max(start, overlap_start)
    if forecast_start <= end:
        forecast_df = _fetch_open_meteo_forecast(lat, lon, str(forecast_start), str(end))



    combined = pd.concat([archive_df, forecast_df], ignore_index=True)
    if combined.empty:
        return combined

    combined["timestamp"] = _ensure_tzaware(combined["timestamp"], local_tz)
    combined = combined.sort_values("timestamp")
    combined = combined.drop_duplicates(subset=["timestamp"], keep="last")
    combined = combined.reset_index(drop=True)
        


    return combined



def add_data_type(df: pd.DataFrame, local_tz: ZoneInfo) -> pd.DataFrame:
    now_local = datetime.now(local_tz)
    out = df.copy()
    out["timestamp"] = _ensure_tzaware(out["timestamp"], local_tz)
    out["data_type"] = np.where(out["timestamp"] <= now_local, "historical", "forecast")
    return out


# -----------------------------
# Power curve + power output
# -----------------------------
def wind_to_power(
    wind_speed_mps,
    cut_in_mps=3.0,
    rated_speed_mps=12.0,
    cut_out_mps=25.0,
    rated_power_mw=3.0
):
    """
    Idealized wind turbine power curve:
      - 0 below cut-in
      - linear ramp cut-in -> rated speed
      - rated power rated speed -> cut-out
      - 0 above cut-out
    """
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
    # hourly data => energy per interval (MWh) is power(MW) * 1 hour
    out["energy_mwh"] = out["power_mw"] * 1.0
    return out

# -----------------------------
# Generation stats you requested
# -----------------------------
def compute_generation_stats(df: pd.DataFrame, p: dict) -> dict:
    """
    Stats computed on the full dataframe (historical + forecast).
    You can compute separately by data_type if desired.
    """
    ws = df["wind_speed_mps"].to_numpy()
    power = df["power_mw"].to_numpy()
    energy = df["energy_mwh"].to_numpy()

    below_cut_in = ws < p["cut_in_mps"]
    above_cut_out = ws > p["cut_out_mps"]
    operating = (~below_cut_in) & (~above_cut_out)

    # % time (of available timestamps)
    denom = max(len(ws), 1)
    pct_below = 100.0 * below_cut_in.sum() / denom
    pct_above = 100.0 * above_cut_out.sum() / denom
    pct_operating = 100.0 * operating.sum() / denom

    total_energy_mwh = float(np.nansum(energy))
    mean_power_mw = float(np.nanmean(power))
    p50_power_mw = float(np.nanpercentile(power, 50))
    p90_power_mw = float(np.nanpercentile(power, 90))
    max_power_mw = float(np.nanmax(power))

    # capacity factor over the period (using mean power / rated power)
    capacity_factor_pct = 100.0 * mean_power_mw / p["rated_power_mw"] if p["rated_power_mw"] > 0 else np.nan

    return {
        "pct_time_below_cut_in": pct_below,
        "pct_time_above_cut_out": pct_above,
        "pct_time_operating": pct_operating,
        "total_energy_mwh": total_energy_mwh,
        "mean_power_mw": mean_power_mw,
        "p50_power_mw": p50_power_mw,
        "p90_power_mw": p90_power_mw,
        "max_power_mw": max_power_mw,
        "capacity_factor_pct": capacity_factor_pct,
    }

def _add_time_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out["year"] = out["timestamp"].dt.year
    out["month"] = out["timestamp"].dt.month
    out["month_name"] = out["timestamp"].dt.strftime("%b")

    # Simple meteorological seasons
    # DJF=Winter, MAM=Spring, JJA=Summer, SON=Fall
    season_map = {
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Spring", 4: "Spring", 5: "Spring",
        6: "Summer", 7: "Summer", 8: "Summer",
        9: "Fall", 10: "Fall", 11: "Fall",
    }
    out["season"] = out["month"].map(season_map)
    return out


def monthly_capacity_factor(df_hist: pd.DataFrame, rated_power_mw: float) -> pd.DataFrame:
    d = _add_time_cols(df_hist)
    # CF over hours = mean power / rated power
    g = d.groupby(["month", "month_name"], as_index=False)["power_mw"].mean()
    g["capacity_factor"] = g["power_mw"] / max(rated_power_mw, 1e-9)
    g = g.sort_values("month")
    return g[["month", "month_name", "capacity_factor"]]


def seasonal_capacity_factor(df_hist: pd.DataFrame, rated_power_mw: float) -> pd.DataFrame:
    d = _add_time_cols(df_hist)
    g = d.groupby("season", as_index=False)["power_mw"].mean()
    g["capacity_factor"] = g["power_mw"] / max(rated_power_mw, 1e-9)

    order = ["Winter", "Spring", "Summer", "Fall"]
    g["season"] = pd.Categorical(g["season"], categories=order, ordered=True)
    return g.sort_values("season")[["season", "capacity_factor"]]


def annual_energy_by_year(df_hist: pd.DataFrame) -> pd.DataFrame:
    d = _add_time_cols(df_hist)
    g = d.groupby("year", as_index=False)["energy_mwh"].sum()
    g = g.sort_values("year")
    g.rename(columns={"energy_mwh": "annual_energy_mwh"}, inplace=True)
    return g


def bootstrap_p50_p90_annual_energy(
    df_hist: pd.DataFrame,
    n_boot: int = 2000,
    random_state: int = 7
) -> dict:
    """
    Simple bootstrap over historical years:
    - compute annual energy for each year
    - sample years with replacement and take mean annual energy
    - report P50/P90 of that bootstrap distribution
    """
    yearly = annual_energy_by_year(df_hist)
    vals = yearly["annual_energy_mwh"].to_numpy()
    if len(vals) == 0:
        return {"p50_mwh": np.nan, "p90_mwh": np.nan, "boot_mean_mwh": np.nan, "n_years": 0}

    rng = np.random.default_rng(random_state)
    boots = rng.choice(vals, size=(n_boot, len(vals)), replace=True).mean(axis=1)

    return {
        "p50_mwh": float(np.percentile(boots, 50)),
        "p90_mwh": float(np.percentile(boots, 90)),
        "boot_mean_mwh": float(np.mean(boots)),
        "n_years": int(len(vals)),
    }


def exceedance_curve(df_hist: pd.DataFrame, max_speed: float = 35.0, step: float = 0.5) -> pd.DataFrame:
    """
    Exceedance curve: for each threshold v, compute P(wind_speed >= v).
    """
    ws = df_hist["wind_speed_mps"].astype(float).to_numpy()
    ws = ws[~np.isnan(ws)]
    if len(ws) == 0:
        return pd.DataFrame(columns=["threshold_mps", "exceedance_pct"])

    thresholds = np.arange(0, max_speed + step, step)
    exceed = [(t, 100.0 * np.mean(ws >= t)) for t in thresholds]
    return pd.DataFrame(exceed, columns=["threshold_mps", "exceedance_pct"])


def exceedance_key_points(df_hist: pd.DataFrame, p: dict) -> dict:
    ws = df_hist["wind_speed_mps"].astype(float)
    if ws.dropna().empty:
        return {"pct_above_rated": np.nan, "pct_above_cut_out": np.nan, "pct_below_cut_in": np.nan}

    return {
        "pct_below_cut_in": float(100.0 * (ws < p["cut_in_mps"]).mean()),
        "pct_above_rated":  float(100.0 * (ws >= p["rated_speed_mps"]).mean()),
        "pct_above_cut_out":float(100.0 * (ws > p["cut_out_mps"]).mean()),
    }


def downtime_risk_by_month(df_hist: pd.DataFrame, p: dict) -> pd.DataFrame:
    d = _add_time_cols(df_hist)
    ws = d["wind_speed_mps"].astype(float)

    d["below_cut_in"] = ws < p["cut_in_mps"]
    d["above_cut_out"] = ws > p["cut_out_mps"]

    g = d.groupby(["month", "month_name"], as_index=False).agg(
        pct_below_cut_in=("below_cut_in", lambda x: 100.0 * x.mean()),
        pct_above_cut_out=("above_cut_out", lambda x: 100.0 * x.mean()),
    ).sort_values("month")

    return g[["month", "month_name", "pct_below_cut_in", "pct_above_cut_out"]]


def interannual_variability(df_hist: pd.DataFrame, rated_power_mw: float) -> dict:
    """
    How much each year differs from the 'typical year' monthly CF profile.
    Returns:
      - typical monthly CF (climatology)
      - year-by-year deviation score (RMSE vs typical)
      - annual energy table + coefficient of variation
    """
    d = _add_time_cols(df_hist)

    # monthly CF by year
    monthly_year = d.groupby(["year", "month"], as_index=False)["power_mw"].mean()
    monthly_year["cf"] = monthly_year["power_mw"] / max(rated_power_mw, 1e-9)

    # typical year monthly CF (climatology)
    typical = monthly_year.groupby("month", as_index=False)["cf"].mean()
    typical.rename(columns={"cf": "typical_cf"}, inplace=True)

    # join to compute deviations
    merged = monthly_year.merge(typical, on="month", how="left")
    merged["sq_err"] = (merged["cf"] - merged["typical_cf"]) ** 2

    rmse = merged.groupby("year", as_index=False)["sq_err"].mean()
    rmse["rmse_cf"] = np.sqrt(rmse["sq_err"])
    rmse = rmse.drop(columns=["sq_err"]).sort_values("year")

    # annual energy + variability summary
    yearly_energy = annual_energy_by_year(d)
    cv = float(yearly_energy["annual_energy_mwh"].std(ddof=1) / yearly_energy["annual_energy_mwh"].mean()) if len(yearly_energy) > 1 else np.nan

    return {
        "typical_monthly_cf": typical,
        "yearly_rmse_vs_typical": rmse,
        "yearly_energy": yearly_energy,
        "annual_energy_cv": cv,
    }


# -----------------------------
# Initialize state
# -----------------------------
if "projects" not in st.session_state:
    disk_projects = load_projects_from_disk()
    # Ensure at least demo project exists
    if "demo" not in disk_projects:
        disk_projects["demo"] = DEFAULT_PROJECT
        save_projects_to_disk(disk_projects)
    st.session_state.projects = disk_projects

if "selected_project_id" not in st.session_state:
    st.session_state.selected_project_id = "demo"

# -----------------------------
# Sidebar: create/select project
# -----------------------------
st.sidebar.header("Projects")

project_ids = list(st.session_state.projects.keys())
selected = st.sidebar.selectbox(
    "Select a project",
    options=project_ids,
    index=project_ids.index(st.session_state.selected_project_id) if st.session_state.selected_project_id in project_ids else 0,
)
st.session_state.selected_project_id = selected


with st.sidebar.expander("Create a new project", expanded=False):
    pricing_model = st.selectbox("Pricing model", PRICE_MODELS, index=0)
    
    with st.form("create_project_form"):
        name = st.text_input("Project name", value="New Wind Project")
        lat = st.number_input("Latitude", value=37.75, format="%.6f")
        lon = st.number_input("Longitude", value=-122.45, format="%.6f")

        fixed_price = None
        iso_name = None
        lmp_market = None
        location_type = None

        if pricing_model == "Fixed price ($/MWh)":
            fixed_price = st.number_input("Fixed price ($/MWh)", value=50.0, step=1.0)
        else:
            iso_name = st.selectbox("ISO / Market", ISO_CHOICES, index=0)
            lmp_market = st.selectbox("LMP market", LMP_MARKETS, index=0)
            # For an MVP, use hubs (much smaller than nodal/bus data)
            location_type = st.selectbox("Location type", ["Hub"], index=0)


        # Replace with:
        st.info("Wind reference height is fixed at 100m (archive uses 100m directly; forecast derives 100m from 80m & 120m).")
        hub_height_m = 100

        cut_in = st.number_input("Cut-in speed (m/s)", value=3.0, step=0.1)
        rated_speed = st.number_input("Rated speed (m/s)", value=12.0, step=0.1)
        cut_out = st.number_input("Cut-out speed (m/s)", value=25.0, step=0.1)
        rated_power = st.number_input("Rated power (MW)", value=3.0, step=0.1)

        new_id = st.text_input("Project ID (unique)", value=f"proj_{len(project_ids)+1}")

        persist = st.checkbox("Save to projects.json (recommended)", value=True)

        submitted = st.form_submit_button("Create project")

    if submitted:
        if new_id in st.session_state.projects:
            st.error("That Project ID already exists. Choose another.")
        else:
            st.session_state.projects[new_id] = {
                "project_id": new_id,
                "name": name,
                "latitude": float(lat),
                "longitude": float(lon),
                # "hub_height_m": int(hub_height_m),
                "hub_height_m": 100,
                "cut_in_mps": float(cut_in),
                "rated_speed_mps": float(rated_speed),
                "cut_out_mps": float(cut_out),
                "rated_power_mw": float(rated_power),
                "pricing_model": "fixed" if pricing_model.startswith("Fixed") else "market",
                "fixed_price_usd_mwh": float(fixed_price) if fixed_price is not None else None,
                "iso_name": iso_name,
                "lmp_market": lmp_market,
                "location_type": location_type,

            }
            if persist:
                save_projects_to_disk(st.session_state.projects)
            st.session_state.selected_project_id = new_id
            st.success(f"Created project: {new_id}")

with st.sidebar.expander("Delete a project"):
    deletable_ids = [k for k in st.session_state.projects.keys() if k != "demo"]
    delete_id = st.selectbox("Choose project to delete", options=deletable_ids)
    if st.button("Delete this project", type="secondary"):
        st.session_state.projects.pop(delete_id, None)
        save_projects_to_disk(st.session_state.projects)
        if st.session_state.selected_project_id == delete_id:
            st.session_state.selected_project_id = "demo"
        st.cache_data.clear()
        st.rerun()

st.sidebar.divider()
if st.sidebar.button("Clear cached data"):
    st.cache_data.clear()
    st.sidebar.success("Cache cleared.")

# -----------------------------
# Main page: dashboard
# -----------------------------
p = st.session_state.projects[st.session_state.selected_project_id]
st.title("Wind Energy Dashboard")
st.caption("Historical vs forecast wind + estimated generation using an idealized turbine power curve.")

# Date range controls
# with st.sidebar:
#     st.subheader("Data Range")
#     date_start = st.date_input("Start date", value=pd.to_datetime("2025-11-20").date())
#     date_end = st.date_input("End date", value=pd.to_datetime("2025-12-10").date())

with st.sidebar:
    st.subheader("Query")

    with st.form("query_form"):
        date_start = st.date_input("Start date", value=pd.to_datetime("2025-11-20").date())
        date_end   = st.date_input("End date",   value=pd.to_datetime("2025-12-10").date())

        include_prices = st.checkbox("Pull price data", value=True)
        # (optional) let them run wind-only even if pricing_model is market
        # include_prices controls the actual fetch.

        requery = st.form_submit_button("Requery")


# Calculate local time
# local_tz = get_tz_from_latlon(p["latitude"], p["longitude"])


# # Fetch + compute
# try:
#     with st.spinner("Loading wind data..."):
#         df = get_wind_data(p["latitude"], p["longitude"], str(date_start), str(date_end), p["hub_height_m"])
#         df = add_data_type(df, local_tz)
#         df = add_power_output(df, p)
#         df = add_prices_and_revenue(df, p, local_tz)
# except Exception as e:
#     st.error(f"Failed to load data: {e}")
#     st.stop()

local_tz = get_tz_from_latlon(p["latitude"], p["longitude"])

need_load = ("df_cached" not in st.session_state) or requery

if need_load:
    try:
        with st.spinner("Loading wind data..."):
            df = get_wind_data(p["latitude"], p["longitude"], str(date_start), str(date_end), p["hub_height_m"])
            df = add_data_type(df, local_tz)
            df = add_power_output(df, p)
            df = add_prices_and_revenue(df, p, local_tz, include_prices=include_prices)

        st.session_state.df_cached = df
        st.session_state.include_prices_cached = include_prices
        st.session_state.date_start_cached = date_start
        st.session_state.date_end_cached = date_end

    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()
else:
    df = st.session_state.df_cached
    include_prices = st.session_state.get("include_prices_cached", True)

    

# Layout
colA, colB = st.columns([1, 2]) # Modulates width of each column

with colA:
    st.subheader("Project")
    st.write(f"**{p['name']}** (`{p['project_id']}`)")
    st.write(f"Lat/Lon: **{p['latitude']:.4f}, {p['longitude']:.4f}**")
    st.write(f"ISO Name: **{iso_name}**")
    st.write(f"lmp_market: **{lmp_market}**")

    st.write(f"Hub height used: **{int(df['hub_height_m'].iloc[0])} m**")
    st.write(f"Rated power: **{p['rated_power_mw']} MW**")
    st.write(f"Cut-in / Rated / Cut-out: **{p['cut_in_mps']} / {p['rated_speed_mps']} / {p['cut_out_mps']} m/s**")

    # map_df = pd.DataFrame({"lat": [p["latitude"]], "lon": [p["longitude"]]})
    # st.map(map_df, zoom=6)
    dot_color = [255, 120, 0]

with colB:
    map_df = pd.DataFrame({
        "lat": [p["latitude"]],
        "lon": [p["longitude"]],
        "label": [f"{p['name']}\n({p['latitude']:.4f}, {p['longitude']:.4f})"],
    })

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position="[lon, lat]",
        get_radius=2000,   # adjust for zoom/scale
        pickable=True, 
        get_fill_color=dot_color,

    )

    tooltip = {
        "text": "{label}"
    }

    view_state = pdk.ViewState(
        latitude=p["latitude"],
        longitude=p["longitude"],
        zoom=7,
    )

    st.pydeck_chart(pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style='light'
    ))


# with colB:
st.subheader("Wind Speed (m/s)")
fig_ws = px.line(
    df,
    x="timestamp",
    y="wind_speed_mps",
    color="data_type",
    line_dash="data_type",
    hover_data=["wind_direction_deg", "hub_height_m"],
    labels={"wind_speed_mps": "Wind speed (m/s)", "timestamp": "Time"},
)
fig_ws.add_hline(y=p["cut_in_mps"], line_dash="dot", annotation_text="Cut-in")
fig_ws.add_hline(y=p["cut_out_mps"], line_dash="dot", annotation_text="Cut-out")

fig_ws.update_traces(
    selector=dict(name="historical"),
    line=dict(dash="solid")
)

fig_ws.update_traces(
    selector=dict(name="forecast"),
    line=dict(dash="dot")
)

fig_ws.update_layout(
    height=350
)


st.plotly_chart(fig_ws, use_container_width=True)

st.subheader("Estimated Power Output (MW)")
fig_p = px.line(
    df,
    x="timestamp",
    y="power_mw",
    color="data_type",
    line_dash="data_type",
    labels={"power_mw": "Power (MW)", "timestamp": "Time"},
)

fig_p.update_traces(
    selector=dict(name="historical"),
    line=dict(dash="solid")
)

fig_p.update_traces(
    selector=dict(name="forecast"),
    line=dict(dash="dot")
)

fig_p.update_layout(
    height=350
)


st.plotly_chart(fig_p, use_container_width=True)


# Stats section
st.subheader("Generation Stats")
stats_all = compute_generation_stats(df, p)

c1, c2, c3, c4 = st.columns(4)
c1.metric("% time below cut-in", f"{stats_all['pct_time_below_cut_in']:.1f}%")
c2.metric("% time above cut-out", f"{stats_all['pct_time_above_cut_out']:.1f}%")
c3.metric("% time operating", f"{stats_all['pct_time_operating']:.1f}%")
c4.metric("Capacity factor (period)", f"{stats_all['capacity_factor_pct']:.1f}%")

d1, d2, d3, d4 = st.columns(4)
d1.metric("Total energy (MWh)", f"{stats_all['total_energy_mwh']:.1f}")
d2.metric("Mean power (MW)", f"{stats_all['mean_power_mw']:.2f}")
d3.metric("P50 power (MW)", f"{stats_all['p50_power_mw']:.2f}")
d4.metric("P90 power (MW)", f"{stats_all['p90_power_mw']:.2f}")

st.divider()
st.header("Analytics (Historical)")

hist_df = df[df["data_type"] == "historical"].copy()

if hist_df.empty:
    st.info("No historical data in the selected range (only forecast). Expand the date range backward.")
else:
    # --- Monthly/Seasonal CF ---
    m_cf = monthly_capacity_factor(hist_df, p["rated_power_mw"])
    s_cf = seasonal_capacity_factor(hist_df, p["rated_power_mw"])

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Monthly capacity factor profile")
        fig_mcf = px.bar(m_cf, x="month_name", y="capacity_factor",
                         labels={"month_name": "Month", "capacity_factor": "Capacity factor"})
        st.plotly_chart(fig_mcf, use_container_width=True)

    with c2:
        st.subheader("Seasonal capacity factor profile")
        fig_scf = px.bar(s_cf, x="season", y="capacity_factor",
                         labels={"season": "Season", "capacity_factor": "Capacity factor"})
        st.plotly_chart(fig_scf, use_container_width=True)

    # --- P50/P90 annual energy (bootstrap) ---
    st.subheader("Annual energy (historical years) + P50/P90 bootstrap")
    yearly = annual_energy_by_year(hist_df)
    boot = bootstrap_p50_p90_annual_energy(hist_df, n_boot=2000)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Years in sample", f"{boot['n_years']}")
    k2.metric("Bootstrap mean AEP (MWh)", f"{boot['boot_mean_mwh']:,.0f}" if not np.isnan(boot["boot_mean_mwh"]) else "—")
    k3.metric("P50 AEP (MWh)", f"{boot['p50_mwh']:,.0f}" if not np.isnan(boot["p50_mwh"]) else "—")
    k4.metric("P90 AEP (MWh)", f"{boot['p90_mwh']:,.0f}" if not np.isnan(boot["p90_mwh"]) else "—")

    fig_aep = px.line(yearly, x="year", y="annual_energy_mwh",
                      markers=True, labels={"annual_energy_mwh": "Annual energy (MWh)", "year": "Year"})
    st.plotly_chart(fig_aep, use_container_width=True)

    # --- Exceedance curves ---
    st.subheader("Exceedance curves")
    exc = exceedance_curve(hist_df, max_speed=max(35.0, float(hist_df["wind_speed_mps"].max() or 35.0)), step=0.5)
    fig_exc = px.line(exc, x="threshold_mps", y="exceedance_pct",
                      labels={"threshold_mps": "Wind speed threshold (m/s)", "exceedance_pct": "% of hours ≥ threshold"})
    st.plotly_chart(fig_exc, use_container_width=True)

    pts = exceedance_key_points(hist_df, p)
    e1, e2, e3 = st.columns(3)
    e1.metric("% hours below cut-in", f"{pts['pct_below_cut_in']:.2f}%")
    e2.metric("% hours ≥ rated speed", f"{pts['pct_above_rated']:.2f}%")
    e3.metric("% hours above cut-out", f"{pts['pct_above_cut_out']:.2f}%")

    # --- Downtime risk by month ---
    st.subheader("Downtime risk by month")
    down = downtime_risk_by_month(hist_df, p)
    down_long = down.melt(id_vars=["month", "month_name"],
                          value_vars=["pct_below_cut_in", "pct_above_cut_out"],
                          var_name="condition", value_name="pct_hours")

    fig_down = px.bar(
        down_long.sort_values("month"),
        x="month_name", y="pct_hours", color="condition",
        barmode="group",
        labels={"month_name": "Month", "pct_hours": "% of hours", "condition": "Condition"},
    )
    st.plotly_chart(fig_down, use_container_width=True)

    # --- Interannual variability vs typical year ---
    st.subheader("Interannual variability")
    iv = interannual_variability(hist_df, p["rated_power_mw"])

    v1, v2 = st.columns(2)
    with v1:
        st.caption("How different each year’s monthly CF pattern is from the typical (climatology) year.")
        fig_rmse = px.bar(iv["yearly_rmse_vs_typical"], x="year", y="rmse_cf",
                          labels={"rmse_cf": "RMSE of monthly CF vs typical", "year": "Year"})
        st.plotly_chart(fig_rmse, use_container_width=True)

    with v2:
        st.caption("Annual energy variability across years.")
        fig_y = px.bar(iv["yearly_energy"], x="year", y="annual_energy_mwh",
                       labels={"annual_energy_mwh": "Annual energy (MWh)", "year": "Year"})
        st.plotly_chart(fig_y, use_container_width=True)

    st.write(f"**Coefficient of variation (annual energy):** {iv['annual_energy_cv']:.3f}" if not np.isnan(iv["annual_energy_cv"]) else "**Coefficient of variation:** —")



st.subheader("Energy Price Time Series ($/MWh)")

fig_price = px.line(
    df,
    x="timestamp",
    y="price_usd_mwh",
    color="data_type",
    line_dash="data_type",
    labels={"price_usd_mwh": "Price ($/MWh)", "timestamp": "Time"},
)

# Optional: match your dash styling
fig_price.update_traces(selector=dict(name="historical"), line=dict(dash="solid"))
fig_price.update_traces(selector=dict(name="forecast"), line=dict(dash="dot"))

fig_price.update_layout(height=300)

st.plotly_chart(fig_price, width="stretch")



st.subheader("Revenue Time Series")

fig_r = px.line(
    df,
    x="timestamp",
    y="revenue_usd",
    color="data_type",
    line_dash="data_type",
    labels={"revenue_usd": "Revenue ($/hr)", "timestamp": "Time"},
)
st.plotly_chart(fig_r, use_container_width=True)

st.subheader("Revenue Stats")

hist = df[df["data_type"] == "historical"].copy()
fcst = df[df["data_type"] == "forecast"].copy()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Avg price ($/MWh)", f"{hist['price_usd_mwh'].mean():.2f}" if len(hist) else "—")
col2.metric("Historical revenue ($)", f"{hist['revenue_usd'].sum():,.0f}" if len(hist) else "—")
col3.metric("Forecast revenue ($)", f"{fcst['revenue_usd'].sum():,.0f}" if len(fcst) else "—")
col4.metric("Total revenue ($)", f"{df['revenue_usd'].sum():,.0f}")


st.caption("Note: Energy is computed from hourly power estimates (power * 1 hour). The power curve is idealized; add an OEM curve later for realism.")
