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

from shapely.geometry import Polygon, Point

ISO_CHOICES = ["CAISO", "Ercot", "ISONE", "MISO", "NYISO", "PJM"]
LMP_MARKETS = ["DAY_AHEAD_HOURLY", 
            #    "REAL_TIME_HOURLY", 
            #    "REAL_TIME_5_MIN"
               ]

ISO_BOUNDARIES = {
    "CAISO": Polygon([
        (-124.5, 42.0), (-124.0, 41.5), (-124.2, 40.5), (-124.1, 39.0),
        (-123.5, 38.0), (-122.5, 37.5), (-121.5, 36.5), (-120.5, 35.5),
        (-119.5, 34.5), (-118.5, 33.5), (-117.5, 32.5), (-116.5, 32.5),
        (-115.5, 33.0), (-114.5, 34.0), (-114.0, 35.0), (-114.5, 36.0),
        (-115.5, 37.0), (-117.0, 37.5), (-118.5, 38.0), (-120.0, 39.0),
        (-121.5, 40.0), (-123.0, 41.0), (-124.5, 42.0)
    ]).buffer(0),
    "Ercot": Polygon([
        (-106.6, 36.5), (-100.0, 36.0), (-97.0, 35.5), (-96.0, 34.0),
        (-95.5, 32.5), (-95.0, 30.0), (-94.0, 28.0), (-93.0, 27.0),
        (-94.0, 26.0), (-97.0, 25.5), (-99.0, 26.0), (-101.0, 28.0),
        (-103.0, 30.0), (-105.0, 33.0), (-106.6, 36.5)
    ]).buffer(0),
    "ISONE": Polygon([
        (-74.0, 45.0), (-73.0, 45.0), (-70.0, 45.0), (-70.0, 42.0),
        (-69.0, 42.0), (-69.0, 41.0), (-71.0, 41.0), (-73.0, 42.0),
        (-74.0, 44.0), (-74.0, 45.0)
    ]).buffer(0),
    "NYISO": Polygon([
        (-79.5, 45.0), (-79.5, 42.0), (-72.0, 42.0), (-72.0, 40.5),
        (-74.0, 40.5), (-75.0, 41.0), (-76.5, 41.5), (-79.5, 42.0),
        (-79.5, 45.0)
    ]).buffer(0),
    "MISO": Polygon([
        (-97.0, 49.0), (-89.0, 49.0), (-82.0, 49.0), (-82.0, 34.5),
        (-84.0, 34.5), (-86.0, 34.5), (-88.0, 34.5), (-90.0, 34.5),
        (-91.0, 34.0), (-92.0, 33.0), (-93.0, 32.0), (-93.0, 31.0),
        (-97.0, 31.0)
    ]).buffer(0),
    "PJM": Polygon([
        (-91.5, 42.5), (-91.5, 34.0), (-72.0, 34.0), (-72.0, 42.5)
    ]).buffer(0),
}

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

# -----------------------------
# Persistence
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

# -----------------------------
# Timezone helpers
# -----------------------------
def get_tz_from_latlon(lat: float, lon: float) -> ZoneInfo:
    tf = TimezoneFinder()
    tzname = tf.timezone_at(lat=lat, lng=lon)
    return ZoneInfo(tzname) if tzname else ZoneInfo("UTC")

def _ensure_tzaware(dt_like, tz: ZoneInfo):
    """
    Ensure datetime-like is tz-aware in tz.
    Works for: Series, DatetimeIndex, list/array of datetime-like.
    """
    x = pd.to_datetime(dt_like)

    if isinstance(x, pd.DatetimeIndex):
        if x.tz is None:
            return x.tz_localize(tz, ambiguous="infer", nonexistent="shift_forward")
        return x.tz_convert(tz)

    if isinstance(x, pd.Series):
        if x.dt.tz is None:
            return x.dt.tz_localize(tz, ambiguous="infer", nonexistent="shift_forward")
        return x.dt.tz_convert(tz)

    s = pd.Series(x)
    if s.dt.tz is None:
        return s.dt.tz_localize(tz, ambiguous="infer", nonexistent="shift_forward")
    return s.dt.tz_convert(tz)

def _localize_safely(naive_dt, tz: ZoneInfo):
    """
    Localize datetimes to tz while handling DST transitions.
    """
    s = pd.to_datetime(naive_dt)

    def _localize_idx(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
        if idx.tz is not None:
            return idx.tz_convert(tz)
        try:
            return idx.tz_localize(tz, ambiguous="infer", nonexistent="shift_forward")
        except Exception:
            return idx.tz_localize(tz, ambiguous=True, nonexistent="shift_forward")

    if isinstance(s, pd.DatetimeIndex):
        return _localize_idx(s)

    # Series path
    if getattr(s.dt, "tz", None) is not None:
        return s.dt.tz_convert(tz)

    try:
        return s.dt.tz_localize(tz, ambiguous="infer", nonexistent="shift_forward")
    except Exception:
        return s.dt.tz_localize(tz, ambiguous=True, nonexistent="shift_forward")

def tz_mismatch_message(project_tz: ZoneInfo, iso_name: str):
    iso_tz_name = ISO_TZ.get(iso_name)
    if iso_tz_name is None:
        return None

    iso_tz = ZoneInfo(iso_tz_name)

    if project_tz.key != iso_tz.key:
        return {
            "project_tz": project_tz.key,
            "iso_tz": iso_tz.key,
        }

    return None

# -----------------------------
# Wind physics helpers
# -----------------------------
def _wind_100m_from_80_120(df: pd.DataFrame,
                          v80_col: str = "wind_speed_80m_mps",
                          v120_col: str = "wind_speed_120m_mps",
                          alpha_default: float = 0.14,
                          alpha_min: float = 0.0,
                          alpha_max: float = 0.5) -> pd.Series:
    v80 = df[v80_col].astype(float)
    v120 = df[v120_col].astype(float)
    valid = (v80 > 0) & (v120 > 0)

    alpha = pd.Series(alpha_default, index=df.index, dtype=float)
    alpha.loc[valid] = np.log(v120[valid] / v80[valid]) / np.log(120.0 / 80.0)
    alpha = alpha.clip(alpha_min, alpha_max)

    v100 = v80 * (100.0 / 80.0) ** alpha
    missing_v100 = v100.isna() & (v120 > 0)
    v100.loc[missing_v100] = v120[missing_v100] * (100.0 / 120.0) ** alpha_default
    return v100

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
    """
    Fetch market prices from gridstatus for a given ISO.
    
    Note: Different ISOs have different API requirements:
    - CAISO, ISONE, MISO: Don't support location_type parameter
    - NYISO: Uses 'zone' or 'generator' instead of 'Hub'
    - PJM: Requires PJM_API_KEY environment variable
    - Ercot: Uses get_spp() instead of get_lmp(), limited historical data
    
    Args:
        iso_name: Name of the ISO (CAISO, Ercot, ISONE, MISO, NYISO, PJM)
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        market: Market type (DAY_AHEAD_HOURLY, REAL_TIME_HOURLY, REAL_TIME_5_MIN)
        location_type: Location type (varies by ISO)
        local_tz: Timezone for output timestamps
        
    Returns:
        DataFrame with timestamp and price_usd_mwh columns
        
    Raises:
        ValueError: If no data returned or ISO not supported
    """
    iso_class = getattr(gridstatus, iso_name, None)
    if iso_class is None:
        raise ValueError(f"Unsupported ISO class: {iso_name}")
    
    try:
        iso = iso_class()
    except Exception as e:
        raise ValueError(f"Failed to initialize {iso_name}: {e}")

    dfs = []
    errors = []
    
    for s, e in _date_chunks(start_date, end_date, freq="MS"):
        try:
            df_chunk = None
            
            if iso_name == "Ercot":
                # ERCOT uses get_spp() with different parameters
                # Note: Ercot's end parameter appears to be exclusive, so add 1 day
                # Also, DAM prices are published the day before, so extend range
                df_chunk = iso.get_spp(
                    date=pd.Timestamp(s) - pd.Timedelta(days=1),  # Start 1 day earlier to catch DAM prices
                    end=pd.Timestamp(e) + pd.Timedelta(days=1),   # End is exclusive, so add 1 day
                    market=market,
                    location_type="Trading Hub",
                )
            elif iso_name == "NYISO":
                # NYISO uses 'zone' instead of 'Hub'
                df_chunk = iso.get_lmp(
                    start=pd.Timestamp(s),
                    end=pd.Timestamp(e) + pd.Timedelta(days=1),
                    market=market,
                    location_type="zone",
                )
            elif iso_name in ("CAISO", "ISONE", "MISO"):
                # These ISOs don't support location_type parameter
                df_chunk = iso.get_lmp(
                    start=pd.Timestamp(s),
                    end=pd.Timestamp(e) + pd.Timedelta(days=1),
                    market=market,
                )
            elif iso_name == "PJM":
                # PJM requires API key - check signature for location_type support
                sig = inspect.signature(iso.get_lmp)
                kwargs = {
                    "start": pd.Timestamp(s),
                    "end": pd.Timestamp(e) + pd.Timedelta(days=1),
                    "market": market,
                }
                if "location_type" in sig.parameters:
                    kwargs["location_type"] = location_type
                df_chunk = iso.get_lmp(**kwargs)
            else:
                # Generic fallback - try with location_type first
                sig = inspect.signature(iso.get_lmp)
                kwargs = {
                    "start": pd.Timestamp(s),
                    "end": pd.Timestamp(e) + pd.Timedelta(days=1),
                    "market": market,
                }
                if "location_type" in sig.parameters:
                    kwargs["location_type"] = location_type
                df_chunk = iso.get_lmp(**kwargs)

            if df_chunk is not None and len(df_chunk) > 0:
                dfs.append(df_chunk)
                
        except Exception as chunk_error:
            errors.append(f"{s} to {e}: {str(chunk_error)[:100]}")
            continue

    if not dfs:
        error_details = "; ".join(errors) if errors else "No specific error"
        raise ValueError(
            f"No price data returned for {iso_name} between {start_date} and {end_date}. "
            f"Errors: {error_details}"
        )

    df = pd.concat(dfs, ignore_index=True)

    # Find timestamp column
    time_col = next((c for c in ["Time", "time", "timestamp", "Datetime", "DATETIME", "Interval Start"] if c in df.columns), None)
    if time_col is None:
        raise ValueError(f"Could not find a time column in LMP data. Columns: {list(df.columns)}")
    
    # Find price column
    price_col = next((c for c in ["LMP", "lmp", "price", "Price", "LBMP", "SPP"] if c in df.columns), None)
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
    
    # Filter to requested date range (some APIs return extra data)
    # Convert start/end dates to timezone-aware timestamps for comparison
    start_ts = pd.Timestamp(start_date).tz_localize(local_tz)
    end_ts = pd.Timestamp(end_date).tz_localize(local_tz) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    
    out = out[(out["timestamp"] >= start_ts) & (out["timestamp"] <= end_ts)]
    
    return out

class MarketPriceError(Exception):
    """Raised when market price data cannot be fetched."""
    pass


def add_prices_and_revenue(df: pd.DataFrame, project: dict, local_tz: ZoneInfo, include_prices: bool) -> pd.DataFrame:
    """
    Add price and revenue columns to wind data DataFrame.
    
    For market pricing, attempts to fetch prices from gridstatus API.
    If the fetch fails, returns DataFrame with NaN prices and a warning attribute.
    
    Args:
        df: DataFrame with wind/power data
        project: Project configuration dict
        local_tz: Local timezone
        include_prices: Whether to include price data
        
    Returns:
        DataFrame with price_usd_mwh and revenue_usd columns added
    """
    out = df.copy()
    out["timestamp"] = _ensure_tzaware(out["timestamp"], local_tz)
    
    # Initialize warning message attribute
    out.attrs["price_warning"] = None

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

    # Validate ISO name
    if not iso_name:
        out["price_usd_mwh"] = np.nan
        out["revenue_usd"] = np.nan
        out.attrs["price_warning"] = "No ISO specified for market pricing"
        return out

    start_date = str(out["timestamp"].min().date())
    end_date = str(out["timestamp"].max().date())

    try:
        prices = get_market_prices_gridstatus(
            iso_name=iso_name,
            start_date=start_date,
            end_date=end_date,
            market=market,
            location_type=location_type,
            local_tz=local_tz,
        )
    except ValueError as e:
        # Market price fetch failed - return with NaN prices and warning
        out["price_usd_mwh"] = np.nan
        out["revenue_usd"] = np.nan
        out.attrs["price_warning"] = f"Could not fetch market prices for {iso_name}: {str(e)[:200]}"
        return out
    except Exception as e:
        # Unexpected error - still return gracefully
        out["price_usd_mwh"] = np.nan
        out["revenue_usd"] = np.nan
        out.attrs["price_warning"] = f"Unexpected error fetching prices: {str(e)[:200]}"
        return out

    if prices is None or prices.empty:
        out["price_usd_mwh"] = np.nan
        out["revenue_usd"] = np.nan
        out.attrs["price_warning"] = f"No price data returned for {iso_name}"
        return out

    prices["timestamp"] = _ensure_tzaware(prices["timestamp"], local_tz)

    # Keep only the columns we need from prices to prevent collisions
    prices = prices[["timestamp", "price_usd_mwh"]].sort_values("timestamp")
    out = out.sort_values("timestamp")

    merged = pd.merge_asof(
        out,
        prices,
        on="timestamp",
        direction="backward",
    )

    # At this point, merged should have exactly one "price_usd_mwh"
    if "price_usd_mwh" not in merged.columns:
        candidates = [c for c in merged.columns if "price_usd_mwh" in c]
        raise KeyError(f"Expected 'price_usd_mwh' after merge. Found: {candidates}")

    merged["price_usd_mwh"] = merged["price_usd_mwh"].ffill()
    merged["revenue_usd"] = merged["energy_mwh"] * merged["price_usd_mwh"]
    
    # Preserve warning attribute
    merged.attrs["price_warning"] = out.attrs.get("price_warning")

    return merged


# -----------------------------
# Analytics (multi-year)
# -----------------------------
def monthly_capacity_factor(df: pd.DataFrame, rated_power_mw: float) -> pd.DataFrame:
    x = df.copy()
    x["month"] = x["timestamp"].dt.to_period("M").dt.to_timestamp()
    m = x.groupby("month", as_index=False)["power_mw"].mean()
    m["capacity_factor"] = m["power_mw"] / rated_power_mw
    return m

def seasonal_capacity_factor(df: pd.DataFrame, rated_power_mw: float) -> pd.DataFrame:
    x = df.copy()
    x["month_num"] = x["timestamp"].dt.month
    def season(m):
        if m in [12,1,2]: return "Winter"
        if m in [3,4,5]: return "Spring"
        if m in [6,7,8]: return "Summer"
        return "Fall"
    x["season"] = x["month_num"].map(season)
    s = x.groupby("season", as_index=False)["power_mw"].mean()
    s["capacity_factor"] = s["power_mw"] / rated_power_mw
    # keep nice order
    order = ["Winter","Spring","Summer","Fall"]
    s["season"] = pd.Categorical(s["season"], categories=order, ordered=True)
    return s.sort_values("season")

def exceedance_curve(df: pd.DataFrame, threshold_mps: float) -> pd.DataFrame:
    """
    Returns exceedance curve: % of hours wind_speed >= v
    """
    ws = df["wind_speed_mps"].dropna().to_numpy()
    if len(ws) == 0:
        return pd.DataFrame(columns=["wind_speed_mps", "pct_hours_ge"])

    grid = np.sort(np.unique(np.round(ws, 2)))
    pct = [(ws >= v).mean() * 100.0 for v in grid]
    out = pd.DataFrame({"wind_speed_mps": grid, "pct_hours_ge": pct})

    # Also include one marker row at requested threshold
    thr_pct = (ws >= threshold_mps).mean() * 100.0
    out.attrs["threshold_pct"] = thr_pct
    return out

def downtime_risk_by_month(df: pd.DataFrame, cut_in: float, cut_out: float) -> pd.DataFrame:
    x = df.copy()
    x["month"] = x["timestamp"].dt.month
    x["below_cut_in"] = x["wind_speed_mps"] < cut_in
    x["above_cut_out"] = x["wind_speed_mps"] > cut_out
    g = x.groupby("month", as_index=False).agg(
        pct_below_cut_in=("below_cut_in", "mean"),
        pct_above_cut_out=("above_cut_out", "mean"),
    )
    g["pct_below_cut_in"] *= 100
    g["pct_above_cut_out"] *= 100
    return g

def annual_energy_bootstrap(df_hist: pd.DataFrame, n_boot: int = 2000, seed: int = 7) -> dict:
    """
    Simple bootstrap on *annual* energy totals (MWh).
    Requires df_hist to contain multiple years of historical data.
    """
    x = df_hist.copy()
    x["year"] = x["timestamp"].dt.year
    annual = x.groupby("year", as_index=False)["energy_mwh"].sum()

    vals = annual["energy_mwh"].to_numpy()
    if len(vals) < 2:
        return {"years": len(vals), "p50": np.nan, "p90": np.nan, "p10": np.nan, "annual_table": annual}

    rng = np.random.default_rng(seed)
    samples = rng.choice(vals, size=(n_boot, len(vals)), replace=True).mean(axis=1)

    return {
        "years": len(vals),
        "p50": float(np.percentile(samples, 50)),
        "p90": float(np.percentile(samples, 90)),
        "p10": float(np.percentile(samples, 10)),
        "annual_table": annual.sort_values("year"),
    }

def interannual_variability(df_hist: pd.DataFrame) -> pd.DataFrame:
    x = df_hist.copy()
    x["year"] = x["timestamp"].dt.year
    a = x.groupby("year", as_index=False).agg(
        avg_wind=("wind_speed_mps", "mean"),
        avg_power=("power_mw", "mean"),
        total_energy=("energy_mwh", "sum")
    )
    a["energy_pct_vs_mean"] = 100.0 * (a["total_energy"] - a["total_energy"].mean()) / a["total_energy"].mean()
    return a.sort_values("year")


# -----------------------------
# ISO Price Functions
# -----------------------------

ISO_PRICES_CACHE_DIR = Path("data")
ISO_PRICES_CACHE_FILE = ISO_PRICES_CACHE_DIR / "iso_prices_2024.json"

ISO_DEFAULT_PRICES = {
    "CAISO": 32.50,
    "Ercot": 28.75,
    "ISONE": 45.20,
    "MISO": 25.10,
    "NYISO": 52.30,
    "PJM": 38.90,
}

def load_iso_prices_from_cache(year: int = 2024) -> dict:
    """Load cached ISO annual prices."""
    cache_file = ISO_PRICES_CACHE_DIR / f"iso_prices_{year}.json"
    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text())
        except Exception as e:
            print(f"Error loading ISO prices cache: {e}")
    return {}


def save_iso_prices_to_cache(prices: dict, year: int = 2024) -> bool:
    """Save ISO prices to cache."""
    try:
        ISO_PRICES_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file = ISO_PRICES_CACHE_DIR / f"iso_prices_{year}.json"
        cache_file.write_text(json.dumps(prices, indent=2))
        return True
    except Exception as e:
        print(f"Error saving ISO prices cache: {e}")
        return False


def get_iso_annual_price(
    iso_name: str,
    year: int = 2024,
    use_cache: bool = True,
    force_refresh: bool = False
) -> float:
    """
    Fetch annual average LMP price for an ISO.
    
    Uses cached prices if available, otherwise fetches from gridstatus API.
    Only 1 API call per ISO (not per location).
    
    Args:
        iso_name: ISO name (e.g., "CAISO", "PJM")
        year: Year to fetch prices for
        use_cache: Whether to read from cache file
        force_refresh: Force re-fetch from API even if cached
        
    Returns:
        Annual average price in $/MWh, or default price if unavailable
    """
    if iso_name not in ISO_BOUNDARIES:
        print(f"Warning: ISO {iso_name} not in ISO_BOUNDARIES")
        return 0.0
    
    if use_cache and not force_refresh:
        cached = load_iso_prices_from_cache(year)
        if iso_name in cached:
            return cached[iso_name]
    
    try:
        print(f"Fetching {iso_name} prices for {year}...")
        
        iso_class = getattr(gridstatus, iso_name, None)
        if iso_class is None:
            print(f"Warning: {iso_name} not available in gridstatus")
            return ISO_DEFAULT_PRICES.get(iso_name, 0.0)
        
        iso = iso_class()
        
        if iso_name == "Ercot":
            df_chunk = iso.get_spp(
                date=f"{year}-01-01",
                end=f"{year}-12-31",
                market="DAY_AHEAD_HOURLY",
                location_type="Trading Hub",
            )
            price_col = "SPP"
        else:
            df_chunk = iso.get_lmp(
                start=f"{year}-01-01",
                end=f"{year}-12-31",
                market="DAY_AHEAD_HOURLY",
                location_type="Hub",
            )
            price_col = next((c for c in ["LMP", "lmp", "price", "Price", "LBMP"] if c in df_chunk.columns), None)
        
        if df_chunk is None or df_chunk.empty:
            print(f"No price data returned for {iso_name}")
            return ISO_DEFAULT_PRICES.get(iso_name, 0.0)
        
        if price_col is None:
            print(f"Could not find price column in {iso_name} data")
            return ISO_DEFAULT_PRICES.get(iso_name, 0.0)
        
        avg_price = df_chunk[price_col].mean()
        price_usd_mwh = float(avg_price)
        
        print(f"  {iso_name} avg price: ${price_usd_mwh:.2f}/MWh")
        
        return price_usd_mwh
        
    except Exception as e:
        print(f"Error fetching {iso_name} prices: {e}")
        return ISO_DEFAULT_PRICES.get(iso_name, 0.0)


def get_all_iso_annual_prices(
    year: int = 2024,
    use_cache: bool = True,
    force_refresh: bool = False
) -> dict:
    """
    Fetch annual average prices for all supported ISOs.
    
    Args:
        year: Year to fetch prices for
        use_cache: Whether to use cached prices
        force_refresh: Force re-fetch from API
        
    Returns:
        Dictionary mapping ISO names to annual average prices
    """
    if use_cache and not force_refresh:
        cached = load_iso_prices_from_cache(year)
        if cached:
            return cached
    
    prices = {}
    for iso_name in ISO_BOUNDARIES.keys():
        prices[iso_name] = get_iso_annual_price(iso_name, year, use_cache=False, force_refresh=False)
    
    if use_cache and not force_refresh:
        save_iso_prices_to_cache(prices, year)
    
    return prices


def get_iso_for_location(lat: float, lon: float) -> str:
    """
    Determine which ISO a location belongs to.
    
    Args:
        lat: Latitude
        lon: Longitude
        
    Returns:
        ISO name string, or empty string if location is outside all ISOs
    """
    point = Point(lon, lat)
    
    for iso_name, polygon in ISO_BOUNDARIES.items():
        if polygon.contains(point):
            return iso_name
    
    return ""


def _find_nearest_iso(lat: float, lon: float) -> tuple[str, float]:
    """
    Find the nearest ISO region to a given point.
    
    Args:
        lat: Latitude
        lon: Longitude
        
    Returns:
        Tuple of (iso_name, distance_km). Returns ("", inf) if no ISOs defined.
    """
    point = Point(lon, lat)
    min_distance = float('inf')
    nearest_iso = ""
    
    for iso_name, polygon in ISO_BOUNDARIES.items():
        # distance() returns degrees; convert to approximate km (1 deg ~ 111 km at equator)
        dist_deg = polygon.distance(point)
        dist_km = dist_deg * 111.0  # rough approximation
        
        if dist_km < min_distance:
            min_distance = dist_km
            nearest_iso = iso_name
    
    return nearest_iso, min_distance


def _is_offshore_location(lat: float, lon: float) -> bool:
    """
    Determine if a location is likely offshore (over water).
    
    Uses geographic heuristics for US coastlines. Points are considered
    offshore if they fall within known offshore wind development zones.
    
    Args:
        lat: Latitude
        lon: Longitude
        
    Returns:
        True if location appears to be offshore
    """
    # Check if inside any ISO first - if so, definitely not offshore
    point = Point(lon, lat)
    for polygon in ISO_BOUNDARIES.values():
        if polygon.contains(point):
            return False
    
    # Pacific offshore: west of California/Oregon/Washington coast
    # Approximate coastline follows roughly -124.5 at north to -117.5 at south
    if lon < -117.0 and 32.0 < lat < 49.0:
        # More refined check: coast curves inward
        if lat > 42.0 and lon < -124.0:  # Oregon/Washington
            return True
        if 37.0 < lat <= 42.0 and lon < -123.5:  # Northern California
            return True
        if 34.0 < lat <= 37.0 and lon < -121.5:  # Central California
            return True
        if lat <= 34.0 and lon < -118.5:  # Southern California
            return True
    
    # Atlantic offshore: east of the East Coast
    # New England (Maine to Cape Cod): coast around -70 to -66
    if 41.0 < lat < 45.0 and lon > -71.0:
        return True
    
    # Mid-Atlantic (NY to VA): coast around -74 to -72
    if 36.5 < lat <= 41.0 and lon > -74.5:
        # Eastern edge of land is roughly -74 at NJ, -75 at VA
        coast_lon = -74.0 + (lat - 40.0) * -0.25  # approximate coastline
        if lon > coast_lon:
            return True
    
    # Southeast (NC to FL): coast roughly -75.5 to -80
    if 25.0 < lat <= 36.5 and lon > -81.0:
        # Florida curves west; Carolina coast is around -75 to -76
        if lat > 32.0 and lon > -76.0:  # Carolinas
            return True
        if lat <= 32.0 and lon > -80.0:  # Georgia/Florida
            return True
    
    # Gulf of Mexico offshore
    if -98.0 < lon < -82.0 and 24.0 < lat < 30.0:
        # South of Texas/Louisiana coast
        if lat < 29.0:  # Clearly in the Gulf
            return True
        # Near the coast - use approximate coastline
        if lon < -90.0 and lat < 29.5:  # Louisiana/Texas
            return True
    
    return False


# Offshore wind premium (percentage above nearest ISO price)
OFFSHORE_PREMIUM_PERCENT = 15.0  # 15% premium for offshore wind

# Maximum distance (km) from an ISO boundary to still use nearest-ISO pricing
# Beyond this, location is considered too remote for reliable pricing
# Set high enough to include Pacific NW (Seattle ~620km from CAISO)
MAX_NEAREST_ISO_DISTANCE_KM = 750.0


def get_price_for_location(
    lat: float,
    lon: float,
    iso_prices: dict,
    apply_offshore_premium: bool = True,
    offshore_premium_pct: float = OFFSHORE_PREMIUM_PERCENT,
    max_fallback_distance_km: float = MAX_NEAREST_ISO_DISTANCE_KM
) -> tuple[float, str, dict]:
    """
    Get the appropriate price for a location based on its ISO region.
    
    Implements hybrid handling:
    1. If inside an ISO polygon -> use that ISO's price
    2. If offshore near an ISO -> use nearest ISO price + offshore premium
    3. If onshore outside ISOs but near one -> use nearest ISO price (fallback)
    4. If too far from any ISO -> return 0 (no reliable price available)
    
    Args:
        lat: Latitude
        lon: Longitude
        iso_prices: Dictionary mapping ISO names to prices
        apply_offshore_premium: Whether to add premium for offshore locations
        offshore_premium_pct: Premium percentage for offshore (default 15%)
        max_fallback_distance_km: Max distance for nearest-ISO fallback
        
    Returns:
        Tuple of (price_usd_mwh, pricing_type, metadata)
        pricing_type: "iso", "offshore", "fallback", or "none"
        metadata: Additional info (iso_name, distance_km, etc.)
    """
    # First, check if inside any ISO
    iso_name = get_iso_for_location(lat, lon)
    
    if iso_name and iso_name in iso_prices:
        return (
            iso_prices[iso_name],
            "iso",
            {"iso_name": iso_name, "distance_km": 0.0}
        )
    
    # Not inside any ISO - find nearest
    nearest_iso, distance_km = _find_nearest_iso(lat, lon)
    
    if not nearest_iso or nearest_iso not in iso_prices:
        return (0.0, "none", {"reason": "no_iso_found"})
    
    # Check if too far from any ISO
    if distance_km > max_fallback_distance_km:
        return (
            0.0,
            "none",
            {"reason": "too_far", "nearest_iso": nearest_iso, "distance_km": distance_km}
        )
    
    base_price = iso_prices[nearest_iso]
    
    # Check if offshore
    if _is_offshore_location(lat, lon):
        if apply_offshore_premium:
            premium_multiplier = 1.0 + (offshore_premium_pct / 100.0)
            offshore_price = base_price * premium_multiplier
            return (
                offshore_price,
                "offshore",
                {
                    "nearest_iso": nearest_iso,
                    "base_price": base_price,
                    "premium_pct": offshore_premium_pct,
                    "distance_km": distance_km
                }
            )
        else:
            return (
                base_price,
                "offshore",
                {"nearest_iso": nearest_iso, "distance_km": distance_km}
            )
    
    # Onshore but outside ISO polygons - use nearest ISO (fallback)
    return (
        base_price,
        "fallback",
        {"nearest_iso": nearest_iso, "distance_km": distance_km}
    )
