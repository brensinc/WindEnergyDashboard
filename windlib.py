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
    # Compare to typical year
    a["energy_pct_vs_mean"] = 100.0 * (a["total_energy"] - a["total_energy"].mean()) / a["total_energy"].mean()
    return a.sort_values("year")
