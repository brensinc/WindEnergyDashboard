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
