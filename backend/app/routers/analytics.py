from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from datetime import date, datetime
from typing import Optional, Literal
import sys
import os
import json
from pathlib import Path

# Add parent directory to path to import windlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

router = APIRouter(prefix="/api/analytics", tags=["analytics"])

# Path to cached hourly prices
HOURLY_PRICES_DIR = Path(__file__).parent.parent.parent.parent / "data" / "hourly_prices"


class AnalyticsRequest(BaseModel):
    project_id: str
    latitude: float
    longitude: float
    hub_height_m: int = 100
    rotor_diameter_m: int = 100
    rated_power_kw: int = 3000
    start_date: str
    end_date: str
    # Pricing options
    pricing_mode: Literal["market", "fixed"] = "market"
    fixed_price: float = 50.0  # $/MWh for fixed pricing
    iso_override: Optional[str] = None  # Override auto-detected ISO


class TimeSeriesData(BaseModel):
    timestamps: list[str]
    wind_speeds: list[float]
    power_outputs: list[float]  # in kW
    energy_outputs: list[float]  # in kWh (hourly)
    prices: list[float]  # $/MWh
    revenues: list[float]  # $ per hour


class MonthlyStats(BaseModel):
    months: list[str]
    energy_mwh: list[float]
    revenue: list[float]
    capacity_factors: list[float]
    avg_wind_speed: list[float]
    avg_price: list[float]


class HourlyPatternData(BaseModel):
    hours: list[int]  # 0-23
    avg_power_by_hour: list[float]
    avg_wind_by_hour: list[float]
    avg_price_by_hour: list[float]


class WindDistributionData(BaseModel):
    bins: list[float]  # wind speed bins
    frequencies: list[float]  # % of time in each bin
    power_contribution: list[float]  # % of total energy from each bin


class SeasonalData(BaseModel):
    seasons: list[str]
    energy_mwh: list[float]
    capacity_factors: list[float]
    avg_wind_speed: list[float]


class SummaryStats(BaseModel):
    total_energy_mwh: float
    total_revenue: float
    capacity_factor: float
    avg_wind_speed: float
    avg_price: float
    peak_power_kw: float
    hours_at_rated: int
    hours_below_cutin: int
    iso_region: Optional[str]
    pricing_mode: str
    data_period_days: int


class AnalyticsResponse(BaseModel):
    summary: SummaryStats
    timeseries: TimeSeriesData
    monthly: MonthlyStats
    hourly_pattern: HourlyPatternData
    wind_distribution: WindDistributionData
    seasonal: SeasonalData


def load_hourly_prices(iso_name: str, year: int = 2024):
    """Load cached hourly prices for an ISO."""
    import pandas as pd
    
    cache_file = HOURLY_PRICES_DIR / f"{iso_name}_{year}_hourly.parquet"
    
    if not cache_file.exists():
        # Try alternate capitalizations
        for alt_name in [iso_name.upper(), iso_name.lower(), iso_name.capitalize()]:
            alt_file = HOURLY_PRICES_DIR / f"{alt_name}_{year}_hourly.parquet"
            if alt_file.exists():
                cache_file = alt_file
                break
    
    if not cache_file.exists():
        return None
    
    df = pd.read_parquet(cache_file)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def load_iso_annual_prices() -> dict:
    """Load cached ISO annual average prices from data file."""
    data_path = Path(__file__).parent.parent.parent.parent / "data" / "iso_prices_2024.json"
    if data_path.exists():
        return json.loads(data_path.read_text())
    return {
        "CAISO": 45.0,
        "Ercot": 32.0,
        "ISONE": 48.0,
        "MISO": 28.0,
        "NYISO": 55.0,
        "PJM": 38.0,
    }


@router.get("/isos")
async def get_available_isos():
    """Get list of ISOs with available price data."""
    isos = []
    if HOURLY_PRICES_DIR.exists():
        for f in HOURLY_PRICES_DIR.glob("*_2024_hourly.parquet"):
            iso_name = f.stem.replace("_2024_hourly", "")
            # Load stats if available
            stats_file = HOURLY_PRICES_DIR / f"{iso_name}_2024_stats.json"
            if stats_file.exists():
                stats = json.loads(stats_file.read_text())
                isos.append({
                    "name": iso_name,
                    "mean_price": stats.get("mean_price", 0),
                    "min_price": stats.get("min_price", 0),
                    "max_price": stats.get("max_price", 0),
                })
            else:
                isos.append({"name": iso_name, "mean_price": 0, "min_price": 0, "max_price": 0})
    return isos


@router.post("", response_model=AnalyticsResponse)
async def get_project_analytics(request: AnalyticsRequest):
    """
    Generate comprehensive analytics for a project.
    Fetches weather data and combines with ISO prices to compute
    power output, revenue, and various statistical breakdowns.
    
    Supports both market pricing (hourly ISO prices) and fixed pricing.
    """
    try:
        import windlib
        import numpy as np
        import pandas as pd
        
        # Parse dates
        start = date.fromisoformat(request.start_date)
        end = date.fromisoformat(request.end_date)
        
        # Fetch wind data from Open-Meteo
        df = windlib.get_wind_data(
            lat=request.latitude,
            lon=request.longitude,
            date_start=str(start),
            date_end=str(end),
        )
        
        if df.empty:
            raise HTTPException(
                status_code=404, 
                detail="No wind data available for this location/date range"
            )
        
        # Determine ISO region
        if request.iso_override:
            iso_region = request.iso_override
        else:
            iso_region = windlib.get_iso_for_location(request.latitude, request.longitude)
            if not iso_region:
                # Try to find nearest ISO
                iso_prices = load_iso_annual_prices()
                _, _, price_metadata = windlib.get_price_for_location(
                    request.latitude, request.longitude, iso_prices
                )
                iso_region = price_metadata.get("nearest_iso", "MISO")  # Default to MISO
        
        # Calculate power output using windlib
        rated_power_mw = request.rated_power_kw / 1000.0
        
        # Standard turbine parameters
        cut_in_mps = 3.0
        rated_speed_mps = 12.0
        cut_out_mps = 25.0
        
        # Calculate power for each hour
        df["power_mw"] = windlib.wind_to_power(
            df["wind_speed_mps"].values,
            cut_in_mps=cut_in_mps,
            rated_speed_mps=rated_speed_mps,
            cut_out_mps=cut_out_mps,
            rated_power_mw=rated_power_mw,
        )
        df["power_kw"] = df["power_mw"] * 1000
        df["energy_kwh"] = df["power_kw"] * 1.0  # 1 hour intervals
        df["energy_mwh"] = df["power_mw"] * 1.0
        
        # Ensure timestamp is datetime and timezone-aware
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Apply pricing based on mode
        if request.pricing_mode == "fixed":
            # Fixed price for all hours
            df["price_usd_mwh"] = request.fixed_price
        else:
            # Load hourly market prices
            prices_df = load_hourly_prices(iso_region)
            
            if prices_df is not None and not prices_df.empty:
                # Convert both to UTC for matching
                if df["timestamp"].dt.tz is not None:
                    df["timestamp_utc"] = df["timestamp"].dt.tz_convert("UTC")
                else:
                    df["timestamp_utc"] = df["timestamp"]
                
                if prices_df["timestamp"].dt.tz is not None:
                    prices_df["timestamp_utc"] = prices_df["timestamp"].dt.tz_convert("UTC")
                else:
                    prices_df["timestamp_utc"] = prices_df["timestamp"].dt.tz_localize("UTC")
                
                # Merge prices with wind data
                prices_df = prices_df[["timestamp_utc", "price_usd_mwh"]].drop_duplicates(subset=["timestamp_utc"])
                df = df.sort_values("timestamp_utc")
                prices_df = prices_df.sort_values("timestamp_utc")
                
                df = pd.merge_asof(
                    df,
                    prices_df,
                    on="timestamp_utc",
                    direction="nearest",
                    tolerance=pd.Timedelta("2h")
                )
                
                # Fill any missing prices with ISO average
                iso_prices = load_iso_annual_prices()
                default_price = iso_prices.get(iso_region, 35.0)
                df["price_usd_mwh"] = df["price_usd_mwh"].fillna(default_price)
                
                # Drop temp column
                df = df.drop(columns=["timestamp_utc"])
            else:
                # Fallback to fixed price if no hourly data
                iso_prices = load_iso_annual_prices()
                df["price_usd_mwh"] = iso_prices.get(iso_region, 35.0)
        
        # Calculate revenue
        df["revenue_usd"] = df["energy_mwh"] * df["price_usd_mwh"]
        
        # ============ BUILD RESPONSE ============
        
        # --- Summary Stats ---
        total_energy_mwh = float(df["energy_mwh"].sum())
        total_revenue = float(df["revenue_usd"].sum())
        capacity_factor = float(df["power_mw"].mean() / rated_power_mw) if rated_power_mw > 0 else 0
        avg_wind_speed = float(df["wind_speed_mps"].mean())
        avg_price = float(df["price_usd_mwh"].mean())
        peak_power_kw = float(df["power_kw"].max())
        hours_at_rated = int((df["power_mw"] >= rated_power_mw * 0.99).sum())
        hours_below_cutin = int((df["wind_speed_mps"] < cut_in_mps).sum())
        data_period_days = (df["timestamp"].max() - df["timestamp"].min()).days + 1
        
        summary = SummaryStats(
            total_energy_mwh=round(total_energy_mwh, 2),
            total_revenue=round(total_revenue, 2),
            capacity_factor=round(capacity_factor, 4),
            avg_wind_speed=round(avg_wind_speed, 2),
            avg_price=round(avg_price, 2),
            peak_power_kw=round(peak_power_kw, 2),
            hours_at_rated=hours_at_rated,
            hours_below_cutin=hours_below_cutin,
            iso_region=iso_region,
            pricing_mode=request.pricing_mode,
            data_period_days=data_period_days,
        )
        
        # --- Time Series (downsample for large datasets) ---
        # If more than 2000 hours, resample to daily
        if len(df) > 2000:
            df_ts = df.set_index("timestamp").resample("D").agg({
                "wind_speed_mps": "mean",
                "power_kw": "mean",
                "energy_kwh": "sum",
                "price_usd_mwh": "mean",
                "revenue_usd": "sum",
            }).reset_index()
        else:
            df_ts = df.copy()
        
        timeseries = TimeSeriesData(
            timestamps=[t.isoformat() for t in df_ts["timestamp"]],
            wind_speeds=df_ts["wind_speed_mps"].round(2).tolist(),
            power_outputs=df_ts["power_kw"].round(2).tolist(),
            energy_outputs=df_ts["energy_kwh"].round(2).tolist(),
            prices=df_ts["price_usd_mwh"].round(2).tolist(),
            revenues=df_ts["revenue_usd"].round(2).tolist(),
        )
        
        # --- Monthly Stats ---
        df["month"] = df["timestamp"].dt.to_period("M")
        monthly_agg = df.groupby("month").agg({
            "energy_mwh": "sum",
            "revenue_usd": "sum",
            "power_mw": "mean",
            "wind_speed_mps": "mean",
            "price_usd_mwh": "mean",
        }).reset_index()
        monthly_agg["capacity_factor"] = monthly_agg["power_mw"] / rated_power_mw
        
        monthly = MonthlyStats(
            months=[str(m) for m in monthly_agg["month"]],
            energy_mwh=monthly_agg["energy_mwh"].round(2).tolist(),
            revenue=monthly_agg["revenue_usd"].round(2).tolist(),
            capacity_factors=monthly_agg["capacity_factor"].round(4).tolist(),
            avg_wind_speed=monthly_agg["wind_speed_mps"].round(2).tolist(),
            avg_price=monthly_agg["price_usd_mwh"].round(2).tolist(),
        )
        
        # --- Hourly Pattern (average by hour of day) ---
        df["hour"] = df["timestamp"].dt.hour
        hourly_agg = df.groupby("hour").agg({
            "power_kw": "mean",
            "wind_speed_mps": "mean",
            "price_usd_mwh": "mean",
        }).reset_index()
        
        hourly_pattern = HourlyPatternData(
            hours=hourly_agg["hour"].tolist(),
            avg_power_by_hour=hourly_agg["power_kw"].round(2).tolist(),
            avg_wind_by_hour=hourly_agg["wind_speed_mps"].round(2).tolist(),
            avg_price_by_hour=hourly_agg["price_usd_mwh"].round(2).tolist(),
        )
        
        # --- Wind Distribution ---
        wind_speeds = df["wind_speed_mps"].dropna()
        bins = list(range(0, 26, 2))  # 0, 2, 4, ..., 24 m/s
        hist, bin_edges = np.histogram(wind_speeds, bins=bins)
        frequencies = (hist / len(wind_speeds) * 100).tolist()
        
        # Power contribution per wind speed bin
        power_contributions = []
        total_energy = df["energy_mwh"].sum()
        for i in range(len(bins) - 1):
            mask = (df["wind_speed_mps"] >= bins[i]) & (df["wind_speed_mps"] < bins[i + 1])
            bin_energy = df.loc[mask, "energy_mwh"].sum()
            pct = (bin_energy / total_energy * 100) if total_energy > 0 else 0
            power_contributions.append(round(pct, 2))
        
        wind_distribution = WindDistributionData(
            bins=[float(b) for b in bins[:-1]],
            frequencies=[round(f, 2) for f in frequencies],
            power_contribution=power_contributions,
        )
        
        # --- Seasonal Data ---
        def get_season(month):
            if month in [12, 1, 2]:
                return "Winter"
            elif month in [3, 4, 5]:
                return "Spring"
            elif month in [6, 7, 8]:
                return "Summer"
            else:
                return "Fall"
        
        df["season"] = df["timestamp"].dt.month.map(get_season)
        seasonal_agg = df.groupby("season").agg({
            "energy_mwh": "sum",
            "power_mw": "mean",
            "wind_speed_mps": "mean",
        }).reset_index()
        seasonal_agg["capacity_factor"] = seasonal_agg["power_mw"] / rated_power_mw
        
        # Ensure consistent order
        season_order = ["Winter", "Spring", "Summer", "Fall"]
        seasonal_agg["season"] = pd.Categorical(
            seasonal_agg["season"], 
            categories=season_order, 
            ordered=True
        )
        seasonal_agg = seasonal_agg.sort_values("season")
        
        seasonal = SeasonalData(
            seasons=seasonal_agg["season"].tolist(),
            energy_mwh=seasonal_agg["energy_mwh"].round(2).tolist(),
            capacity_factors=seasonal_agg["capacity_factor"].round(4).tolist(),
            avg_wind_speed=seasonal_agg["wind_speed_mps"].round(2).tolist(),
        )
        
        return AnalyticsResponse(
            summary=summary,
            timeseries=timeseries,
            monthly=monthly,
            hourly_pattern=hourly_pattern,
            wind_distribution=wind_distribution,
            seasonal=seasonal,
        )
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error generating analytics: {str(e)}")


@router.get("/iso-prices")
async def get_iso_prices():
    """Get cached ISO annual average prices."""
    return load_iso_annual_prices()
