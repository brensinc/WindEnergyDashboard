#!/usr/bin/env python3
"""
Script to generate cached hourly ISO price data.
Uses realistic synthetic data based on historical averages and patterns.
"""

import sys
from pathlib import Path
from datetime import datetime
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

# Configuration
CACHE_DIR = Path(__file__).parent.parent / "data" / "hourly_prices"
YEAR = 2024
ISO_LIST = ["CAISO", "MISO", "PJM", "NYISO", "ISONE", "Ercot"]

# Historical average prices and characteristics by ISO
ISO_CHARACTERISTICS = {
    "CAISO": {
        "base_price": 45.0,
        "volatility": 0.25,
        "peak_premium": 0.35,  # California has high peak prices
        "summer_premium": 0.20,  # Summer AC load
        "tz": "America/Los_Angeles",
    },
    "Ercot": {
        "base_price": 32.0,
        "volatility": 0.40,  # ERCOT is very volatile
        "peak_premium": 0.30,
        "summer_premium": 0.35,  # Texas summer heat
        "tz": "America/Chicago",
    },
    "ISONE": {
        "base_price": 48.0,
        "volatility": 0.20,
        "peak_premium": 0.25,
        "winter_premium": 0.25,  # New England winter heating
        "tz": "America/New_York",
    },
    "MISO": {
        "base_price": 28.0,
        "volatility": 0.20,
        "peak_premium": 0.20,
        "summer_premium": 0.15,
        "tz": "America/Chicago",
    },
    "NYISO": {
        "base_price": 55.0,
        "volatility": 0.22,
        "peak_premium": 0.30,
        "summer_premium": 0.20,
        "winter_premium": 0.15,
        "tz": "America/New_York",
    },
    "PJM": {
        "base_price": 38.0,
        "volatility": 0.18,
        "peak_premium": 0.22,
        "summer_premium": 0.18,
        "tz": "America/New_York",
    },
}


def generate_hourly_prices(iso_name: str, year: int) -> pd.DataFrame:
    """
    Generate realistic hourly prices for an ISO.
    Incorporates: daily patterns, seasonal patterns, weekday effects, and random volatility.
    """
    config = ISO_CHARACTERISTICS.get(iso_name, {
        "base_price": 35.0,
        "volatility": 0.20,
        "peak_premium": 0.25,
        "tz": "UTC",
    })
    
    from zoneinfo import ZoneInfo
    tz = ZoneInfo(config["tz"])
    
    # Create hourly timestamps for the year
    start = pd.Timestamp(f"{year}-01-01", tz=tz)
    end = pd.Timestamp(f"{year}-12-31 23:00", tz=tz)
    timestamps = pd.date_range(start=start, end=end, freq="h")
    
    n_hours = len(timestamps)
    np.random.seed(hash(iso_name) % 2**32)  # Reproducible per ISO
    
    # Base price
    base = config["base_price"]
    prices = np.full(n_hours, base, dtype=float)
    
    # Extract time components
    hours = np.array([t.hour for t in timestamps])
    months = np.array([t.month for t in timestamps])
    weekdays = np.array([t.weekday() for t in timestamps])  # 0=Mon, 6=Sun
    day_of_year = np.array([t.dayofyear for t in timestamps])
    
    # 1. Daily pattern (time-of-use)
    # Off-peak: 10pm - 6am (lower prices)
    # Mid-peak: 6am-2pm, 9pm-10pm
    # On-peak: 2pm - 9pm (highest prices)
    daily_factor = np.ones(n_hours)
    
    # Off-peak hours (22-6)
    off_peak = (hours >= 22) | (hours < 6)
    daily_factor[off_peak] = 0.7
    
    # Mid-peak hours (6-14, 21-22)
    mid_peak = ((hours >= 6) & (hours < 14)) | (hours == 21)
    daily_factor[mid_peak] = 0.95
    
    # On-peak hours (14-21)
    on_peak = (hours >= 14) & (hours < 21)
    daily_factor[on_peak] = 1.0 + config["peak_premium"]
    
    # Super peak (16-20)
    super_peak = (hours >= 16) & (hours < 20)
    daily_factor[super_peak] = 1.0 + config["peak_premium"] * 1.5
    
    prices *= daily_factor
    
    # 2. Seasonal pattern
    seasonal_factor = np.ones(n_hours)
    
    # Summer (Jun-Sep) - months 6,7,8,9
    summer = (months >= 6) & (months <= 9)
    summer_premium = config.get("summer_premium", 0.15)
    seasonal_factor[summer] = 1.0 + summer_premium
    
    # Peak summer (Jul-Aug)
    peak_summer = (months >= 7) & (months <= 8)
    seasonal_factor[peak_summer] = 1.0 + summer_premium * 1.3
    
    # Winter (Dec-Feb) - months 12,1,2
    winter = (months == 12) | (months <= 2)
    winter_premium = config.get("winter_premium", 0.10)
    seasonal_factor[winter] = 1.0 + winter_premium
    
    # Shoulder seasons (Mar-May, Oct-Nov) - lower prices
    shoulder = ((months >= 3) & (months <= 5)) | ((months >= 10) & (months <= 11))
    seasonal_factor[shoulder] = 0.90
    
    prices *= seasonal_factor
    
    # 3. Weekday/weekend effect
    weekend = weekdays >= 5
    prices[weekend] *= 0.85  # Lower weekend prices
    
    # 4. Random volatility (hourly noise)
    volatility = config["volatility"]
    noise = np.random.lognormal(0, volatility * 0.5, n_hours)
    prices *= noise
    
    # 5. Occasional price spikes (0.5% of hours)
    spike_prob = 0.005
    spikes = np.random.random(n_hours) < spike_prob
    spike_multiplier = np.random.uniform(2.0, 6.0, spikes.sum())
    prices[spikes] *= spike_multiplier
    
    # 6. Occasional negative/very low prices (0.2% - common in high-renewables markets)
    if iso_name in ["CAISO", "Ercot", "MISO"]:  # Markets with high renewables
        low_prob = 0.002
        low_hours = np.random.random(n_hours) < low_prob
        prices[low_hours] = np.random.uniform(-5, 10, low_hours.sum())
    
    # Ensure reasonable bounds
    prices = np.clip(prices, -10, 500)  # Allow some negative, cap at 500
    
    df = pd.DataFrame({
        "timestamp": timestamps,
        "price_usd_mwh": prices.round(2)
    })
    
    return df


def cache_prices_for_iso(iso_name: str, year: int) -> dict:
    """Generate and cache prices for a single ISO."""
    print(f"Generating {iso_name} prices for {year}...")
    
    df = generate_hourly_prices(iso_name, year)
    
    # Save to cache
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{iso_name}_{year}_hourly.parquet"
    
    # Convert timestamp to UTC for storage
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_convert("UTC")
    df.to_parquet(cache_file, index=False)
    
    # Calculate stats
    stats = {
        "iso": iso_name,
        "year": year,
        "records": len(df),
        "min_price": float(df["price_usd_mwh"].min()),
        "max_price": float(df["price_usd_mwh"].max()),
        "mean_price": float(df["price_usd_mwh"].mean()),
        "median_price": float(df["price_usd_mwh"].median()),
        "std_price": float(df["price_usd_mwh"].std()),
        "negative_hours": int((df["price_usd_mwh"] < 0).sum()),
        "generated_at": datetime.now().isoformat(),
    }
    
    stats_file = CACHE_DIR / f"{iso_name}_{year}_stats.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"  Saved {len(df)} records (mean: ${stats['mean_price']:.2f}/MWh)")
    
    return stats


def main():
    print("=" * 60)
    print(f"Generating hourly ISO prices for {YEAR}")
    print("=" * 60)
    
    all_stats = {}
    for iso_name in ISO_LIST:
        try:
            stats = cache_prices_for_iso(iso_name, YEAR)
            all_stats[iso_name] = stats
        except Exception as e:
            print(f"  Failed: {e}")
    
    # Save combined stats
    combined_file = CACHE_DIR / f"all_isos_{YEAR}_summary.json"
    with open(combined_file, "w") as f:
        json.dump(all_stats, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Cache complete!")
    print(f"Files saved to: {CACHE_DIR}")
    print("=" * 60)
    
    # Print summary table
    print("\nSummary:")
    print(f"{'ISO':<10} {'Mean':>10} {'Min':>10} {'Max':>10} {'Std':>10}")
    print("-" * 50)
    for iso, s in all_stats.items():
        print(f"{iso:<10} ${s['mean_price']:>8.2f} ${s['min_price']:>8.2f} ${s['max_price']:>8.2f} ${s['std_price']:>8.2f}")


if __name__ == "__main__":
    main()
