#!/usr/bin/env python3
"""
One-time script to pre-generate heatmap data for the wind energy dashboard.

This script fetches wind data for a grid of sample points across the continental US,
calculates annual revenue for each point, and creates an interpolated heatmap that
is cached for use by the dashboard.

Usage:
    python generate_heatmap.py [--year YEAR] [--density DENSITY] [--verbose]

Arguments:
    --year YEAR      Year of data to generate (default: 2024)
    --density N      Number of sample points per axis (default: 15, total = N^2)
    --verbose        Print detailed progress information

Example:
    python generate_heatmap.py --year 2024 --density 15 --verbose

Notes:
    - This script takes 15-30 minutes to run due to API rate limits
    - Results are cached in data/heatmap_cache_{year}.joblib
    - Run this once before shipping the app
"""

import argparse
import sys
import time
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from heatmap_utils import (
    INDUSTRY_STANDARD_TURBINE,
    DEFAULT_PRICE_USD_MWH,
    US_BOUNDS,
    create_sample_grid,
    interpolate_revenue_data,
    calculate_data_availability_mask,
    create_heatmap_data_structure,
    save_heatmap_cache,
    load_us_boundary_with_buffer,
    create_us_mask,
)

from windlib import (
    get_wind_data,
    add_data_type,
    add_power_output,
    get_tz_from_latlon,
    get_iso_for_location,
    get_all_iso_annual_prices,
    get_price_for_location,
)


def calculate_annual_revenue_for_point(
    lat: float,
    lon: float,
    year: int,
    turbine_params: Dict,
    iso_prices: Dict[str, float],
    verbose: bool = False
) -> float:
    """
    Calculate annual revenue for a single location using ISO-specific prices.
    
    Args:
        lat: Latitude
        lon: Longitude
        year: Year of data
        turbine_params: Turbine parameters (cut_in_mps, rated_speed_mps, etc.)
        iso_prices: Dictionary mapping ISO names to annual average prices ($/MWh)
        verbose: Print progress info
        
    Returns:
        Annual revenue in USD, or 0 if data unavailable / outside ISOs
    """
    try:
        # Fetch wind data for the entire year
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"
        
        df = get_wind_data(lat, lon, start_date, end_date)
        
        if df is None or df.empty:
            if verbose:
                print(f"  No data for ({lat:.2f}, {lon:.2f})")
            return 0.0
        
        # Get local timezone
        local_tz = get_tz_from_latlon(lat, lon)
        
        # Add data type (historical vs forecast)
        df = add_data_type(df, local_tz)
        
        # Filter to historical data only (no forecasts in annual calculation)
        df_hist = df[df["data_type"] == "historical"].copy()
        
        if df_hist.empty:
            if verbose:
                print(f"  No historical data for ({lat:.2f}, {lon:.2f})")
            return 0.0
        
        # Calculate power output
        df_hist = add_power_output(df_hist, turbine_params)
        
        # Calculate energy (power_mw * 1 hour = energy_mwh)
        total_energy_mwh = df_hist["energy_mwh"].sum()
        
        # Get price using hybrid ISO/offshore/fallback logic
        price_usd_mwh, pricing_type, pricing_meta = get_price_for_location(lat, lon, iso_prices)
        
        if pricing_type == "none":
            if verbose:
                reason = pricing_meta.get("reason", "unknown")
                print(f"  ({lat:.2f}, {lon:.2f}): No pricing available ({reason})")
            return 0.0
        
        # Calculate revenue
        annual_revenue = total_energy_mwh * price_usd_mwh
        
        if verbose:
            hours = len(df_hist)
            capacity_factor = df_hist["power_mw"].mean() / turbine_params["rated_power_mw"]
            
            # Build descriptive label for pricing type
            if pricing_type == "iso":
                iso_label = pricing_meta.get("iso_name", "?")
                price_label = f"[{iso_label}]"
            elif pricing_type == "offshore":
                iso_label = pricing_meta.get("nearest_iso", "?")
                premium = pricing_meta.get("premium_pct", 0)
                price_label = f"[OFFSHORE near {iso_label} +{premium:.0f}%]"
            elif pricing_type == "fallback":
                iso_label = pricing_meta.get("nearest_iso", "?")
                dist_km = pricing_meta.get("distance_km", 0)
                price_label = f"[FALLBACK: {iso_label} @ {dist_km:.0f}km]"
            else:
                price_label = "[?]"
            
            print(f"  ({lat:.2f}, {lon:.2f}) {price_label}: {hours} hours, CF={capacity_factor:.2%}, ${price_usd_mwh:.2f}/MWh, Revenue=${annual_revenue:,.0f}")
        
        return annual_revenue
        
    except Exception as e:
        if verbose:
            print(f"  Error at ({lat:.2f}, {lon:.2f}): {e}")
        return 0.0


def generate_heatmap_data(
    year: int = 2024,
    density: int = 15,
    turbine_params: Dict = None,
    iso_prices: Dict[str, float] = None,
    verbose: bool = False
) -> Dict:
    """
    Generate complete heatmap data for the continental US using ISO-specific prices.
    
    Args:
        year: Year of data to use
        density: Number of sample points per axis
        turbine_params: Turbine parameters (uses industry standard if None)
        iso_prices: Dictionary mapping ISO names to prices ($/MWh). If None, loads from cache/API.
        verbose: Print progress information
        
    Returns:
        Dictionary containing complete heatmap data
    """
    if turbine_params is None:
        turbine_params = INDUSTRY_STANDARD_TURBINE.copy()
    
    if iso_prices is None:
        print("\nLoading ISO annual prices...")
        iso_prices = get_all_iso_annual_prices(year=year, use_cache=True, force_refresh=False)
        print(f"  ISO prices: {iso_prices}")
    
    print(f"\n{'='*60}")
    print(f"WIND ENERGY HEATMAP GENERATOR (ISO Regional Prices)")
    print(f"{'='*60}")
    print(f"Year: {year}")
    print(f"Sample density: {density}x{density} = {density**2} points")
    print(f"Turbine: {turbine_params['rated_power_mw']} MW")
    print(f"Pricing: ISO-specific regional prices")
    print(f"{'='*60}\n")
    
    # Create sample grid
    print("Creating sample grid...")
    sample_points = create_sample_grid(US_BOUNDS, density)
    print(f"Generated {len(sample_points)} sample points\n")
    
    # Calculate revenue for each sample point
    print("Fetching wind data and calculating revenue...")
    print("(This will take some time due to API rate limits)\n")
    
    sample_revenues = []
    successful_points = []
    
    start_time = time.time()
    
    for i, (lat, lon) in enumerate(sample_points):
        # Progress indicator
        pct = (i + 1) / len(sample_points) * 100
        elapsed = time.time() - start_time
        if i > 0:
            eta = elapsed / i * (len(sample_points) - i)
            eta_str = f"ETA: {eta/60:.1f} min"
        else:
            eta_str = "ETA: calculating..."
        
        print(f"[{i+1}/{len(sample_points)}] ({pct:.1f}%) {eta_str}")
        
        revenue = calculate_annual_revenue_for_point(
            lat, lon, year, turbine_params, iso_prices, verbose
        )
        
        sample_revenues.append(revenue)
        
        if revenue > 0:
            successful_points.append((lat, lon))
        
        # Small delay to be respectful to the API
        time.sleep(0.5)
    
    total_time = time.time() - start_time
    print(f"\nData collection complete in {total_time/60:.1f} minutes")
    print(f"  Successful points: {len(successful_points)}/{len(sample_points)}")
    
    # Interpolate to regular grid
    print("\nInterpolating to regular grid...")
    lat_grid, lon_grid, revenue_grid = interpolate_revenue_data(
        sample_points,
        sample_revenues,
        US_BOUNDS
    )
    print(f"  Grid shape: {lat_grid.shape}")
    
    # Calculate data availability mask
    print("Calculating data availability mask...")
    availability_mask = calculate_data_availability_mask(
        lat_grid, lon_grid, successful_points, max_distance_km=100.0
    )
    valid_points = availability_mask.sum()
    total_points = availability_mask.size
    print(f"  Valid grid points: {valid_points}/{total_points} ({valid_points/total_points*100:.1f}%)")
    
    # Load US boundary with offshore buffer for wind potential
    print("\nLoading US boundary with offshore buffer...")
    offshore_buffer_km = 50.0
    us_boundary = load_us_boundary_with_buffer(buffer_km=offshore_buffer_km)
    if us_boundary is not None:
        print(f"  US boundary loaded with {offshore_buffer_km}km offshore buffer")
        us_mask = create_us_mask(lat_grid, lon_grid, us_boundary=us_boundary)
        us_points = us_mask.sum()
        print(f"  Points within US boundary: {us_points}/{total_points} ({us_points/total_points*100:.1f}%)")
    else:
        print("  Warning: Could not load US boundary, using rectangular bounds")
        us_mask = None
    
    # Create final data structure
    print("\nCreating final data structure...")
    avg_iso_price = sum(iso_prices.values()) / len(iso_prices) if iso_prices else 0.0
    heatmap_data = create_heatmap_data_structure(
        lat_grid=lat_grid,
        lon_grid=lon_grid,
        revenue_grid=revenue_grid,
        availability_mask=availability_mask,
        year=year,
        turbine_params=turbine_params,
        price_usd_mwh=avg_iso_price,
        sample_points=sample_points,
        us_mask=us_mask,
        offshore_buffer_km=offshore_buffer_km,
        iso_prices=iso_prices
    )
    
    # Summary statistics
    valid_revenues = revenue_grid[heatmap_data["availability_mask"]]
    print(f"\nRevenue Statistics:")
    print(f"  Min: ${valid_revenues.min():,.0f}/year")
    print(f"  Max: ${valid_revenues.max():,.0f}/year")
    print(f"  Mean: ${valid_revenues.mean():,.0f}/year")
    print(f"  Median: ${np.median(valid_revenues):,.0f}/year")
    
    return heatmap_data


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Generate annual revenue heatmap data for wind energy dashboard"
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2024,
        help="Year of data to generate (default: 2024)"
    )
    parser.add_argument(
        "--density",
        type=int,
        default=15,
        help="Number of sample points per axis (default: 15)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress information"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.year < 2000 or args.year > datetime.now().year:
        print(f"Error: Invalid year {args.year}")
        sys.exit(1)
    
    if args.density < 5 or args.density > 50:
        print(f"Error: Density must be between 5 and 50")
        sys.exit(1)
    
    # Generate heatmap data
    heatmap_data = generate_heatmap_data(
        year=args.year,
        density=args.density,
        verbose=args.verbose
    )
    
    # Save to cache
    print(f"\nSaving to cache...")
    success = save_heatmap_cache(heatmap_data, args.year)
    
    if success:
        print(f"\n{'='*60}")
        print(f"SUCCESS!")
        print(f"Heatmap saved to: data/heatmap_cache_{args.year}.joblib")
        print(f"{'='*60}\n")
    else:
        print(f"\n{'='*60}")
        print(f"ERROR: Failed to save heatmap cache")
        print(f"{'='*60}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
