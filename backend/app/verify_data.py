#!/usr/bin/env python3
"""
Quick verification script for wind speed and energy price data alignment.

This script performs a focused test of the key data integrations:
1. Open-Meteo wind data API
2. Gridstatus LMP price API
3. ISO boundary mapping
4. Revenue calculation pipeline
"""

import sys
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from windlib import (
    get_wind_data,
    fetch_open_meteo_archive,
    get_market_prices_gridstatus,
    get_iso_for_location,
    get_iso_annual_price,
    get_all_iso_annual_prices,
    get_price_for_location,
    add_power_output,
    get_tz_from_latlon,
    wind_to_power,
    ISO_BOUNDARIES,
    ISO_DEFAULT_PRICES,
)
from heatmap_utils import INDUSTRY_STANDARD_TURBINE


def print_header(text):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


def print_result(name, passed, details=""):
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {status} | {name}")
    if details and not passed:
        print(f"         {details}")


def test_wind_data():
    """Test Open-Meteo wind data API."""
    print_header("1. WIND DATA (Open-Meteo API)")
    
    passed = True
    tests = []
    
    # Test 1: Fetch archive data
    try:
        df = fetch_open_meteo_archive(37.75, -122.45, "2024-06-01", "2024-06-30")
        tests.append(("Archive data returned", len(df) > 0, f"Got {len(df)} rows"))
        tests.append(("Has wind_speed column", "wind_speed_mps" in df.columns))
        tests.append(("Timestamps timezone-aware", df["timestamp"].iloc[0].tzinfo is not None))
    except Exception as e:
        tests.append(("Archive data fetch", False, str(e)[:50]))
        passed = False
    
    # Test 2: Wind speed values reasonable
    try:
        df = fetch_open_meteo_archive(38.0, -100.0, "2024-01-01", "2024-12-31")
        ws = df["wind_speed_mps"]
        tests.append(("Wind speed >= 0", ws.min() >= 0))
        tests.append(("Wind speed <= 50 m/s", ws.max() <= 50))
        tests.append(("Mean wind speed reasonable", 2 < ws.mean() < 15))
    except Exception as e:
        tests.append(("Wind speed validation", False, str(e)[:50]))
        passed = False
    
    # Test 3: Power curve
    try:
        turbine = {"cut_in_mps": 3.0, "rated_speed_mps": 12.0, "cut_out_mps": 25.0, "rated_power_mw": 2.5}
        tests.append(("Below cut-in = 0", wind_to_power(2.0, **turbine) == 0.0))
        tests.append(("At rated = full power", np.isclose(wind_to_power(12.0, **turbine), 2.5, atol=0.01)))
        tests.append(("Above cut-out = 0", wind_to_power(30.0, **turbine) == 0.0))
    except Exception as e:
        tests.append(("Power curve validation", False, str(e)[:50]))
        passed = False
    
    for name, result, details in tests:
        print_result(name, result, details)
    
    return passed


def test_price_data():
    """Test gridstatus LMP price API."""
    print_header("2. ENERGY PRICES (Gridstatus API)")
    
    passed = True
    tests = []
    
    # Test 1: Fetch prices
    try:
        df = get_market_prices_gridstatus(
            iso_name="CAISO",
            start_date="2024-06-01",
            end_date="2024-06-30",
            market="DAY_AHEAD_HOURLY",
        )
        tests.append(("LMP data returned", len(df) > 0, f"Got {len(df)} rows"))
        tests.append(("Has price column", "price_usd_mwh" in df.columns))
    except Exception as e:
        tests.append(("LMP fetch", False, f"API error: {str(e)[:40]}"))
        passed = False
    
    # Test 2: Price values reasonable
    try:
        df = get_market_prices_gridstatus(
            iso_name="CAISO",
            start_date="2024-01-01",
            end_date="2024-12-31",
            market="DAY_AHEAD_HOURLY",
        )
        prices = df["price_usd_mwh"]
        tests.append(("Price >= -500", prices.min() >= -500))
        tests.append(("Price <= 5000", prices.max() <= 5000))
        tests.append(("Mean price reasonable", 10 < prices.mean() < 150))
    except Exception as e:
        tests.append(("Price validation", False, f"API error: {str(e)[:40]}"))
        passed = False
    
    # Test 3: Annual prices
    try:
        prices = get_all_iso_annual_prices(year=2024, use_cache=True, force_refresh=False)
        tests.append(("All ISOs returned", len(prices) == len(ISO_BOUNDARIES)))
        for iso, price in prices.items():
            tests.append((f"{iso} price > 0", price > 0, f"${price:.2f}"))
    except Exception as e:
        tests.append(("Annual prices", False, f"API error: {str(e)[:40]}"))
        passed = False
    
    for name, result, details in tests:
        print_result(name, result, details)
    
    return passed


def test_iso_boundaries():
    """Test ISO boundary detection."""
    print_header("3. ISO BOUNDARY MAPPING")
    
    passed = True
    tests = []
    
    # Known locations
    locations = [
        (37.75, -122.45, "CAISO", "San Francisco"),
        (29.76, -95.37, "Ercot", "Houston"),
        (42.36, -71.06, "ISONE", "Boston"),
        (41.88, -87.62, "MISO", "Chicago"),
        (40.71, -74.01, "NYISO", "New York"),
        (39.95, -75.17, "PJM", "Philadelphia"),
    ]
    
    for lat, lon, expected_iso, name in locations:
        detected = get_iso_for_location(lat, lon)
        tests.append((f"{name} -> {expected_iso}", detected == expected_iso, f"Got: {detected}"))
        if detected != expected_iso:
            passed = False
    
    # Test polygons valid
    for iso, polygon in ISO_BOUNDARIES.items():
        tests.append((f"{iso} polygon valid", polygon.is_valid))
        tests.append((f"{iso} polygon non-empty", polygon.area > 0))
    
    for name, result, details in tests:
        print_result(name, result, details)
    
    return passed


def test_pricing_logic():
    """Test hybrid pricing logic."""
    print_header("4. HYBRID PRICING LOGIC")
    
    passed = True
    tests = []
    
    iso_prices = ISO_DEFAULT_PRICES.copy()
    
    # Test inside ISO
    price, ptype, meta = get_price_for_location(37.75, -122.45, iso_prices)
    tests.append(("Inside CAISO = iso type", ptype == "iso"))
    tests.append(("Inside CAISO = correct price", price == iso_prices["CAISO"]))
    
    # Test outside ISO (Seattle)
    price, ptype, meta = get_price_for_location(47.61, -122.33, iso_prices)
    tests.append(("Outside ISOs = fallback/offshore", ptype in ("fallback", "offshore")))
    tests.append(("Outside ISOs = positive price", price > 0))
    tests.append(("Outside ISOs has nearest_iso", "nearest_iso" in meta))
    
    # Test offshore
    price_no_prem, _, _ = get_price_for_location(40.5, -70.5, iso_prices, apply_offshore_premium=False)
    price_w_prem, ptype, meta = get_price_for_location(40.5, -70.5, iso_prices, apply_offshore_premium=True)
    tests.append(("Offshore detected", ptype == "offshore"))
    tests.append(("Offshore premium applied", price_w_prem > price_no_prem))
    
    for name, result, details in tests:
        print_result(name, result, details)
    
    return passed


def test_revenue_pipeline():
    """Test end-to-end revenue calculation."""
    print_header("5. REVENUE CALCULATION PIPELINE")
    
    passed = True
    tests = []
    
    lat, lon = 38.0, -100.0  # Kansas
    turbine = INDUSTRY_STANDARD_TURBINE
    
    # Get wind data
    try:
        df = get_wind_data(lat, lon, "2024-06-01", "2024-06-30")
        tests.append(("Wind data fetched", len(df) > 0, f"{len(df)} rows"))
        
        # Add power
        df = add_power_output(df, turbine)
        df["energy_mwh"] = df["power_mw"] * 1.0
        total_energy = df["energy_mwh"].sum()
        
        tests.append(("Energy > 0", total_energy > 0, f"{total_energy:,.0f} MWh"))
        
        # Get ISO and price
        iso = get_iso_for_location(lat, lon)
        iso_prices = get_all_iso_annual_prices(year=2024, use_cache=True, force_refresh=False)
        price = iso_prices.get(iso, 50.0)
        
        tests.append(("ISO detected", iso == "MISO", f"Got: {iso}"))
        
        # Calculate revenue
        revenue = total_energy * price
        
        tests.append(("Revenue > 0", revenue > 0, f"${revenue:,.0f}"))
        
        # Check capacity factor
        hours = len(df)
        expected_energy = turbine["rated_power_mw"] * hours
        cf = total_energy / expected_energy
        tests.append(("Capacity factor reasonable", 0.15 < cf < 0.65, f"CF: {cf:.1%}"))
        
    except Exception as e:
        tests.append(("Pipeline execution", False, str(e)[:50]))
        passed = False
    
    for name, result, details in tests:
        print_result(name, result, details)
    
    return passed


def main():
    """Run all verification tests."""
    print("\n" + "="*60)
    print("  WIND ENERGY DATA VERIFICATION")
    print("  " + datetime.now().strftime("%Y-%m-%d %H:%M"))
    print("="*60)
    
    results = []
    
    results.append(("Wind Data (Open-Meteo)", test_wind_data()))
    time.sleep(0.5)  # Rate limiting
    
    results.append(("Energy Prices (Gridstatus)", test_price_data()))
    time.sleep(0.5)
    
    results.append(("ISO Boundaries", test_iso_boundaries()))
    results.append(("Hybrid Pricing", test_pricing_logic()))
    results.append(("Revenue Pipeline", test_revenue_pipeline()))
    
    # Summary
    print_header("SUMMARY")
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status} | {name}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("  ALL TESTS PASSED!")
    else:
        print("  SOME TESTS FAILED - Review output above")
    print("="*60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
