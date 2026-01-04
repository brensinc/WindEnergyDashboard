"""
Heatmap generation and caching utilities for wind energy dashboard.

This module provides functions for:
- Creating spatial grids for heatmap generation
- Interpolating revenue data across geographic areas
- Caching and loading pre-generated heatmap data
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Dict, Optional
from scipy.interpolate import RBFInterpolator

# -----------------------------
# Constants
# -----------------------------
CACHE_DIR = Path("data")

# Continental US bounds (latitude, longitude)
US_BOUNDS = {
    "min_lat": 24.5,   # Southern tip of Florida
    "max_lat": 49.5,   # Northern border
    "min_lon": -125.0, # West coast
    "max_lon": -66.0,  # East coast
}

# Industry-standard turbine parameters for modern utility-scale wind turbines
# Based on typical specifications for turbines like Vestas V110-2.0, GE 2.5-120, etc.
INDUSTRY_STANDARD_TURBINE = {
    "cut_in_mps": 3.0,      # Typical cut-in: 3-4 m/s
    "rated_speed_mps": 12.0, # Typical rated: 11-13 m/s
    "cut_out_mps": 25.0,     # Industry standard cut-out: 25 m/s
    "rated_power_mw": 2.5,   # Common utility-scale: 2-3 MW
    "hub_height_m": 100,     # Modern hub heights: 80-120m
}

# Default fixed price for heatmap generation ($/MWh)
DEFAULT_PRICE_USD_MWH = 50.0

# Grid resolution for heatmap (approximately 5.5km)
# 1 degree latitude ≈ 111 km, so 0.05° ≈ 5.5 km
GRID_RESOLUTION_DEG = 0.05

# Offshore buffer distance for US boundary (km)
# This allows showing potential offshore wind areas
OFFSHORE_BUFFER_KM = 50.0


# -----------------------------
# Cache Management
# -----------------------------
def get_cache_path(year: int) -> Path:
    """Get the path to the heatmap cache file for a given year."""
    return CACHE_DIR / f"heatmap_cache_{year}.joblib"


def load_heatmap_cache(year: int = 2024) -> Optional[Dict]:
    """
    Load pre-generated heatmap data from cache.
    
    Args:
        year: The year of data to load
        
    Returns:
        Dictionary containing heatmap data, or None if cache doesn't exist
    """
    cache_file = get_cache_path(year)
    if cache_file.exists():
        try:
            return joblib.load(cache_file)
        except Exception as e:
            print(f"Error loading heatmap cache: {e}")
            return None
    return None


def save_heatmap_cache(data: Dict, year: int = 2024) -> bool:
    """
    Save heatmap data to cache.
    
    Args:
        data: Dictionary containing heatmap data
        year: The year of data being cached
        
    Returns:
        True if save was successful, False otherwise
    """
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = get_cache_path(year)
    try:
        joblib.dump(data, cache_file)
        return True
    except Exception as e:
        print(f"Error saving heatmap cache: {e}")
        return False


def is_cache_valid(year: int) -> bool:
    """
    Check if the cache exists and is valid for the given year.
    
    Cache is considered valid if:
    - File exists
    - Contains data for the requested year
    - Was generated (no staleness check for pre-shipped data)
    
    Args:
        year: The year to check
        
    Returns:
        True if cache is valid, False otherwise
    """
    cache_file = get_cache_path(year)
    if not cache_file.exists():
        return False
    
    try:
        data = joblib.load(cache_file)
        return data.get("year") == year
    except Exception:
        return False


# -----------------------------
# Grid Generation
# -----------------------------
def create_sample_grid(
    bounds: Dict[str, float] = None,
    density: int = 15
) -> List[Tuple[float, float]]:
    """
    Create evenly-spaced sample points for heatmap data collection.
    
    Args:
        bounds: Dictionary with min_lat, max_lat, min_lon, max_lon
        density: Number of points along each axis (total points = density^2)
        
    Returns:
        List of (latitude, longitude) tuples
    """
    if bounds is None:
        bounds = US_BOUNDS
    
    lats = np.linspace(bounds["min_lat"], bounds["max_lat"], density)
    lons = np.linspace(bounds["min_lon"], bounds["max_lon"], density)
    
    return [(lat, lon) for lat in lats for lon in lons]


def create_interpolation_grid(
    bounds: Dict[str, float] = None,
    resolution_deg: float = GRID_RESOLUTION_DEG
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a regular grid for interpolation at specified resolution.
    
    Args:
        bounds: Dictionary with min_lat, max_lat, min_lon, max_lon
        resolution_deg: Grid spacing in degrees (0.1° ≈ 11km)
        
    Returns:
        Tuple of (lat_grid, lon_grid) as 2D numpy arrays
    """
    if bounds is None:
        bounds = US_BOUNDS
    
    lat_range = bounds["max_lat"] - bounds["min_lat"]
    lon_range = bounds["max_lon"] - bounds["min_lon"]
    
    n_lat = int(np.ceil(lat_range / resolution_deg)) + 1
    n_lon = int(np.ceil(lon_range / resolution_deg)) + 1
    
    lats = np.linspace(bounds["min_lat"], bounds["max_lat"], n_lat)
    lons = np.linspace(bounds["min_lon"], bounds["max_lon"], n_lon)
    
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    return lat_grid, lon_grid


# -----------------------------
# Spatial Interpolation
# -----------------------------
def interpolate_revenue_data(
    sample_points: List[Tuple[float, float]],
    sample_values: List[float],
    bounds: Dict[str, float] = None,
    resolution_deg: float = GRID_RESOLUTION_DEG
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolate sample revenue data to a regular grid using RBF interpolation.
    
    Args:
        sample_points: List of (lat, lon) sample locations
        sample_values: List of revenue values at each sample point
        bounds: Geographic bounds for the grid
        resolution_deg: Grid resolution in degrees
        
    Returns:
        Tuple of (lat_grid, lon_grid, revenue_grid) as 2D numpy arrays
    """
    if bounds is None:
        bounds = US_BOUNDS
    
    # Create the target grid
    lat_grid, lon_grid = create_interpolation_grid(bounds, resolution_deg)
    
    # Prepare sample data for interpolation
    # Filter out any NaN or invalid values
    valid_data = [
        (pt, val) for pt, val in zip(sample_points, sample_values)
        if val is not None and not np.isnan(val) and val > 0
    ]
    
    if len(valid_data) < 3:
        # Not enough data points for interpolation
        revenue_grid = np.full(lat_grid.shape, np.nan)
        return lat_grid, lon_grid, revenue_grid
    
    points = np.array([pt for pt, _ in valid_data])
    values = np.array([val for _, val in valid_data])
    
    # Normalize values for better interpolation
    value_mean = values.mean()
    value_std = values.std() if values.std() > 0 else 1.0
    values_normalized = (values - value_mean) / value_std
    
    # Create RBF interpolator
    # Using thin_plate_spline kernel for smooth interpolation
    try:
        interpolator = RBFInterpolator(
            points,
            values_normalized,
            kernel='thin_plate_spline',
            smoothing=0.1  # Small smoothing to handle noise
        )
        
        # Prepare grid points for interpolation
        grid_points = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
        
        # Interpolate
        interpolated_normalized = interpolator(grid_points)
        
        # Denormalize
        interpolated = interpolated_normalized * value_std + value_mean
        
        # Reshape to grid
        revenue_grid = interpolated.reshape(lat_grid.shape)
        
        # Clip negative values (revenue can't be negative in this context)
        revenue_grid = np.clip(revenue_grid, 0, None)
        
    except Exception as e:
        print(f"Interpolation error: {e}")
        revenue_grid = np.full(lat_grid.shape, np.nan)
    
    return lat_grid, lon_grid, revenue_grid


def calculate_data_availability_mask(
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    sample_points: List[Tuple[float, float]],
    max_distance_km: float = 100.0
) -> np.ndarray:
    """
    Create a boolean mask indicating where interpolated data is reliable.
    
    Points are marked as available only if they are within max_distance_km
    of at least one sample point.
    
    Args:
        lat_grid: 2D array of latitudes
        lon_grid: 2D array of longitudes
        sample_points: List of (lat, lon) sample locations
        max_distance_km: Maximum distance from sample point to consider valid
        
    Returns:
        Boolean 2D numpy array (True = data available)
    """
    if len(sample_points) == 0:
        return np.zeros(lat_grid.shape, dtype=bool)
    
    mask = np.zeros(lat_grid.shape, dtype=bool)
    
    for sample_lat, sample_lon in sample_points:
        # Calculate haversine distance from this sample point to all grid points
        distances = haversine_distance_vectorized(
            lat_grid, lon_grid, sample_lat, sample_lon
        )
        # Mark points within threshold as available
        mask = mask | (distances <= max_distance_km)
    
    return mask


def haversine_distance_vectorized(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: float,
    lon2: float
) -> np.ndarray:
    """
    Calculate haversine distance between grid points and a single point.
    
    Args:
        lat1, lon1: 2D arrays of grid point coordinates
        lat2, lon2: Single point coordinates
        
    Returns:
        2D array of distances in kilometers
    """
    R = 6371.0  # Earth's radius in km
    
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c


# -----------------------------
# Heatmap Data Structure
# -----------------------------
def create_heatmap_data_structure(
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    revenue_grid: np.ndarray,
    availability_mask: np.ndarray,
    year: int,
    turbine_params: Dict,
    price_usd_mwh: float,
    sample_points: List[Tuple[float, float]],
    us_mask: np.ndarray = None,
    offshore_buffer_km: float = OFFSHORE_BUFFER_KM
) -> Dict:
    """
    Create the complete heatmap data structure for caching.
    
    Args:
        lat_grid: 2D array of latitudes
        lon_grid: 2D array of longitudes
        revenue_grid: 2D array of annual revenue values
        availability_mask: Boolean mask for data availability
        year: Year of the data
        turbine_params: Turbine parameters used
        price_usd_mwh: Fixed price used for calculations
        sample_points: Original sample points used
        us_mask: Boolean mask for US boundary (including offshore buffer)
        offshore_buffer_km: Buffer distance in km for offshore areas
        
    Returns:
        Dictionary containing all heatmap data
    """
    if us_mask is None:
        us_mask = create_us_mask(lat_grid, lon_grid, buffer_km=offshore_buffer_km)
    
    combined_mask = availability_mask & us_mask
    revenue_grid_final = revenue_grid.copy()
    revenue_grid_final[~combined_mask] = 0.0
    
    return {
        "year": year,
        "generated_at": datetime.now().isoformat(),
        "bounds": US_BOUNDS,
        "resolution_deg": GRID_RESOLUTION_DEG,
        "resolution_km": int(GRID_RESOLUTION_DEG * 111),
        "lat_grid": lat_grid,
        "lon_grid": lon_grid,
        "revenue_grid": revenue_grid_final,
        "availability_mask": combined_mask,
        "original_availability_mask": availability_mask,
        "us_mask": us_mask,
        "offshore_buffer_km": offshore_buffer_km,
        "turbine_params": turbine_params,
        "price_usd_mwh": price_usd_mwh,
        "sample_points": sample_points,
        "n_sample_points": len(sample_points),
        "grid_shape": lat_grid.shape,
    }


def get_heatmap_data_for_folium(heatmap_data: Dict) -> List[List[float]]:
    """
    Convert heatmap data to format suitable for Folium HeatMap.
    
    Args:
        heatmap_data: Dictionary from load_heatmap_cache()
        
    Returns:
        List of [lat, lon, intensity] for each valid grid point
    """
    if heatmap_data is None:
        return []
    
    lat_grid = heatmap_data["lat_grid"]
    lon_grid = heatmap_data["lon_grid"]
    revenue_grid = heatmap_data["revenue_grid"]
    mask = heatmap_data["availability_mask"]
    
    # Normalize revenue values for heatmap intensity (0-1 scale)
    valid_revenues = revenue_grid[mask]
    if len(valid_revenues) == 0:
        return []
    
    rev_min = valid_revenues.min()
    rev_max = valid_revenues.max()
    rev_range = rev_max - rev_min if rev_max > rev_min else 1.0
    
    heat_data = []
    for i in range(lat_grid.shape[0]):
        for j in range(lat_grid.shape[1]):
            if mask[i, j]:
                # Normalize intensity to 0-1 range
                intensity = (revenue_grid[i, j] - rev_min) / rev_range
                heat_data.append([
                    float(lat_grid[i, j]),
                    float(lon_grid[i, j]),
                    float(intensity)
                ])
    
    return heat_data


def get_revenue_stats(heatmap_data: Dict) -> Dict:
    """
    Get summary statistics from heatmap data.
    
    Args:
        heatmap_data: Dictionary from load_heatmap_cache()
        
    Returns:
        Dictionary with min, max, mean, median revenue values
    """
    if heatmap_data is None:
        return {"min": 0, "max": 0, "mean": 0, "median": 0}
    
    mask = heatmap_data["availability_mask"]
    revenue_grid = heatmap_data["revenue_grid"]
    valid_revenues = revenue_grid[mask]
    
    if len(valid_revenues) == 0:
        return {"min": 0, "max": 0, "mean": 0, "median": 0}
    
    return {
        "min": float(valid_revenues.min()),
        "max": float(valid_revenues.max()),
        "mean": float(valid_revenues.mean()),
        "median": float(np.median(valid_revenues)),
    }


# -----------------------------
# US Boundary with Offshore Buffer
# -----------------------------

try:
    import geopandas as gpd
    from shapely.geometry import Point
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False
    Point = None

NATURAL_EARTH_URL = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
BOUNDARY_CACHE_DIR = Path("data/boundaries")


def get_us_boundary_cache_path() -> Path:
    """Get path for cached US boundary shapefile."""
    return BOUNDARY_CACHE_DIR / "ne_110m_admin_0_countries.zip"


def download_us_boundary(force: bool = False) -> Optional[str]:
    """
    Download US boundary shapefile from Natural Earth.
    
    Args:
        force: If True, re-download even if cached
        
    Returns:
        Path to downloaded shapefile, or None if download failed
    """
    if not GEOPANDAS_AVAILABLE:
        print("Warning: geopandas not available. US boundary mask will not work.")
        return None
    
    cache_path = get_us_boundary_cache_path()
    
    if cache_path.exists() and not force:
        print(f"Using cached boundary data: {cache_path}")
        return str(cache_path)
    
    BOUNDARY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading US boundary from Natural Earth...")
    try:
        import urllib.request
        import ssl
        
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        
        urllib.request.urlretrieve(NATURAL_EARTH_URL, str(cache_path))
        print(f"Downloaded to: {cache_path}")
        return str(cache_path)
    except Exception as e:
        print(f"Error downloading boundary data: {e}")
        return None


def load_us_boundary_with_buffer(
    buffer_km: float = OFFSHORE_BUFFER_KM,
    force_download: bool = False
) -> Optional:
    """
    Load US boundary polygon with offshore buffer for wind potential.
    
    Args:
        buffer_km: Buffer distance in kilometers for offshore wind areas
        force_download: If True, re-download boundary data
        
    Returns:
        Shapely polygon representing US with buffer, or None if unavailable
    """
    if not GEOPANDAS_AVAILABLE:
        print("Warning: geopandas not available for US boundary loading")
        return None
    
    shapefile_path = download_us_boundary(force=force_download)
    if shapefile_path is None:
        return None
    
    try:
        gdf = gpd.read_file(shapefile_path)
        
        us_row = gdf[gdf['NAME'] == 'United States of America']
        if us_row.empty:
            us_row = gdf[gdf['SOVEREIGNT'] == 'United States of America']
        
        if us_row.empty:
            print("Warning: Could not find US boundary in Natural Earth data")
            return None
        
        us_boundary = us_row.geometry.iloc[0]
        
        if buffer_km > 0:
            us_boundary = us_boundary.buffer(buffer_km / 111.0)
        
        return us_boundary
        
    except Exception as e:
        print(f"Error loading US boundary: {e}")
        return None


def create_us_mask(
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    us_boundary: Optional = None,
    buffer_km: float = OFFSHORE_BUFFER_KM
) -> np.ndarray:
    """
    Create boolean mask for points inside US boundary with offshore buffer.
    
    Args:
        lat_grid: 2D array of latitudes
        lon_grid: 2D array of longitudes
        us_boundary: Pre-loaded US boundary polygon (optional)
        buffer_km: Buffer distance for offshore areas
        
    Returns:
        Boolean 2D numpy array (True = inside US/buffer zone)
    """
    if us_boundary is None:
        us_boundary = load_us_boundary_with_buffer(buffer_km)
    
    if us_boundary is None or Point is None:
        return np.ones(lat_grid.shape, dtype=bool)
    
    mask = np.zeros(lat_grid.shape, dtype=bool)
    
    for i in range(lat_grid.shape[0]):
        for j in range(lat_grid.shape[1]):
            point = Point(lon_grid[i, j], lat_grid[i, j])
            mask[i, j] = us_boundary.contains(point)
    
    return mask


def apply_us_boundary_mask(
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    revenue_grid: np.ndarray,
    availability_mask: np.ndarray,
    buffer_km: float = OFFSHORE_BUFFER_KM
) -> tuple:
    """
    Apply US boundary mask to heatmap data.
    
    Sets points outside US boundaries (including buffer zone) to zero revenue.
    
    Args:
        lat_grid: 2D array of latitudes
        lon_grid: 2D array of longitudes
        revenue_grid: 2D array of revenue values
        availability_mask: Existing availability mask
        buffer_km: Buffer distance for offshore areas
        
    Returns:
        Tuple of (revenue_grid, us_mask) with mask applied
    """
    us_mask = create_us_mask(lat_grid, lon_grid, buffer_km=buffer_km)
    
    combined_mask = availability_mask & us_mask
    revenue_grid_masked = revenue_grid.copy()
    revenue_grid_masked[~combined_mask] = 0.0
    
    return revenue_grid_masked, combined_mask
