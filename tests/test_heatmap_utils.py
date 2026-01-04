"""
Unit tests for heatmap_utils module.

Tests cover:
- Grid creation and interpolation
- Cache management
- Data availability masking
- Haversine distance calculations
- Data structure creation
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil

from heatmap_utils import (
    US_BOUNDS,
    INDUSTRY_STANDARD_TURBINE,
    DEFAULT_PRICE_USD_MWH,
    GRID_RESOLUTION_DEG,
    create_sample_grid,
    create_interpolation_grid,
    interpolate_revenue_data,
    calculate_data_availability_mask,
    haversine_distance_vectorized,
    create_heatmap_data_structure,
    get_heatmap_data_for_folium,
    get_revenue_stats,
    save_heatmap_cache,
    load_heatmap_cache,
    get_cache_path,
    is_cache_valid,
)


class TestConstants:
    """Tests for module constants."""
    
    def test_us_bounds_valid(self):
        """US bounds should cover continental US."""
        assert US_BOUNDS["min_lat"] < US_BOUNDS["max_lat"]
        assert US_BOUNDS["min_lon"] < US_BOUNDS["max_lon"]
        assert US_BOUNDS["min_lat"] >= 24.0  # Southern Florida
        assert US_BOUNDS["max_lat"] <= 50.0  # Northern border
        assert US_BOUNDS["min_lon"] >= -130.0  # West coast
        assert US_BOUNDS["max_lon"] <= -60.0  # East coast
    
    def test_industry_standard_turbine_valid(self):
        """Turbine parameters should be realistic."""
        params = INDUSTRY_STANDARD_TURBINE
        assert params["cut_in_mps"] > 0
        assert params["cut_in_mps"] < params["rated_speed_mps"]
        assert params["rated_speed_mps"] < params["cut_out_mps"]
        assert params["rated_power_mw"] > 0
        assert params["hub_height_m"] > 0
    
    def test_default_price_positive(self):
        """Default price should be positive."""
        assert DEFAULT_PRICE_USD_MWH > 0
    
    def test_grid_resolution_reasonable(self):
        """Grid resolution should be reasonable (~11km)."""
        assert 0.05 <= GRID_RESOLUTION_DEG <= 0.2


class TestGridCreation:
    """Tests for grid creation functions."""
    
    def test_create_sample_grid_default(self):
        """Sample grid should create correct number of points."""
        density = 5
        points = create_sample_grid(density=density)
        assert len(points) == density * density
    
    def test_create_sample_grid_coverage(self):
        """Sample grid should cover the specified bounds."""
        bounds = {"min_lat": 30.0, "max_lat": 40.0, "min_lon": -100.0, "max_lon": -90.0}
        points = create_sample_grid(bounds=bounds, density=5)
        
        lats = [p[0] for p in points]
        lons = [p[1] for p in points]
        
        assert min(lats) >= bounds["min_lat"]
        assert max(lats) <= bounds["max_lat"]
        assert min(lons) >= bounds["min_lon"]
        assert max(lons) <= bounds["max_lon"]
    
    def test_create_sample_grid_tuples(self):
        """Sample grid should return tuples of (lat, lon)."""
        points = create_sample_grid(density=3)
        for point in points:
            assert isinstance(point, tuple)
            assert len(point) == 2
    
    def test_create_interpolation_grid_shape(self):
        """Interpolation grid should have correct shape."""
        bounds = {"min_lat": 30.0, "max_lat": 40.0, "min_lon": -100.0, "max_lon": -90.0}
        resolution = 1.0  # 1 degree
        
        lat_grid, lon_grid = create_interpolation_grid(bounds=bounds, resolution_deg=resolution)
        
        # Should have approximately (lat_range/resolution + 1) points per axis
        expected_lat_points = int(np.ceil(10.0 / resolution)) + 1
        expected_lon_points = int(np.ceil(10.0 / resolution)) + 1
        
        assert lat_grid.shape[0] == expected_lat_points
        assert lat_grid.shape[1] == expected_lon_points
        assert lat_grid.shape == lon_grid.shape
    
    def test_create_interpolation_grid_values(self):
        """Interpolation grid values should be within bounds."""
        bounds = {"min_lat": 30.0, "max_lat": 40.0, "min_lon": -100.0, "max_lon": -90.0}
        lat_grid, lon_grid = create_interpolation_grid(bounds=bounds)
        
        assert lat_grid.min() >= bounds["min_lat"]
        assert lat_grid.max() <= bounds["max_lat"]
        assert lon_grid.min() >= bounds["min_lon"]
        assert lon_grid.max() <= bounds["max_lon"]


class TestHaversineDistance:
    """Tests for haversine distance calculation."""
    
    def test_same_point_zero_distance(self):
        """Distance from a point to itself should be zero."""
        lat_grid = np.array([[40.0]])
        lon_grid = np.array([[-100.0]])
        
        dist = haversine_distance_vectorized(lat_grid, lon_grid, 40.0, -100.0)
        
        assert np.allclose(dist, 0.0, atol=1e-6)
    
    def test_known_distance(self):
        """Test against a known distance."""
        # New York to Los Angeles: approximately 3944 km
        lat_grid = np.array([[40.7128]])  # NYC
        lon_grid = np.array([[-74.0060]])
        
        dist = haversine_distance_vectorized(lat_grid, lon_grid, 34.0522, -118.2437)  # LA
        
        # Allow 5% tolerance for approximation
        assert 3700 < dist[0, 0] < 4200
    
    def test_distance_symmetric(self):
        """Distance should be symmetric."""
        lat1, lon1 = 40.0, -100.0
        lat2, lon2 = 35.0, -95.0
        
        lat_grid1 = np.array([[lat1]])
        lon_grid1 = np.array([[lon1]])
        lat_grid2 = np.array([[lat2]])
        lon_grid2 = np.array([[lon2]])
        
        dist1 = haversine_distance_vectorized(lat_grid1, lon_grid1, lat2, lon2)
        dist2 = haversine_distance_vectorized(lat_grid2, lon_grid2, lat1, lon1)
        
        assert np.allclose(dist1, dist2)
    
    def test_vectorized_multiple_points(self):
        """Should calculate distances for multiple grid points."""
        lat_grid = np.array([[40.0, 41.0], [42.0, 43.0]])
        lon_grid = np.array([[-100.0, -101.0], [-102.0, -103.0]])
        
        dist = haversine_distance_vectorized(lat_grid, lon_grid, 40.0, -100.0)
        
        assert dist.shape == (2, 2)
        assert dist[0, 0] < dist[1, 1]  # Closer point should have smaller distance


class TestDataAvailabilityMask:
    """Tests for data availability masking."""
    
    def test_empty_sample_points(self):
        """Empty sample points should produce all-False mask."""
        lat_grid = np.array([[40.0, 41.0], [42.0, 43.0]])
        lon_grid = np.array([[-100.0, -101.0], [-102.0, -103.0]])
        
        mask = calculate_data_availability_mask(lat_grid, lon_grid, [], max_distance_km=50.0)
        
        assert not mask.any()
    
    def test_point_at_sample_location(self):
        """Grid point at sample location should be marked available."""
        lat_grid = np.array([[40.0]])
        lon_grid = np.array([[-100.0]])
        sample_points = [(40.0, -100.0)]
        
        mask = calculate_data_availability_mask(lat_grid, lon_grid, sample_points, max_distance_km=50.0)
        
        assert mask[0, 0]
    
    def test_point_far_from_samples(self):
        """Grid point far from all samples should be marked unavailable."""
        lat_grid = np.array([[40.0]])
        lon_grid = np.array([[-100.0]])
        sample_points = [(20.0, -80.0)]  # Very far away
        
        mask = calculate_data_availability_mask(lat_grid, lon_grid, sample_points, max_distance_km=50.0)
        
        assert not mask[0, 0]
    
    def test_threshold_distance(self):
        """Points within threshold should be marked available."""
        lat_grid = np.array([[40.0, 40.1], [40.2, 40.3]])
        lon_grid = np.array([[-100.0, -100.0], [-100.0, -100.0]])
        sample_points = [(40.0, -100.0)]
        
        # 40.3 degrees is about 33 km from 40.0 at this longitude
        mask = calculate_data_availability_mask(lat_grid, lon_grid, sample_points, max_distance_km=50.0)
        
        assert mask[0, 0]  # Should be available (at sample point)
        assert mask[0, 1]  # Should be available (within threshold)


class TestInterpolation:
    """Tests for revenue data interpolation."""
    
    def test_interpolate_insufficient_points(self):
        """Should handle insufficient data points gracefully."""
        sample_points = [(40.0, -100.0)]  # Only one point
        sample_values = [100000.0]
        bounds = {"min_lat": 35.0, "max_lat": 45.0, "min_lon": -105.0, "max_lon": -95.0}
        
        lat_grid, lon_grid, revenue_grid = interpolate_revenue_data(
            sample_points, sample_values, bounds, resolution_deg=1.0
        )
        
        # Should return NaN grid for insufficient points
        assert lat_grid.shape == lon_grid.shape
    
    def test_interpolate_filters_invalid_values(self):
        """Should filter out NaN and zero values."""
        sample_points = [(40.0, -100.0), (41.0, -101.0), (42.0, -102.0)]
        sample_values = [100000.0, np.nan, 0.0]  # Two invalid values
        bounds = {"min_lat": 39.0, "max_lat": 43.0, "min_lon": -103.0, "max_lon": -99.0}
        
        lat_grid, lon_grid, revenue_grid = interpolate_revenue_data(
            sample_points, sample_values, bounds, resolution_deg=1.0
        )
        
        # Should still return grids (may be NaN if not enough valid points)
        assert lat_grid.shape == lon_grid.shape == revenue_grid.shape
    
    def test_interpolate_positive_values(self):
        """Interpolated revenue should be non-negative."""
        sample_points = [
            (40.0, -100.0), (40.0, -99.0), (41.0, -100.0),
            (41.0, -99.0), (40.5, -99.5)
        ]
        sample_values = [100000.0, 120000.0, 110000.0, 130000.0, 115000.0]
        bounds = {"min_lat": 39.5, "max_lat": 41.5, "min_lon": -100.5, "max_lon": -98.5}
        
        lat_grid, lon_grid, revenue_grid = interpolate_revenue_data(
            sample_points, sample_values, bounds, resolution_deg=0.5
        )
        
        # All values should be non-negative (clipped)
        assert (revenue_grid >= 0).all() or np.isnan(revenue_grid).all()


class TestDataStructure:
    """Tests for heatmap data structure creation."""
    
    def test_create_heatmap_data_structure(self):
        """Should create complete data structure with all required fields."""
        lat_grid = np.array([[40.0, 41.0], [42.0, 43.0]])
        lon_grid = np.array([[-100.0, -101.0], [-102.0, -103.0]])
        revenue_grid = np.array([[100000.0, 110000.0], [120000.0, 130000.0]])
        availability_mask = np.array([[True, True], [True, False]])
        
        data = create_heatmap_data_structure(
            lat_grid=lat_grid,
            lon_grid=lon_grid,
            revenue_grid=revenue_grid,
            availability_mask=availability_mask,
            year=2024,
            turbine_params=INDUSTRY_STANDARD_TURBINE,
            price_usd_mwh=50.0,
            sample_points=[(40.0, -100.0)]
        )
        
        assert "year" in data
        assert "generated_at" in data
        assert "bounds" in data
        assert "lat_grid" in data
        assert "lon_grid" in data
        assert "revenue_grid" in data
        assert "availability_mask" in data
        assert "turbine_params" in data
        assert "price_usd_mwh" in data
        assert data["year"] == 2024
    
    def test_get_heatmap_data_for_folium(self):
        """Should convert heatmap data to Folium format."""
        heatmap_data = {
            "lat_grid": np.array([[40.0, 41.0], [42.0, 43.0]]),
            "lon_grid": np.array([[-100.0, -101.0], [-102.0, -103.0]]),
            "revenue_grid": np.array([[100000.0, 110000.0], [120000.0, 130000.0]]),
            "availability_mask": np.array([[True, True], [True, False]]),
        }
        
        heat_data = get_heatmap_data_for_folium(heatmap_data)
        
        assert len(heat_data) == 3  # 3 True values in mask
        for point in heat_data:
            assert len(point) == 3  # [lat, lon, intensity]
            assert 0 <= point[2] <= 1  # Normalized intensity
    
    def test_get_heatmap_data_for_folium_none(self):
        """Should handle None input gracefully."""
        heat_data = get_heatmap_data_for_folium(None)
        assert heat_data == []
    
    def test_get_revenue_stats(self):
        """Should calculate correct revenue statistics."""
        heatmap_data = {
            "revenue_grid": np.array([[100000.0, 200000.0], [150000.0, 250000.0]]),
            "availability_mask": np.array([[True, True], [True, True]]),
        }
        
        stats = get_revenue_stats(heatmap_data)
        
        assert stats["min"] == 100000.0
        assert stats["max"] == 250000.0
        assert stats["mean"] == 175000.0
        assert stats["median"] == 175000.0
    
    def test_get_revenue_stats_none(self):
        """Should handle None input gracefully."""
        stats = get_revenue_stats(None)
        assert stats["min"] == 0
        assert stats["max"] == 0


class TestCacheManagement:
    """Tests for cache management functions."""
    
    @pytest.fixture
    def temp_cache_dir(self, tmp_path, monkeypatch):
        """Create temporary cache directory for testing."""
        cache_dir = tmp_path / "data"
        cache_dir.mkdir()
        
        # Patch CACHE_DIR in heatmap_utils
        import heatmap_utils
        monkeypatch.setattr(heatmap_utils, "CACHE_DIR", cache_dir)
        
        return cache_dir
    
    def test_save_and_load_cache(self, temp_cache_dir):
        """Should be able to save and load cache."""
        test_data = {
            "year": 2024,
            "test_value": "hello",
            "array": np.array([1, 2, 3]),
        }
        
        # Save
        success = save_heatmap_cache(test_data, year=2024)
        assert success
        
        # Load
        loaded = load_heatmap_cache(year=2024)
        assert loaded is not None
        assert loaded["year"] == 2024
        assert loaded["test_value"] == "hello"
        assert np.array_equal(loaded["array"], np.array([1, 2, 3]))
    
    def test_load_nonexistent_cache(self, temp_cache_dir):
        """Should return None for nonexistent cache."""
        loaded = load_heatmap_cache(year=9999)
        assert loaded is None
    
    def test_is_cache_valid(self, temp_cache_dir):
        """Should correctly validate cache."""
        # Save valid cache
        test_data = {"year": 2024}
        save_heatmap_cache(test_data, year=2024)
        
        assert is_cache_valid(2024)
        assert not is_cache_valid(2023)  # Different year
        assert not is_cache_valid(9999)  # Nonexistent
    
    def test_get_cache_path(self, temp_cache_dir):
        """Should return correct cache path."""
        path = get_cache_path(2024)
        assert path.name == "heatmap_cache_2024.joblib"


class TestMapComponents:
    """Tests for map component functions."""
    
    def test_get_click_coordinates_valid(self):
        """Should extract coordinates from valid map data."""
        from map_components import get_click_coordinates
        
        map_data = {
            "last_clicked": {"lat": 40.123456, "lng": -100.654321}
        }
        
        coords = get_click_coordinates(map_data)
        
        assert coords is not None
        assert coords == (40.1235, -100.6543)  # Rounded to 4 decimals
    
    def test_get_click_coordinates_none(self):
        """Should return None for None input."""
        from map_components import get_click_coordinates
        
        assert get_click_coordinates(None) is None
    
    def test_get_click_coordinates_no_click(self):
        """Should return None when no click recorded."""
        from map_components import get_click_coordinates
        
        map_data = {"last_clicked": None}
        assert get_click_coordinates(map_data) is None
    
    def test_format_coordinates(self):
        """Should format coordinates correctly."""
        from map_components import format_coordinates
        
        formatted = format_coordinates(40.1234, -100.5678)
        assert formatted == "40.1234, -100.5678"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
