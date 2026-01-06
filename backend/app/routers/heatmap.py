from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
import numpy as np
import joblib
import io
from pathlib import Path

router = APIRouter(prefix="/api/heatmap", tags=["heatmap"])

# Cache the heatmap data
_heatmap_cache = None

def get_heatmap_data():
    """Load and cache heatmap data."""
    global _heatmap_cache
    if _heatmap_cache is None:
        # Look for heatmap file in data directory
        data_path = Path(__file__).parent.parent.parent.parent / "data" / "heatmap_cache_2024.joblib"
        if not data_path.exists():
            return None
        _heatmap_cache = joblib.load(data_path)
    return _heatmap_cache


def revenue_to_color(value: float, min_val: float, max_val: float) -> tuple:
    """Convert revenue value to RGBA color (green to yellow to red gradient)."""
    if np.isnan(value) or value <= 0:
        return (0, 0, 0, 0)  # Transparent
    
    # Normalize to 0-1
    normalized = (value - min_val) / (max_val - min_val) if max_val > min_val else 0
    normalized = max(0, min(1, normalized))
    
    # Color gradient: Green (low) -> Yellow (mid) -> Red (high)
    # 0.0 = Green  (0, 200, 0)
    # 0.5 = Yellow (255, 255, 0)
    # 1.0 = Red    (255, 0, 0)
    
    if normalized < 0.5:
        # Green to Yellow (0 to 0.5)
        t = normalized * 2  # Scale to 0-1
        r = int(0 + 255 * t)
        g = int(200 + 55 * t)  # 200 -> 255
        b = 0
    else:
        # Yellow to Red (0.5 to 1.0)
        t = (normalized - 0.5) * 2  # Scale to 0-1
        r = 255
        g = int(255 * (1 - t))
        b = 0
    
    # Opacity: more opaque for higher values
    a = int(140 + 80 * normalized)  # Range: 140-220
    
    return (r, g, b, a)


@router.get("/bounds")
async def get_bounds():
    """Get the geographic bounds of the heatmap."""
    data = get_heatmap_data()
    if data is None:
        raise HTTPException(status_code=404, detail="Heatmap data not found")
    
    return {
        "bounds": data["bounds"],
        "grid_shape": data["grid_shape"],
        "year": data["year"],
    }


@router.get("/image.png")
async def get_heatmap_image():
    """Generate and return the heatmap as a PNG image."""
    data = get_heatmap_data()
    if data is None:
        raise HTTPException(status_code=404, detail="Heatmap data not found")
    
    try:
        from PIL import Image
    except ImportError:
        raise HTTPException(status_code=500, detail="PIL not installed")
    
    revenue_grid = data["revenue_grid"]
    
    # Get min/max for color scaling (excluding zeros and NaN)
    valid_values = revenue_grid[~np.isnan(revenue_grid) & (revenue_grid > 0)]
    if len(valid_values) == 0:
        raise HTTPException(status_code=500, detail="No valid heatmap data")
    
    min_val = np.percentile(valid_values, 5)  # Use 5th percentile to avoid outliers
    max_val = np.percentile(valid_values, 95)  # Use 95th percentile
    
    # Create RGBA image
    height, width = revenue_grid.shape
    img_array = np.zeros((height, width, 4), dtype=np.uint8)
    
    for i in range(height):
        for j in range(width):
            img_array[i, j] = revenue_to_color(revenue_grid[i, j], min_val, max_val)
    
    # Flip vertically because image coordinates are top-down
    img_array = np.flipud(img_array)
    
    # Create PIL image
    img = Image.fromarray(img_array, mode='RGBA')
    
    # Save to bytes
    buffer = io.BytesIO()
    img.save(buffer, format='PNG', optimize=True)
    buffer.seek(0)
    
    return Response(
        content=buffer.getvalue(),
        media_type="image/png",
        headers={
            "Cache-Control": "public, max-age=86400",  # Cache for 24 hours
        }
    )


@router.get("/legend")
async def get_legend():
    """Get legend information for the heatmap."""
    data = get_heatmap_data()
    if data is None:
        raise HTTPException(status_code=404, detail="Heatmap data not found")
    
    revenue_grid = data["revenue_grid"]
    valid_values = revenue_grid[~np.isnan(revenue_grid) & (revenue_grid > 0)]
    
    if len(valid_values) == 0:
        return {"min": 0, "max": 0, "median": 0, "mean": 0, "unit": "$/year"}
    
    min_val = float(np.percentile(valid_values, 5))
    max_val = float(np.percentile(valid_values, 95))
    median_val = float(np.median(valid_values))
    mean_val = float(np.mean(valid_values))
    
    return {
        "min": min_val,
        "max": max_val,
        "median": median_val,
        "mean": mean_val,
        "unit": "$/year",
        "description": "Estimated annual revenue for 3MW turbine",
    }
