from fastapi import APIRouter, HTTPException, Query
from datetime import date
import sys
import os

# Add parent directory to path to import windlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from app.models.wind_data import WindDataResponse, RevenueDataResponse

router = APIRouter(prefix="/api/wind", tags=["wind"])


@router.get("", response_model=WindDataResponse)
async def get_wind_data(
    latitude: float = Query(..., ge=-90, le=90),
    longitude: float = Query(..., ge=-180, le=180),
    start_date: str = Query(..., description="Start date (YYYY-MM-DD)"),
    end_date: str = Query(..., description="End date (YYYY-MM-DD)"),
    hub_height_m: int = Query(100, ge=10, le=200),
    rotor_diameter_m: int = Query(100, ge=20, le=200),
    rated_power_kw: int = Query(None, ge=100, le=20000),
):
    """
    Fetch wind data for a location and calculate power output.
    """
    try:
        # Import windlib here to avoid loading heavy dependencies at startup
        import windlib
        
        # Parse dates
        start = date.fromisoformat(start_date)
        end = date.fromisoformat(end_date)
        
        # Fetch wind data
        df = windlib.get_wind_data(
            lat=latitude,
            lon=longitude,
            start_date=start,
            end_date=end,
            hub_height=hub_height_m
        )
        
        if df.empty:
            raise HTTPException(status_code=404, detail="No wind data available for this location/date range")
        
        # Calculate power output
        df = windlib.calculate_power_output(
            df,
            rotor_diameter=rotor_diameter_m,
            rated_power_kw=rated_power_kw
        )
        
        # Determine ISO region
        iso = windlib.get_iso_for_location(latitude, longitude)
        
        # Calculate capacity factors
        if rated_power_kw:
            capacity_factors = (df['power_kw'] / rated_power_kw).tolist()
        else:
            # Use estimated rated power from power curve
            max_power = df['power_kw'].max()
            capacity_factors = (df['power_kw'] / max_power).tolist() if max_power > 0 else [0.0] * len(df)
        
        return WindDataResponse(
            timestamps=df['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S').tolist(),
            wind_speeds=df['wind_speed_hub'].tolist(),
            wind_directions=df['wind_direction'].tolist() if 'wind_direction' in df.columns else [0.0] * len(df),
            power_outputs=df['power_kw'].tolist(),
            capacity_factors=capacity_factors,
            metadata={
                "latitude": latitude,
                "longitude": longitude,
                "hub_height_m": hub_height_m,
                "rotor_diameter_m": rotor_diameter_m,
                "rated_power_kw": rated_power_kw or int(df['power_kw'].max()),
                "iso": iso,
                "pricing_type": None,
            }
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching wind data: {str(e)}")


@router.get("/revenue", response_model=RevenueDataResponse)
async def get_revenue_data(
    latitude: float = Query(..., ge=-90, le=90),
    longitude: float = Query(..., ge=-180, le=180),
    start_date: str = Query(..., description="Start date (YYYY-MM-DD)"),
    end_date: str = Query(..., description="End date (YYYY-MM-DD)"),
    hub_height_m: int = Query(100, ge=10, le=200),
    rotor_diameter_m: int = Query(100, ge=20, le=200),
    rated_power_kw: int = Query(None, ge=100, le=20000),
):
    """
    Fetch wind data and calculate revenue using market prices.
    """
    try:
        import windlib
        
        # Parse dates
        start = date.fromisoformat(start_date)
        end = date.fromisoformat(end_date)
        
        # Fetch wind data
        df = windlib.get_wind_data(
            lat=latitude,
            lon=longitude,
            start_date=start,
            end_date=end,
            hub_height=hub_height_m
        )
        
        if df.empty:
            raise HTTPException(status_code=404, detail="No wind data available")
        
        # Calculate power output
        df = windlib.calculate_power_output(
            df,
            rotor_diameter=rotor_diameter_m,
            rated_power_kw=rated_power_kw
        )
        
        # Add prices and calculate revenue
        df = windlib.add_prices_and_revenue(df, latitude, longitude)
        
        # Calculate totals
        total_energy_mwh = df['power_kw'].sum() / 1000  # Convert kWh to MWh
        total_revenue = df['revenue'].sum() if 'revenue' in df.columns else 0
        avg_price = df['price_usd_mwh'].mean() if 'price_usd_mwh' in df.columns else 0
        
        # Capacity factor
        if rated_power_kw:
            capacity_factor = df['power_kw'].mean() / rated_power_kw
        else:
            capacity_factor = df['power_kw'].mean() / df['power_kw'].max() if df['power_kw'].max() > 0 else 0
        
        return RevenueDataResponse(
            timestamps=df['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S').tolist(),
            revenues=df['revenue'].tolist() if 'revenue' in df.columns else [0.0] * len(df),
            prices=df['price_usd_mwh'].tolist() if 'price_usd_mwh' in df.columns else [0.0] * len(df),
            power_outputs=df['power_kw'].tolist(),
            total_revenue=float(total_revenue),
            total_energy_mwh=float(total_energy_mwh),
            average_price=float(avg_price),
            capacity_factor=float(capacity_factor),
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating revenue: {str(e)}")
