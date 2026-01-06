from pydantic import BaseModel
from typing import Optional, List


class WindDataRequest(BaseModel):
    latitude: float
    longitude: float
    start_date: str  # YYYY-MM-DD
    end_date: str    # YYYY-MM-DD
    hub_height_m: int = 100
    rotor_diameter_m: int = 100
    rated_power_kw: Optional[int] = None


class WindDataResponse(BaseModel):
    timestamps: List[str]
    wind_speeds: List[float]
    wind_directions: List[float]
    power_outputs: List[float]
    capacity_factors: List[float]
    metadata: dict


class RevenueDataResponse(BaseModel):
    timestamps: List[str]
    revenues: List[float]
    prices: List[float]
    power_outputs: List[float]
    total_revenue: float
    total_energy_mwh: float
    average_price: float
    capacity_factor: float
