from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class ProjectBase(BaseModel):
    name: str
    description: Optional[str] = None
    latitude: float
    longitude: float
    hub_height_m: int = 100
    rotor_diameter_m: int = 100
    turbine_model: Optional[str] = None
    rated_power_kw: Optional[int] = None


class ProjectCreate(ProjectBase):
    user_id: str


class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    hub_height_m: Optional[int] = None
    rotor_diameter_m: Optional[int] = None
    turbine_model: Optional[str] = None
    rated_power_kw: Optional[int] = None


class ProjectCacheUpdate(BaseModel):
    cached_annual_revenue: float
    cached_capacity_factor: float
    cached_iso: Optional[str] = None


class Project(ProjectBase):
    id: str
    user_id: str
    cached_annual_revenue: Optional[float] = None
    cached_capacity_factor: Optional[float] = None
    cached_iso: Optional[str] = None
    cached_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
