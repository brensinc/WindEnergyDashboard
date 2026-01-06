from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Supabase
    supabase_url: str
    supabase_service_key: str  # Use service key for backend operations
    
    # Optional API keys for market data
    pjm_api_key: str | None = None
    
    # CORS settings
    frontend_url: str = "http://localhost:3000"
    
    # App settings
    debug: bool = False
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
