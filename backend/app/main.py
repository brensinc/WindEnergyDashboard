import logging
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routers import projects, wind, heatmap, analytics, reports

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Log startup initiation
logger.info("STARTUP: Initializing FastAPI application...")

try:
    from app.config import get_settings
    logger.info("STARTUP: Loading configuration...")
    settings = get_settings()
    logger.info(f"STARTUP: Configuration loaded - SUPABASE_URL={settings.supabase_url[:50]}...")
    logger.info(f"STARTUP: FRONTEND_URL={settings.frontend_url}")
except Exception as e:
    logger.error(f"STARTUP ERROR: Failed to load configuration: {e}")
    raise

# Create FastAPI app
app = FastAPI(
    title="Wind Energy Dashboard API",
    description="Backend API for wind energy analysis and project management",
    version="1.0.0",
)

# Configure CORS
logger.info("STARTUP: Configuring CORS...")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        settings.frontend_url,
        "http://localhost:3000",  # Local development
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger.info("STARTUP: CORS configured")

# Include routers
logger.info("STARTUP: Including routers...")
app.include_router(projects.router)
app.include_router(wind.router)
app.include_router(heatmap.router)
app.include_router(analytics.router)
app.include_router(reports.router)
logger.info("STARTUP: All routers included")


@app.get("/")
def root():
    """Root endpoint for health checks."""
    return {"status": "ok"}


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "1.0.0"}


@app.get("/api/prices/isos")
async def get_iso_list():
    """Get list of supported ISO regions."""
    return ["CAISO", "ERCOT", "ISONE", "MISO", "NYISO", "PJM"]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
