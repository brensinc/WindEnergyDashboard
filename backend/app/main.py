from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routers import projects, wind, heatmap, analytics, reports

# Create FastAPI app
app = FastAPI(
    title="Wind Energy Dashboard API",
    description="Backend API for wind energy analysis and project management",
    version="1.0.0",
)

# Configure CORS
settings = get_settings()
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

# Include routers
app.include_router(projects.router)
app.include_router(wind.router)
app.include_router(heatmap.router)
app.include_router(analytics.router)
app.include_router(reports.router)


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
