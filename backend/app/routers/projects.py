from fastapi import APIRouter, HTTPException, Query
from typing import List
from datetime import datetime

from app.db.supabase import get_supabase_client
from app.models.project import Project, ProjectCreate, ProjectUpdate, ProjectCacheUpdate

router = APIRouter(prefix="/api/projects", tags=["projects"])


@router.get("", response_model=List[Project])
async def get_projects(user_id: str = Query(...)):
    """Get all projects for a user."""
    supabase = get_supabase_client()
    
    response = supabase.table("projects").select("*").eq("user_id", user_id).execute()
    
    return response.data


@router.get("/{project_id}", response_model=Project)
async def get_project(project_id: str):
    """Get a single project by ID."""
    supabase = get_supabase_client()
    
    response = supabase.table("projects").select("*").eq("id", project_id).single().execute()
    
    if not response.data:
        raise HTTPException(status_code=404, detail="Project not found")
    
    return response.data


@router.post("", response_model=Project)
async def create_project(project: ProjectCreate):
    """Create a new project."""
    supabase = get_supabase_client()
    
    data = project.model_dump()
    data["created_at"] = datetime.utcnow().isoformat()
    data["updated_at"] = datetime.utcnow().isoformat()
    
    response = supabase.table("projects").insert(data).execute()
    
    if not response.data:
        raise HTTPException(status_code=400, detail="Failed to create project")
    
    return response.data[0]


@router.patch("/{project_id}", response_model=Project)
async def update_project(project_id: str, project: ProjectUpdate):
    """Update a project."""
    supabase = get_supabase_client()
    
    # Filter out None values
    data = {k: v for k, v in project.model_dump().items() if v is not None}
    data["updated_at"] = datetime.utcnow().isoformat()
    
    response = supabase.table("projects").update(data).eq("id", project_id).execute()
    
    if not response.data:
        raise HTTPException(status_code=404, detail="Project not found")
    
    return response.data[0]


@router.delete("/{project_id}")
async def delete_project(project_id: str):
    """Delete a project."""
    supabase = get_supabase_client()
    
    response = supabase.table("projects").delete().eq("id", project_id).execute()
    
    if not response.data:
        raise HTTPException(status_code=404, detail="Project not found")
    
    return {"status": "deleted"}


@router.post("/{project_id}/cache", response_model=Project)
async def update_project_cache(project_id: str, cache_data: ProjectCacheUpdate):
    """Update the cached results for a project."""
    supabase = get_supabase_client()
    
    data = cache_data.model_dump()
    data["cached_at"] = datetime.utcnow().isoformat()
    data["updated_at"] = datetime.utcnow().isoformat()
    
    response = supabase.table("projects").update(data).eq("id", project_id).execute()
    
    if not response.data:
        raise HTTPException(status_code=404, detail="Project not found")
    
    return response.data[0]
