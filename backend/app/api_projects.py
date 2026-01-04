from fastapi import APIRouter, Depends, HTTPException
from uuid import UUID

from .auth import get_user
from .db import supabase_admin
from .schemas import ProjectCreate, ProjectUpdate

router = APIRouter(prefix="/projects", tags=["projects"])

@router.get("")
def list_projects(user=Depends(get_user)):
    resp = (
        supabase_admin.table("projects")
        .select("*")
        .eq("user_id", user.id)
        .order("updated_at", desc=True)
        .execute()
    )
    return resp.data

@router.post("")
def create_project(payload: ProjectCreate, user=Depends(get_user)):
    row = {
        "user_id": user.id,
        "name": payload.name,
        "config": payload.config,
    }
    resp = supabase_admin.table("projects").insert(row).execute()
    if not resp.data:
        raise HTTPException(500, "Insert failed")
    return resp.data[0]

@router.get("/{project_id}")
def get_project(project_id: UUID, user=Depends(get_user)):
    resp = (
        supabase_admin.table("projects")
        .select("*")
        .eq("id", str(project_id))
        .eq("user_id", user.id)
        .single()
        .execute()
    )
    if not resp.data:
        raise HTTPException(404, "Not found")
    return resp.data

@router.put("/{project_id}")
def update_project(project_id: UUID, payload: ProjectUpdate, user=Depends(get_user)):
    updates = {k: v for k, v in payload.model_dump().items() if v is not None}
    if not updates:
        raise HTTPException(400, "No updates provided")

    resp = (
        supabase_admin.table("projects")
        .update(updates)
        .eq("id", str(project_id))
        .eq("user_id", user.id)
        .execute()
    )
    if not resp.data:
        raise HTTPException(404, "Not found")
    return resp.data[0]

@router.delete("/{project_id}")
def delete_project(project_id: UUID, user=Depends(get_user)):
    resp = (
        supabase_admin.table("projects")
        .delete()
        .eq("id", str(project_id))
        .eq("user_id", user.id)
        .execute()
    )
    return {"deleted": len(resp.data or [])}
