from pydantic import BaseModel, Field
from typing import Any, Dict, Optional
from uuid import UUID

class ProjectCreate(BaseModel):
    name: str
    config: Dict[str, Any] = Field(default_factory=dict)

class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    config: Optional[Dict[str, Any]] = None

class ProjectOut(BaseModel):
    id: UUID
    user_id: UUID
    name: str
    config: Dict[str, Any]
