"""Shared base models for the agent service."""
from __future__ import annotations

from typing import Any, Dict, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class TenantScoped(BaseModel):
    tenant_id: UUID
    client_id: UUID


class TenantScopedRequest(TenantScoped):
    client_profile: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Client profile: industry, headcount, revenue, persona, demographic, etc.",
    )


class StatusResponse(BaseModel):
    status: str = "complete"
    error: Optional[str] = None
