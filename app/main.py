"""
Agent Service
-------------
Stateful service for agents, workflows, surveys, analysis,
ingest pipelines, context generation, and panel participants.

Run with:
    uvicorn app.main:app --host 0.0.0.0 --port 8003
"""
from __future__ import annotations

import logging
import os

import dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.middleware.rate_limiter import RateLimiterMiddleware

dotenv.load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)

from app.router import agent_router, persona_router, enrich_router, survey_router
from app.routers.ingest_router import router as ingest_router
from app.routers.context_router import router as context_router
from app.routers.search_router import router as search_router
from app.routers.panel_router import router as panel_router
from app.routers.admin_router import router as admin_ops_router
from app.harness_pkg.router import router as harness_router
from app.routers.optimizer_router import router as optimizer_router
from app.routers.insights_router import router as insights_router
from app.routers.form_router import router as form_router
from app.routers.kg_router import router as kg_router

app = FastAPI(
    title="Valid Agent Service",
    description="Agents, surveys, ingest pipelines, context generation, RAG, and panel participants.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RateLimiterMiddleware)

# Existing agent routers
app.include_router(agent_router)
app.include_router(persona_router)
app.include_router(enrich_router)
app.include_router(survey_router)

# Moved from core API
app.include_router(ingest_router)
app.include_router(context_router)
app.include_router(search_router)
app.include_router(panel_router)
app.include_router(admin_ops_router)
app.include_router(harness_router)
app.include_router(optimizer_router)
app.include_router(insights_router)
app.include_router(form_router)
app.include_router(kg_router)


@app.get("/health", tags=["health"])
def health():
    return {
        "service": "valid-agents",
        "version": "2.0.0",
        "status": "ok",
        "core_api": os.environ.get("CORE_API_URL", "http://localhost:8000"),
    }
