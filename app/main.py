"""
Agent Service
-------------
Standalone stateless service for agents, workflows, surveys, and analysis.

All data access goes through the Core API via HTTP.

Run with:
    uvicorn app.main:app --host 0.0.0.0 --port 8003
"""
from __future__ import annotations

import logging
import os

import dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

dotenv.load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)

from app.router import agent_router, persona_router, enrich_router, survey_router

app = FastAPI(
    title="Agent Service",
    description="Stateless agents, surveys, and analysis. Calls Core API for all data.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(agent_router)
app.include_router(persona_router)
app.include_router(enrich_router)
app.include_router(survey_router)


@app.get("/health", tags=["health"])
def health():
    return {
        "service": "agents",
        "status": "ok",
        "core_api": os.environ.get("CORE_API_URL", "http://localhost:8000"),
    }
