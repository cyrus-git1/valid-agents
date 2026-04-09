"""
Optimizer API — trigger optimization runs, manage genomes, inspect history.
"""
from __future__ import annotations

import threading
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query

from app.supabase_client import get_supabase
from app.optimizer.genome import baseline_genome
from app.optimizer.genome_store import (
    list_versions,
    load_active_genome,
    load_genome,
    save_genome,
    set_active,
)
from app.optimizer.models import OptimizerConfig, OptimizationRun
from app.optimizer.optimizer import run_optimization

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/optimizer", tags=["optimizer"])

# In-memory run status tracking
_runs: Dict[str, OptimizationRun] = {}
_runs_lock = threading.Lock()


# ── Optimization runs ───────────────────────────────────────────────────────


def _run_optimization_background(config: OptimizerConfig):
    """Background task wrapper for run_optimization."""
    sb = get_supabase()
    try:
        result = run_optimization(sb, config)
        with _runs_lock:
            _runs[result.run_id] = result
    except Exception as e:
        logger.exception("Optimization run failed")
        with _runs_lock:
            # Find the run by step_name if we can
            for run in _runs.values():
                if run.status == "running" and run.step_name == config.step_name:
                    run.status = "failed"
                    run.summary = f"Unhandled error: {e}"
                    break


@router.post("/run")
def start_optimization(
    config: OptimizerConfig,
    background_tasks: BackgroundTasks,
):
    """Trigger an optimization run as a background task."""
    # Create a placeholder run
    import uuid
    run_id = str(uuid.uuid4())
    run = OptimizationRun(
        run_id=run_id,
        step_name=config.step_name,
        status="starting",
    )
    with _runs_lock:
        _runs[run_id] = run

    background_tasks.add_task(_run_optimization_background, config)

    return {"run_id": run_id, "status": "started", "step_name": config.step_name}


@router.get("/status/{run_id}")
def get_run_status(run_id: str):
    """Get the status of an optimization run."""
    with _runs_lock:
        run = _runs.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    return run.model_dump()


@router.get("/runs")
def list_runs():
    """List all optimization runs (most recent first)."""
    with _runs_lock:
        runs = list(_runs.values())
    return [r.model_dump() for r in reversed(runs)]


# ── Genome management ───────────────────────────────────────────────────────


@router.get("/genomes/{step_name}")
def list_genome_versions(step_name: str):
    """List all genome versions for a step."""
    sb = get_supabase()
    return list_versions(sb, step_name)


@router.get("/genomes/{step_name}/active")
def get_active(step_name: str):
    """Get the currently active genome for a step."""
    sb = get_supabase()
    genome = load_active_genome(sb, step_name)
    if genome is None:
        return {"active": False, "message": "No active genome. Using hardcoded defaults."}
    return {"active": True, "genome": genome.model_dump()}


@router.get("/genomes/{step_name}/{version}")
def get_genome_version(step_name: str, version: int):
    """Get a specific genome version."""
    sb = get_supabase()
    genome = load_genome(sb, step_name, version)
    if genome is None:
        raise HTTPException(status_code=404, detail=f"Genome v{version} not found for {step_name}")
    return genome.model_dump()


@router.post("/genomes/{step_name}/activate/{version}")
def activate_genome(step_name: str, version: int):
    """Manually activate a genome version (for rollback or manual promotion)."""
    sb = get_supabase()
    genome = load_genome(sb, step_name, version)
    if genome is None:
        raise HTTPException(status_code=404, detail=f"Genome v{version} not found for {step_name}")
    set_active(sb, step_name, version)
    return {"activated": version, "step_name": step_name}


@router.post("/genomes/{step_name}/deactivate")
def deactivate_genome(step_name: str):
    """Deactivate all genomes for a step (revert to hardcoded defaults)."""
    sb = get_supabase()
    sb.table("harness_genomes").update({"is_active": False}).eq("step_name", step_name).execute()
    return {"step_name": step_name, "active": False, "message": "Reverted to hardcoded defaults."}


@router.post("/genomes/{step_name}/baseline")
def create_baseline(step_name: str):
    """Create and save the baseline (v0) genome from hardcoded constants."""
    sb = get_supabase()
    existing = load_genome(sb, step_name, 0)
    if existing:
        return {"message": "Baseline v0 already exists.", "genome": existing.model_dump()}

    genome = baseline_genome(step_name)
    save_genome(sb, genome)
    return {"message": "Created baseline v0.", "genome": genome.model_dump()}


@router.get("/genomes/{step_name}/diff/{v1}/{v2}")
def diff_genomes(step_name: str, v1: int, v2: int):
    """Show field-by-field diff between two genome versions."""
    sb = get_supabase()
    g1 = load_genome(sb, step_name, v1)
    g2 = load_genome(sb, step_name, v2)

    if g1 is None:
        raise HTTPException(status_code=404, detail=f"Genome v{v1} not found")
    if g2 is None:
        raise HTTPException(status_code=404, detail=f"Genome v{v2} not found")

    d1 = g1.model_dump()
    d2 = g2.model_dump()

    diff = {}
    compare_fields = [
        "manager_prompt", "rubric", "score_threshold", "max_retries",
        "agent_system_prompt", "output_format_prompt",
    ]
    for field in compare_fields:
        val1 = d1.get(field)
        val2 = d2.get(field)
        if val1 != val2:
            diff[field] = {"v1": val1, "v2": val2}

    return {
        "step_name": step_name,
        "v1": v1,
        "v2": v2,
        "changed_fields": list(diff.keys()),
        "diff": diff,
    }
