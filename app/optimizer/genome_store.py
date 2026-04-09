"""
CRUD for harness genomes via Supabase harness_genomes table.
"""
from __future__ import annotations

import logging
from typing import Any

from app.optimizer.genome import HarnessGenome

logger = logging.getLogger(__name__)

TABLE = "harness_genomes"


def save_genome(sb: Any, genome: HarnessGenome) -> None:
    """Insert a genome version row."""
    row = {
        "step_name": genome.step_name,
        "version": genome.version,
        "is_active": False,
        "parent_version": genome.parent_version,
        "manager_prompt": genome.manager_prompt,
        "rubric": genome.rubric,
        "score_threshold": genome.score_threshold,
        "max_retries": genome.max_retries,
        "agent_system_prompt": genome.agent_system_prompt,
        "output_format_prompt": genome.output_format_prompt,
        "optimization_notes": genome.optimization_notes,
        "test_score": genome.test_score,
        "test_details": genome.test_details,
    }
    sb.table(TABLE).insert(row).execute()
    logger.info("Saved genome v%d for %s", genome.version, genome.step_name)


def load_genome(sb: Any, step_name: str, version: int) -> HarnessGenome | None:
    """Load a specific genome version."""
    res = (
        sb.table(TABLE)
        .select("*")
        .eq("step_name", step_name)
        .eq("version", version)
        .limit(1)
        .execute()
    )
    rows = res.data or []
    if not rows:
        return None
    return _row_to_genome(rows[0])


def load_active_genome(sb: Any, step_name: str) -> HarnessGenome | None:
    """Load the currently active genome for a step."""
    res = (
        sb.table(TABLE)
        .select("*")
        .eq("step_name", step_name)
        .eq("is_active", True)
        .limit(1)
        .execute()
    )
    rows = res.data or []
    if not rows:
        return None
    return _row_to_genome(rows[0])


def set_active(sb: Any, step_name: str, version: int) -> None:
    """Deactivate all genomes for this step, then activate the target version."""
    # Deactivate all
    sb.table(TABLE).update({"is_active": False}).eq("step_name", step_name).execute()
    # Activate target
    sb.table(TABLE).update({"is_active": True}).eq("step_name", step_name).eq("version", version).execute()
    logger.info("Activated genome v%d for %s", version, step_name)


def list_versions(sb: Any, step_name: str) -> list[dict[str, Any]]:
    """List all genome versions with summary metadata."""
    res = (
        sb.table(TABLE)
        .select("version, is_active, parent_version, test_score, optimization_notes, created_at")
        .eq("step_name", step_name)
        .order("version", desc=True)
        .execute()
    )
    return res.data or []


def next_version(sb: Any, step_name: str) -> int:
    """Get the next available version number."""
    res = (
        sb.table(TABLE)
        .select("version")
        .eq("step_name", step_name)
        .order("version", desc=True)
        .limit(1)
        .execute()
    )
    rows = res.data or []
    if not rows:
        return 0
    return rows[0]["version"] + 1


def _row_to_genome(row: dict[str, Any]) -> HarnessGenome:
    return HarnessGenome(
        version=row["version"],
        step_name=row["step_name"],
        parent_version=row.get("parent_version"),
        manager_prompt=row.get("manager_prompt", ""),
        rubric=row.get("rubric", []),
        score_threshold=row.get("score_threshold", 0.7),
        max_retries=row.get("max_retries", 2),
        agent_system_prompt=row.get("agent_system_prompt", ""),
        output_format_prompt=row.get("output_format_prompt", ""),
        optimization_notes=row.get("optimization_notes", ""),
        test_score=row.get("test_score"),
        test_details=row.get("test_details", {}),
    )
