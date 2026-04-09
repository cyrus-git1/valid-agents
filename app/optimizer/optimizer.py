"""
Meta-Harness optimizer — uses Claude Code as the proposer agent.

Following Yoonho Lee's Meta-Harness approach:
1. Dump all traces, genome, and source code to a workspace directory
2. Spawn Claude Code with full filesystem access to that workspace
3. Claude Code reads traces (grep, cat), diagnoses failures, proposes changes
4. The harness tests the proposed genome, writes results back
5. Repeat — Claude Code reads prior proposals + test results for next iteration

The key insight: no trace compression. Claude Code gets raw files and
self-directs what to read (41% source code, 40% traces in Meta-Harness study).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

from app.optimizer.genome import HarnessGenome, baseline_genome
from app.optimizer.genome_store import (
    load_active_genome,
    load_genome,
    next_version,
    save_genome,
    set_active,
)
from app.optimizer.models import OptimizerConfig, OptimizationRun
from app.optimizer.prompt import OPTIMIZER_SYSTEM_PROMPT, compute_aggregate_stats
from app.optimizer.test_runner import build_test_set, evaluate_genome
from app.optimizer.trace_persistence import load_all_full_traces

logger = logging.getLogger(__name__)

WORKSPACE_ROOT = Path(os.environ.get("HARNESS_WORKSPACE_DIR", "./data/optimizer_workspace"))

# JSON schema for the structured output Claude Code returns
PROPOSAL_SCHEMA = {
    "type": "object",
    "properties": {
        "diagnosis": {
            "type": "string",
            "description": "What patterns were identified in the traces",
        },
        "hypothesis": {
            "type": "string",
            "description": "Why this change should help",
        },
        "changes": {
            "type": "object",
            "description": "Partial update: genome field name -> new value. Only include fields being changed.",
            "properties": {
                "manager_prompt": {"type": "string"},
                "rubric": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "weight": {"type": "number"},
                            "description": {"type": "string"},
                        },
                        "required": ["name", "weight", "description"],
                    },
                },
                "score_threshold": {"type": "number"},
                "max_retries": {"type": "integer"},
                "agent_system_prompt": {"type": "string"},
                "output_format_prompt": {"type": "string"},
            },
        },
        "expected_impact": {
            "type": "string",
            "description": "Predicted effect on scores",
        },
    },
    "required": ["diagnosis", "hypothesis", "changes", "expected_impact"],
}


# ── Workspace management ────────────────────────────────────────────────────


def _setup_workspace(
    run_id: str,
    genome: HarnessGenome,
    traces: list[dict[str, Any]],
    stats: dict[str, Any],
    source_files: dict[str, str],
) -> Path:
    """Create the workspace directory with all files Claude Code will read.

    Structure:
        workspace/{run_id}/
            genome/
                current.json          — full current genome
            traces/
                summary_stats.json    — aggregate statistics
                trace_001.json        — full raw trace
                trace_002.json
                ...
            source/
                survey_prompts.py     — current prompt source
                harness_configs.py    — current config source
                ...
            iterations/
                (empty — Claude Code writes proposals here, test results go here)
    """
    ws = WORKSPACE_ROOT / run_id
    ws.mkdir(parents=True, exist_ok=True)

    # Genome
    (ws / "genome").mkdir(exist_ok=True)
    (ws / "genome" / "current.json").write_text(
        json.dumps(genome.model_dump(), indent=2, default=str)
    )

    # Traces — individual files so Claude Code can grep across them
    traces_dir = ws / "traces"
    traces_dir.mkdir(exist_ok=True)
    (traces_dir / "summary_stats.json").write_text(
        json.dumps(stats, indent=2, default=str)
    )
    for i, trace in enumerate(traces):
        (traces_dir / f"trace_{i + 1:03d}.json").write_text(
            json.dumps(trace, indent=2, default=str)
        )

    # Source files
    source_dir = ws / "source"
    source_dir.mkdir(exist_ok=True)
    for name, content in source_files.items():
        (source_dir / name).write_text(content)

    # Iterations dir for proposals and test results
    (ws / "iterations").mkdir(exist_ok=True)

    return ws


def _collect_source_files() -> dict[str, str]:
    """Read the relevant source files that Claude Code might want to inspect."""
    base = Path(__file__).resolve().parent.parent  # app/
    files = {}
    paths = [
        "prompts/survey_prompts.py",
        "prompts/harness_prompts.py",
        "harness_configs.py",
        "harness.py",
        "workflows/survey_workflow.py",
    ]
    for p in paths:
        full = base / p
        if full.exists():
            files[p.replace("/", "_")] = full.read_text()
    return files


def _write_iteration_result(
    ws: Path,
    iteration: int,
    proposal_idx: int,
    genome: HarnessGenome,
    test_result: dict[str, Any],
) -> None:
    """Write test results back to the workspace so Claude Code can read them next iteration."""
    iter_dir = ws / "iterations" / f"iter_{iteration:02d}"
    iter_dir.mkdir(parents=True, exist_ok=True)

    result_file = iter_dir / f"candidate_{proposal_idx:02d}_result.json"
    result_file.write_text(json.dumps({
        "genome_version": genome.version,
        "optimization_notes": genome.optimization_notes,
        "test_score": genome.test_score,
        "test_details": genome.test_details,
        "changes_applied": {
            k: v for k, v in genome.model_dump().items()
            if k not in ("version", "step_name", "parent_version", "optimization_notes", "test_score", "test_details")
        },
    }, indent=2, default=str))


# ── Claude Code invocation ──────────────────────────────────────────────────


async def _invoke_claude_code(
    workspace: Path,
    iteration: int,
    system_prompt: str,
    task_prompt: str,
) -> dict[str, Any] | None:
    """Spawn Claude Code to analyze the workspace and propose a genome change.

    Uses the Claude Agent SDK (claude_agent_sdk) with:
    - cwd set to the workspace directory
    - Full file read access (Read, Grep, Glob)
    - Structured JSON output matching PROPOSAL_SCHEMA
    - System prompt establishing the optimizer persona
    """
    try:
        from claude_agent_sdk import query, ClaudeAgentOptions, ResultMessage
    except ImportError:
        raise RuntimeError(
            "claude-agent-sdk is required for the optimizer. "
            "Install with: pip install claude-agent-sdk"
        )

    options = ClaudeAgentOptions(
        cwd=str(workspace),
        system_prompt=system_prompt,
        allowed_tools=["Read", "Grep", "Glob", "Bash(ls *)", "Bash(wc *)", "Bash(find *)", "Bash(head *)", "Bash(tail *)"],
        output_format={
            "type": "json_schema",
            "schema": PROPOSAL_SCHEMA,
        },
        max_turns=25,
    )

    result_data = None

    async for message in query(prompt=task_prompt, options=options):
        if isinstance(message, ResultMessage):
            if message.structured_output:
                result_data = message.structured_output
            logger.info(
                "Claude Code optimizer iter %d: %d turns, %.1fs, $%.4f",
                iteration,
                message.num_turns or 0,
                (message.duration_ms or 0) / 1000,
                message.total_cost_usd or 0,
            )

    return result_data


def _build_task_prompt(iteration: int, candidates_requested: int) -> str:
    """Build the task prompt for Claude Code."""
    if iteration == 1:
        return (
            "Analyze the traces and current genome in this workspace.\n\n"
            "1. Read traces/summary_stats.json for aggregate patterns\n"
            "2. Grep through traces/ for low-scoring dimensions and repeated failures\n"
            "3. Read genome/current.json to understand the current configuration\n"
            "4. Read source files in source/ to understand how prompts are structured\n\n"
            "Based on your analysis, propose ONE focused change to improve the harness. "
            "Change 1-3 genome fields. Small, targeted changes are better than rewrites.\n\n"
            "Return your proposal as structured JSON."
        )
    else:
        return (
            f"This is iteration {iteration} of the optimization loop.\n\n"
            "1. Read iterations/ to see what was tried before and how it scored\n"
            "2. Read traces/ for patterns the previous proposals didn't address\n"
            "3. Read genome/current.json for the current best configuration\n\n"
            "Based on what worked and what didn't in prior iterations, "
            "propose a NEW focused change. Don't repeat previous proposals.\n\n"
            "Return your proposal as structured JSON."
        )


# ── Main loop ───────────────────────────────────────────────────────────────


def _apply_proposal(
    current: HarnessGenome,
    proposal: dict[str, Any],
    new_version: int,
) -> HarnessGenome:
    """Apply a proposal's changes to the current genome."""
    data = current.model_dump()
    data["version"] = new_version
    data["parent_version"] = current.version
    data["optimization_notes"] = f"Hypothesis: {proposal.get('hypothesis', '')}"
    data["test_score"] = None
    data["test_details"] = {}

    changes = proposal.get("changes", {})
    for field_name, new_value in changes.items():
        if field_name in data and field_name not in ("version", "step_name", "parent_version"):
            data[field_name] = new_value

    return HarnessGenome(**data)


def run_optimization(
    sb: Any,
    config: OptimizerConfig,
) -> OptimizationRun:
    """Run the Meta-Harness optimization loop.

    For each iteration:
    1. Set up workspace with traces + genome + source
    2. Spawn Claude Code to analyze and propose changes
    3. Test the proposed genome
    4. Write results back to workspace
    5. If improved, promote and continue
    """
    return asyncio.run(_run_optimization_async(sb, config))


async def _run_optimization_async(
    sb: Any,
    config: OptimizerConfig,
) -> OptimizationRun:
    run_id = str(uuid.uuid4())
    run = OptimizationRun(
        run_id=run_id,
        step_name=config.step_name,
        status="running",
    )

    logger.info("Optimizer: starting run %s for %s", run_id, config.step_name)

    # ── Load or create baseline genome ──────────────────────────────────
    current = load_active_genome(sb, config.step_name)
    if current is None:
        current = baseline_genome(config.step_name)
        save_genome(sb, current)
        set_active(sb, config.step_name, current.version)
        logger.info("Optimizer: created baseline genome v0")

    run.starting_version = current.version

    # ── Load traces ─────────────────────────────────────────────────────
    traces = load_all_full_traces(sb, config.step_name, limit=200)
    if len(traces) < 3:
        run.status = "failed"
        run.summary = f"Not enough traces ({len(traces)}). Need at least 3."
        return run

    stats = compute_aggregate_stats(traces)

    # ── Build test set ──────────────────────────────────────────────────
    test_inputs = build_test_set(traces, size=config.test_set_size)
    if len(test_inputs) < 3:
        run.status = "failed"
        run.summary = f"Not enough test inputs ({len(test_inputs)}). Need at least 3."
        return run

    logger.info("Optimizer: %d traces, %d test inputs", len(traces), len(test_inputs))

    # ── Evaluate baseline ───────────────────────────────────────────────
    baseline_result = evaluate_genome(current, test_inputs)
    run.starting_score = baseline_result.avg_composite_score
    best_score = baseline_result.avg_composite_score
    best_genome = current

    logger.info(
        "Optimizer: baseline v%d score=%.3f pass_rate=%.1f%%",
        current.version, baseline_result.avg_composite_score, baseline_result.pass_rate * 100,
    )

    # ── Set up workspace ────────────────────────────────────────────────
    source_files = _collect_source_files()
    ws = _setup_workspace(run_id, current, traces, stats, source_files)
    logger.info("Optimizer: workspace at %s", ws)

    # ── Iteration loop ──────────────────────────────────────────────────
    for iteration in range(1, config.iterations + 1):
        logger.info("Optimizer: iteration %d/%d", iteration, config.iterations)

        # Update current genome in workspace
        (ws / "genome" / "current.json").write_text(
            json.dumps(current.model_dump(), indent=2, default=str)
        )

        # Spawn Claude Code
        task_prompt = _build_task_prompt(iteration, config.candidates_per_iteration)

        try:
            proposal = await _invoke_claude_code(
                workspace=ws,
                iteration=iteration,
                system_prompt=OPTIMIZER_SYSTEM_PROMPT,
                task_prompt=task_prompt,
            )
        except Exception as e:
            logger.warning("Optimizer: Claude Code failed on iteration %d: %s", iteration, e)
            continue

        if not proposal or not proposal.get("changes"):
            logger.info("Optimizer: no proposal on iteration %d", iteration)
            continue

        logger.info(
            "Optimizer: proposal — %s",
            proposal.get("hypothesis", "")[:100],
        )

        # Apply and test
        ver = next_version(sb, config.step_name)
        candidate = _apply_proposal(current, proposal, ver)

        try:
            test_result = evaluate_genome(candidate, test_inputs)
        except Exception as e:
            logger.warning("Optimizer: evaluation failed for v%d: %s", ver, e)
            _write_iteration_result(ws, iteration, 0, candidate, {"error": str(e)})
            continue

        candidate.test_score = test_result.avg_composite_score
        candidate.test_details = {
            "pass_rate": test_result.pass_rate,
            "per_dimension_avg": test_result.per_dimension_avg,
            "per_input_scores": test_result.per_input_scores,
            "diagnosis": proposal.get("diagnosis", ""),
            "hypothesis": proposal.get("hypothesis", ""),
            "expected_impact": proposal.get("expected_impact", ""),
        }
        save_genome(sb, candidate)
        run.versions_tested.append(ver)

        # Write results back to workspace for next iteration
        _write_iteration_result(ws, iteration, 0, candidate, candidate.test_details)

        logger.info(
            "Optimizer: v%d score=%.3f (baseline=%.3f, best=%.3f)",
            ver, test_result.avg_composite_score, run.starting_score, best_score,
        )

        # Promote if improved enough
        if test_result.avg_composite_score > best_score + config.improvement_threshold:
            set_active(sb, config.step_name, candidate.version)
            best_score = test_result.avg_composite_score
            best_genome = candidate
            current = candidate

            logger.info(
                "Optimizer: promoted v%d (score=%.3f, +%.3f)",
                candidate.version, test_result.avg_composite_score,
                test_result.avg_composite_score - run.starting_score,
            )
        else:
            logger.info("Optimizer: no improvement on iteration %d", iteration)
            if iteration > 1:
                run.status = "converged"
                break

        run.iterations_run = iteration

    # ── Finalize ────────────────────────────────────────────────────────
    if run.status == "running":
        run.status = "completed"

    run.final_version = best_genome.version
    run.final_score = best_score
    run.iterations_run = run.iterations_run or config.iterations

    improvement = best_score - run.starting_score
    run.summary = (
        f"Optimized from v{run.starting_version} (score={run.starting_score:.3f}) "
        f"to v{run.final_version} (score={run.final_score:.3f}). "
        f"Improvement: {improvement:+.3f}. "
        f"Tested {len(run.versions_tested)} candidates across {run.iterations_run} iterations. "
        f"Workspace: {ws}"
    )

    logger.info("Optimizer: %s", run.summary)
    return run
