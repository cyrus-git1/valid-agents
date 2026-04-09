"""
Data models for the optimization loop.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class GenomeProposal(BaseModel):
    """A proposed change from the optimizer agent."""
    diagnosis: str = Field(description="What trace patterns were identified")
    hypothesis: str = Field(description="Why this change should help")
    changes: dict[str, Any] = Field(description="Partial update: genome field name -> new value")
    expected_impact: str = Field(description="Predicted effect on scores")


class TestResult(BaseModel):
    """Result of evaluating a genome against a test set."""
    genome_version: int
    avg_composite_score: float
    pass_rate: float
    per_dimension_avg: dict[str, float] = Field(default_factory=dict)
    per_input_scores: list[dict[str, Any]] = Field(default_factory=list)


class OptimizerConfig(BaseModel):
    """Configuration for an optimization run."""
    step_name: str = "survey_generation"
    iterations: int = Field(default=5, ge=1, le=20)
    candidates_per_iteration: int = Field(default=3, ge=1, le=10)
    test_set_size: int = Field(default=10, ge=3, le=50)
    improvement_threshold: float = Field(default=0.02, ge=0.0, le=0.2)


class OptimizationRun(BaseModel):
    """Summary of a completed optimization run."""
    run_id: str
    step_name: str
    status: str = "running"                        # running | completed | failed | converged
    iterations_run: int = 0
    starting_version: int = 0
    starting_score: float = 0.0
    final_version: int = 0
    final_score: float = 0.0
    versions_tested: list[int] = Field(default_factory=list)
    summary: str = ""
