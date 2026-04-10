"""
Harness package — two-tier validation + feedback-driven retries for LLM generation.

Public interface:
    from app.harness_pkg import run_with_harness, StepConfig, CheapCheckResult, RubricDimension
    from app.harness_pkg.configs import SURVEY_STEP_CONFIG, get_active_survey_config
    from app.harness_pkg import store  # for direct store access
"""
from app.harness_pkg.engine import (
    AttemptTrace,
    CheapCheckResult,
    DimensionScore,
    HarnessResult,
    HarnessStatus,
    ManagerVerdict,
    RubricDimension,
    StepConfig,
    StepOutput,
    run_with_harness,
)

__all__ = [
    "AttemptTrace",
    "CheapCheckResult",
    "DimensionScore",
    "HarnessResult",
    "HarnessStatus",
    "ManagerVerdict",
    "RubricDimension",
    "StepConfig",
    "StepOutput",
    "run_with_harness",
]
