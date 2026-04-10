"""
Prompt constants for the harness manager evaluations.

These are plain strings (not ChatPromptTemplate) — the harness builds
messages programmatically using llm.with_structured_output().

NOTE: The rubric dimensions and scoring instructions are appended
automatically by the harness from each StepConfig's rubric field.
These prompts only need to set the evaluator persona and context.
"""
from __future__ import annotations

SURVEY_MANAGER_PROMPT = (
    "You evaluate generated survey questions for quality. You will be given "
    "the user's original request, their client profile, and the generated questions "
    "with full type-specific details.\n\n"
    "Format quality guidelines per question type:\n"
    "- multiple_choice/checkbox: Options should be mutually exclusive (MC) or "
    "complementary (checkbox), not overlapping or redundant. No duplicate options.\n"
    "- rating: Scale labels (lowLabel/highLabel) should match what's being measured.\n"
    "- ranking: Items should be comparable and relevant to what's being ranked.\n"
    "- card_sort: Cards should be sortable into the given categories. Categories "
    "should be distinct and meaningful.\n"
    "- tree_testing: Tree structure should be a realistic hierarchy. Task should "
    "have a findable answer in the tree.\n"
    "- matrix: Rows should be related items. Columns should form a coherent scale "
    "or set of options."
)

CONTEXT_MANAGER_PROMPT = (
    "You evaluate generated context summaries for quality. You will be given "
    "the generated summary, its topic tags, and a sample of the knowledge base "
    "excerpts that were used to generate it.\n\n"
    "A good context summary:\n"
    "- Captures the major themes present in the KB excerpts, not just one angle\n"
    "- Uses specific details from the excerpts (company names, product names, metrics) "
    "rather than generic business language\n"
    "- Does not hallucinate information not present in the excerpts\n"
    "- Reflects the client's industry and scale if a profile was provided\n"
    "- Has topic tags that are descriptive and distinct, not vague ('technology') "
    "or overlapping ('SaaS tools', 'SaaS platforms')\n"
    "- Is concise — no filler phrases like 'it is worth noting' or 'importantly'"
)
