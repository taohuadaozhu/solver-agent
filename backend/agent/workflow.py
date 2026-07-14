"""State machine orchestrator — drives the 7-step optimization workflow."""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from backend.llm.openai_client import chat_completion
from backend.llm.prompt import (
    STEP_PROMPTS,
    step1_handler,
    step2_handler,
    step3_handler,
    step4_handler,
    step5_handler,
    step6_handler,
    step7_handler,
)
from backend.rag.retriever import semantic_search

logger = logging.getLogger(__name__)

WORKFLOW_STEPS = [
    "STEP_1_PROBLEM",
    "STEP_2_DATASET",
    "STEP_3_VARIABLE",
    "STEP_4_OBJECTIVE",
    "STEP_5_CONSTRAINT",
    "STEP_6_CLASSIFY",
    "STEP_7_ALGO",
]

RAG_STEPS = {"STEP_2_DATASET", "STEP_7_ALGO"}


@dataclass
class WorkflowState:
    problem: str = ""
    dataset_type: str = ""
    dataset_name: str = ""
    variables: List[str] = field(default_factory=list)
    objectives: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    classification: str = ""
    recommended_algorithm: str = ""
    rag_docs: List[Dict] = field(default_factory=list)
    step_results: Dict[str, str] = field(default_factory=dict)
    execution_result: Optional[Dict[str, Any]] = None
    charts: List[Dict[str, Any]] = field(default_factory=list)

    def to_context(self) -> Dict[str, str]:
        return {
            "problem": self.problem,
            "problem_summary": self.step_results.get("STEP_1_PROBLEM", ""),
            "dataset_type": self.dataset_type,
            "dataset": self.dataset_name,
            "variables": ", ".join(self.variables),
            "objectives": ", ".join(self.objectives),
            "constraints": ", ".join(self.constraints),
            "classification": self.classification,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "problem": self.problem,
            "dataset_type": self.dataset_type,
            "dataset_name": self.dataset_name,
            "variables": self.variables,
            "objectives": self.objectives,
            "constraints": self.constraints,
            "classification": self.classification,
            "recommended_algorithm": self.recommended_algorithm,
            "rag_docs": self.rag_docs,
            "step_results": self.step_results,
            "execution_result": self.execution_result,
            "charts": self.charts,
        }


async def rag_search(query: str, top_k: int = 3, doc_type: Optional[str] = None) -> List[Dict]:
    """Run semantic search and return simplified results."""
    try:
        results = await semantic_search(query, top_k=top_k, doc_type=doc_type)
        return [
            {
                "name": r["name"],
                "type": r["type"],
                "summary": r["summary"],
                "similarity": round(r["similarity"], 4),
            }
            for r in results
        ]
    except Exception:
        logger.exception("RAG search failed for query=%s", query)
        return []


async def run_step(
    step: str,
    state: WorkflowState,
    user_query: str,
) -> str:
    """Execute a single workflow step: call LLM with step prompt + context, optionally RAG."""
    logger.info("Running %s", step)

    # Run RAG for data-dependent steps
    if step == "STEP_2_DATASET" and state.problem:
        docs = await rag_search(f"dataset for {state.problem}", top_k=3, doc_type="datasets")
        state.rag_docs.extend(docs)
        state.dataset_name = docs[0]["name"] if docs else ""
    if step == "STEP_7_ALGO" and state.classification:
        docs = await rag_search(
            f"{state.classification} {', '.join(state.objectives)}",
            top_k=5,
            doc_type="algorithms",
        )
        # Filter out example/tutorial directories
        docs = [d for d in docs if "example" not in d.get("name", "").lower()]
        state.rag_docs.extend(docs)
        state.recommended_algorithm = docs[0]["name"] if docs else ""

    # Build prompt with accumulated context
    context = state.to_context()
    if step == "STEP_1_PROBLEM":
        context["problem"] = user_query

    handler_map = {
        "STEP_1_PROBLEM": step1_handler,
        "STEP_2_DATASET": step2_handler,
        "STEP_3_VARIABLE": step3_handler,
        "STEP_4_OBJECTIVE": step4_handler,
        "STEP_5_CONSTRAINT": step5_handler,
        "STEP_6_CLASSIFY": step6_handler,
        "STEP_7_ALGO": step7_handler,
    }

    handler = handler_map[step]
    prompt = handler(context)

    response = await chat_completion(
        prompt,
        system=(
            "You are an optimization expert. Answer concisely in Chinese. "
            "For each step, extract structured information from the user's problem. "
            "Respond with clear, actionable output. Use bullet points when listing items."
        ),
    )

    state.step_results[step] = response

    # Parse structured data from LLM responses
    _parse_step_output(step, response, state)

    return response


def _parse_step_output(step: str, response: str, state: WorkflowState) -> None:
    """Extract structured fields from LLM responses where possible."""
    if step == "STEP_3_VARIABLE":
        # Extract variable names from bullet points
        lines = [l.strip("-*• ").strip() for l in response.split("\n") if l.strip().startswith(("-", "*", "•", "1", "2", "3"))]
        if lines:
            state.variables = lines[:6]
    elif step == "STEP_4_OBJECTIVE":
        lines = [l.strip("-*• ").strip() for l in response.split("\n") if l.strip().startswith(("-", "*", "•", "1", "2", "3"))]
        if lines:
            state.objectives = lines[:5]
    elif step == "STEP_5_CONSTRAINT":
        lines = [l.strip("-*• ").strip() for l in response.split("\n") if l.strip().startswith(("-", "*", "•", "1", "2", "3"))]
        if lines:
            state.constraints = lines[:8]
    elif step == "STEP_6_CLASSIFY":
        state.classification = response.strip().split("\n")[0][:120]


async def run_workflow(user_query: str) -> WorkflowState:
    """Run the full 7-step workflow and return the final state."""
    state = WorkflowState(problem=user_query)

    for step in WORKFLOW_STEPS:
        await run_step(step, state, user_query)

    return state


def _parse_coverage(analysis: str) -> dict:
    """Extract the coverage JSON block from a STEP_1 LLM response."""
    import re
    match = re.search(r'```coverage\s*\n(\{.*?\})\s*\n```', analysis, re.DOTALL)
    if not match:
        return {"variables": False, "objectives": False, "constraints": False}
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return {"variables": False, "objectives": False, "constraints": False}


async def start_workflow(user_query: str) -> dict:
    """Run STEP_1 only, then search datasets via RAG. Returns analysis + dataset options + skip info."""
    state = WorkflowState(problem=user_query)

    # STEP_1: Problem understanding
    analysis = await run_step("STEP_1_PROBLEM", state, user_query)
    coverage = _parse_coverage(analysis)

    # Clean the coverage block from the displayed analysis
    import re
    clean_analysis = re.sub(r'```coverage\s*\n\{.*?\}\s*\n```\n?', '', analysis, flags=re.DOTALL).strip()

    # RAG search for dataset options
    dataset_docs = await rag_search(f"dataset for {user_query}", top_k=10, doc_type="datasets")

    # Determine which steps are pre-covered
    skip_steps = []
    if coverage.get("variables"):
        skip_steps.append("STEP_3_VARIABLE")
    if coverage.get("objectives"):
        skip_steps.append("STEP_4_OBJECTIVE")
    if coverage.get("constraints"):
        skip_steps.append("STEP_5_CONSTRAINT")

    return {
        "analysis": clean_analysis,
        "datasets": dataset_docs,
        "state": state.to_dict(),
        "coverage": coverage,
        "skip_steps": skip_steps,
    }


async def continue_workflow(state_dict: dict, selected_dataset: str, user_query: str) -> WorkflowState:
    """Continue workflow from STEP_2 with a user-selected dataset, then run through STEP_7."""
    state = WorkflowState(
        problem=state_dict.get("problem", user_query),
        dataset_name=selected_dataset,
        step_results=state_dict.get("step_results", {}),
    )

    for step in WORKFLOW_STEPS:
        if step == "STEP_1_PROBLEM":
            continue  # Already done
        if step == "STEP_2_DATASET":
            # Use user-selected dataset instead of RAG
            state.step_results[step] = f"User selected dataset: {selected_dataset}"
            continue
        await run_step(step, state, user_query)

    return state
