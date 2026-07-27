"""State machine orchestrator — drives the 7-step optimization workflow."""

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from backend.llm.openai_client import chat_completion, chat_completion_stream
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
from backend.rag.retriever import semantic_search, hybrid_search_with_rerank

logger = logging.getLogger(__name__)

# Known capability keywords to scan algorithm summaries for
_CAPABILITY_PATTERNS = {
    "objectives": {
        "makespan": ["makespan", "completion time", "完工时间", "completion"],
        "tardiness": ["tardiness", "lateness", "delay", "延迟", "拖期"],
        "cost": ["cost", "成本", "expense", "economic"],
        "throughput": ["throughput", "through-put", "吞吐", "productivity"],
        "distance": ["distance", "路程", "travel", "route length"],
        "energy": ["energy", "能耗", "power", "fuel"],
        "utilization": ["utilization", "利用率", "load balance", "负载"],
        "multi_objective": ["multi-objective", "multi objective", "多目标", "pareto", "nsga"],
    },
    "constraints": {
        "time_windows": ["time window", "时间窗", "time-window", "tw"],
        "precedence": ["precedence", "sequence", "顺序", "先后", "dependency"],
        "capacity": ["capacity", "容量", "load", "weight", "载重"],
        "machine_eligibility": ["machine eligibility", "eligible", "机器约束", "dedicated"],
        "setup_time": ["setup", "换模", "changeover", "preparation"],
        "breakdowns": ["breakdown", "failure", "故障", "stochastic", "uncertain"],
        "no_wait": ["no-wait", "no wait", "零等待", "blocking"],
        "recirculation": ["recirculation", "reentrant", "重入"],
    },
    "features": {
        "large_scale": ["large scale", "大规模", "large-scale", "benchmark"],
        "real_time": ["real-time", "实时", "online", "dynamic"],
        "robustness": ["robust", "鲁棒", "uncertainty", "stochastic"],
    },
}

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


async def rag_search(query: str, top_k: int = 3, doc_type: Optional[str] = None, use_hybrid: bool = True) -> List[Dict]:
    """Run semantic or hybrid search and return simplified results."""
    try:
        if use_hybrid:
            results = await hybrid_search_with_rerank(query, top_k=top_k, doc_type=doc_type)
            return [
                {
                    "name": r["name"],
                    "type": r["type"],
                    "summary": r["summary"],
                    "similarity": r.get("rerank_score") or r.get("similarity", 0),
                }
                for r in results
            ]
        else:
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
    match = re.search(r'```coverage\s*\n(\{.*?\})\s*\n```', analysis, re.DOTALL)
    if not match:
        return {"variables": False, "objectives": False, "constraints": False}
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return {"variables": False, "objectives": False, "constraints": False}


async def _extract_domain_keywords(analysis: str, user_query: str) -> str:
    """Use LLM to extract a focused search query from the problem analysis.

    Returns a short English keyword string optimized for embedding search,
    e.g. "job shop scheduling with setup times and machine eligibility".
    """
    try:
        response = await chat_completion(
            prompt=(
                f"User problem: {user_query}\n\n"
                f"Analysis: {analysis[:1500]}\n\n"
                "Extract 3-8 key technical terms or phrases (in English) that best describe "
                "this optimization problem for searching a knowledge base of datasets and algorithms. "
                "Output only the keywords separated by spaces, no other text."
            ),
            system="You are a technical keyword extractor. Be concise and precise.",
            temperature=0.0,
            max_tokens=80,
        )
        keywords = response.strip()
        logger.info("Extracted domain keywords: %s", keywords)
        return keywords if keywords else user_query
    except Exception:
        logger.warning("Keyword extraction failed, falling back to raw query")
        return user_query


def _analyze_algorithm_capabilities(algorithm_docs: List[Dict]) -> dict:
    """Scan algorithm summaries for capability keywords.

    Returns a structured dict of supported objectives, constraints, and features
    with which algorithms support each.
    """
    capabilities: Dict[str, Dict[str, List[str]]] = {
        "objectives": {},
        "constraints": {},
        "features": {},
    }

    for alg in algorithm_docs:
        name = alg.get("name", "")
        summary = (alg.get("summary", "") + " " + name).lower()

        for category in ["objectives", "constraints", "features"]:
            for cap_key, patterns in _CAPABILITY_PATTERNS[category].items():
                if any(p in summary for p in patterns):
                    if cap_key not in capabilities[category]:
                        capabilities[category][cap_key] = []
                    capabilities[category][cap_key].append(name)

    # Deduplicate algorithm lists
    for category in capabilities:
        for key in capabilities[category]:
            capabilities[category][key] = list(dict.fromkeys(capabilities[category][key]))

    return capabilities


def _format_capabilities_hint(capabilities: dict) -> str:
    """Render capabilities as a human-readable hint string for the frontend."""
    parts = []

    obj_caps = capabilities.get("objectives", {})
    if obj_caps:
        lines = ["**Supported optimization objectives (from available algorithms):**"]
        for cap, algos in sorted(obj_caps.items(), key=lambda x: -len(x[1])):
            label = cap.replace("_", " ").title()
            algo_list = ", ".join(algos[:3])
            lines.append(f"  • {label} — {algo_list}")
        parts.append("\n".join(lines))

    cst_caps = capabilities.get("constraints", {})
    if cst_caps:
        lines = ["**Supported constraint types:**"]
        for cap, algos in sorted(cst_caps.items(), key=lambda x: -len(x[1])):
            label = cap.replace("_", " ").title()
            algo_list = ", ".join(algos[:3])
            lines.append(f"  • {label} — {algo_list}")
        parts.append("\n".join(lines))

    feat_caps = capabilities.get("features", {})
    if feat_caps:
        lines = ["**Algorithm features:**"]
        for cap, algos in sorted(feat_caps.items(), key=lambda x: -len(x[1])):
            label = cap.replace("_", " ").title()
            algo_list = ", ".join(algos[:3])
            lines.append(f"  • {label} — {algo_list}")
        parts.append("\n".join(lines))

    return "\n\n".join(parts) if parts else ""


async def start_workflow(user_query: str) -> dict:
    """Run STEP_1, then analysis-driven parallel RAG for datasets + algorithms.

    Returns analysis, dataset options, algorithm options, capability hints, and skip info
    so the frontend can guide the user with knowledge of what the system actually supports.
    """
    state = WorkflowState(problem=user_query)

    # STEP_1: Problem understanding
    analysis = await run_step("STEP_1_PROBLEM", state, user_query)
    coverage = _parse_coverage(analysis)

    # Clean the coverage block from the displayed analysis
    clean_analysis = re.sub(r'```coverage\s*\n\{.*?\}\s*\n```\n?', '', analysis, flags=re.DOTALL).strip()

    # Extract focused domain keywords from the LLM analysis
    domain_keywords = await _extract_domain_keywords(analysis, user_query)

    # Parallel RAG: datasets + algorithms, both driven by the analysis
    dataset_docs, algorithm_docs = await asyncio.gather(
        rag_search(f"dataset for {domain_keywords}", top_k=10, doc_type="datasets"),
        rag_search(f"algorithm for {domain_keywords}", top_k=10, doc_type="algorithms"),
    )

    # Filter out example/tutorial entries from algorithms
    algorithm_docs = [d for d in algorithm_docs if "example" not in d.get("name", "").lower()]

    # Analyze what the candidate algorithms can actually do
    capabilities = _analyze_algorithm_capabilities(algorithm_docs)
    capabilities_hint = _format_capabilities_hint(capabilities)

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
        "algorithms": algorithm_docs,
        "capabilities": capabilities,
        "capabilities_hint": capabilities_hint,
        "state": state.to_dict(),
        "coverage": coverage,
        "skip_steps": skip_steps,
    }


async def stream_workflow_continue(
    query: str,
    selected_dataset: str,
    state_dict: dict,
    skip_steps: List[str],
    system_prompt: str = "You are an optimization expert. Answer concisely in Chinese.",
):
    """Async generator that yields SSE events for each remaining workflow step."""

    state = WorkflowState(problem=query, dataset_name=selected_dataset,
                          step_results=state_dict.get("step_results", {}))

    # STEP_2 (dataset selected)
    state.step_results["STEP_2_DATASET"] = f"User selected dataset: {selected_dataset}"
    state.dataset_name = selected_dataset
    yield _sse("step_done", {"step": "STEP_2_DATASET", "output": f"Dataset: {selected_dataset}"})

    skip_set = set(skip_steps)

    # Pre-fill skipped steps
    skip_labels = {
        "STEP_3_VARIABLE": "Variables already described in query.",
        "STEP_4_OBJECTIVE": "Objectives already described in query.",
        "STEP_5_CONSTRAINT": "Constraints already described in query.",
    }
    for s in skip_set:
        state.step_results[s] = skip_labels.get(s, "Pre-filled from query.")
        yield _sse("step_done", {"step": s, "output": skip_labels.get(s, ""), "skipped": True})

    # Remaining steps with streaming LLM
    remaining = [s for s in ["STEP_3_VARIABLE", "STEP_4_OBJECTIVE", "STEP_5_CONSTRAINT", "STEP_6_CLASSIFY", "STEP_7_ALGO"]
                 if s not in skip_set]

    handler_map = {
        "STEP_1_PROBLEM": step1_handler,
        "STEP_3_VARIABLE": step3_handler,
        "STEP_4_OBJECTIVE": step4_handler,
        "STEP_5_CONSTRAINT": step5_handler,
        "STEP_6_CLASSIFY": step6_handler,
        "STEP_7_ALGO": step7_handler,
    }

    for step in remaining:
        yield _sse("step_start", {"step": step})

        # Run RAG if needed
        if step == "STEP_7_ALGO" and state.classification:
            docs = await rag_search(
                f"{state.classification} {', '.join(state.objectives)}",
                top_k=5, doc_type="algorithms",
            )
            docs = [d for d in docs if "example" not in d.get("name", "").lower()]
            state.rag_docs.extend(docs)
            if docs:
                state.recommended_algorithm = docs[0]["name"]

        context = state.to_context()
        if step == "STEP_1_PROBLEM":
            context["problem"] = query

        handler = handler_map[step]
        prompt = handler(context)

        # Stream LLM response
        full = []
        async for token in chat_completion_stream(prompt, system=system_prompt):
            full.append(token)
            yield _sse("step_chunk", {"step": step, "text": token})

        output = "".join(full)
        state.step_results[step] = output
        _parse_step_output(step, output, state)

        yield _sse("step_done", {"step": step, "output": output})

    # Done
    yield _sse("done", {"state": state.to_dict()})


def _sse(event: str, data: dict) -> str:
    """Format a dict as an SSE message string."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


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
