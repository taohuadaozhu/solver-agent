"""LangGraph StateGraph for the 7-step optimization workflow.

Replaces the hand-rolled for-loop state machine in agent/workflow.py with
a declarative graph. Key improvements over the old approach:

- Nodes are isolated async functions instead of a dispatch map
- Conditional edges replace if/else chains for step skipping
- interrupt() replaces manual frontend-backend state passing
- astream_events() provides structured streaming events
- Checkpointer provides automatic state persistence
"""

import asyncio
import json
import logging
import re
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver

from langchain_openai import ChatOpenAI

from backend.llm.openai_client import CHAT_MODEL, CHAT_TEMPERATURE
from backend.llm.prompt import (
    step1_handler,
    step2_handler,
    step3_handler,
    step4_handler,
    step5_handler,
    step6_handler,
    step7_handler,
)
from backend.rag.retriever import hybrid_search_with_rerank
from backend.agent.schema import (
    AlgorithmMetadata,
    ProblemType,
    VariableType,
    ObjectiveType,
    ValidationResult,
    validate_algorithm_for_problem,
    rank_algorithms,
)

logger = logging.getLogger(__name__)

WORKFLOW_SYSTEM_PROMPT = (
    "You are an optimization expert. Answer concisely in Chinese. "
    "For each step, extract structured information from the user's problem. "
    "Respond with clear, actionable output. Use bullet points when listing items."
)

_llm = ChatOpenAI(model=CHAT_MODEL, temperature=CHAT_TEMPERATURE)
_kw_llm = ChatOpenAI(model=CHAT_MODEL, temperature=0.0, max_tokens=80)


class WorkflowState(TypedDict):
    problem: str
    analysis: str
    datasets: List[Dict]
    algorithms: List[Dict]
    capabilities: Dict[str, Dict[str, List[str]]]
    capabilities_hint: str
    selected_dataset: str
    variables: Annotated[List[str], add_messages]
    objectives: Annotated[List[str], add_messages]
    constraints: Annotated[List[str], add_messages]
    classification: str
    recommended_algorithm: str
    validated_algorithms: List[Dict]   # ★ new: ranked + validated candidates
    step_results: Dict[str, str]
    coverage: Dict[str, bool]
    skip_steps: List[str]
    execution_result: Optional[Dict]
    charts: List[Dict]
    error: str


# ── Capability scanning (same logic as before, adapted for TypedDict) ──────

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
    },
    "features": {
        "large_scale": ["large scale", "大规模", "large-scale", "benchmark"],
        "real_time": ["real-time", "实时", "online", "dynamic"],
        "robustness": ["robust", "鲁棒", "uncertainty", "stochastic"],
    },
}


def _analyze_capabilities(algorithm_docs: List[Dict]) -> Dict:
    capabilities: Dict = {"objectives": {}, "constraints": {}, "features": {}}
    for alg in algorithm_docs:
        name = alg.get("name", "")
        summary = (alg.get("summary", "") + " " + name).lower()
        for category in ["objectives", "constraints", "features"]:
            for cap_key, patterns in _CAPABILITY_PATTERNS[category].items():
                if any(p in summary for p in patterns):
                    capabilities[category].setdefault(cap_key, []).append(name)
    for cat in capabilities:
        for key in capabilities[cat]:
            capabilities[cat][key] = list(dict.fromkeys(capabilities[cat][key]))
    return capabilities


def _format_hint(capabilities: Dict) -> str:
    parts = []
    for cat, label in [("objectives", "Supported optimization objectives"),
                       ("constraints", "Supported constraint types"),
                       ("features", "Algorithm features")]:
        caps = capabilities.get(cat, {})
        if caps:
            lines = [f"**{label} (from available algorithms):**"]
            for cap, algos in sorted(caps.items(), key=lambda x: -len(x[1])):
                lines.append(f"  - {cap.replace('_',' ').title()} — {', '.join(algos[:3])}")
            parts.append("\n".join(lines))
    return "\n\n".join(parts) if parts else ""


def _parse_coverage(analysis: str) -> Dict[str, bool]:
    match = re.search(r'```coverage\s*\n(\{.*?\})\s*\n```', analysis, re.DOTALL)
    if not match:
        return {"variables": False, "objectives": False, "constraints": False}
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return {"variables": False, "objectives": False, "constraints": False}


def _parse_variables(response: str) -> List[str]:
    lines = [l.strip("-*• ").strip() for l in response.split("\n")
             if l.strip().startswith(("-", "*", "•", "1", "2", "3"))]
    return lines[:6]


def _parse_objectives(response: str) -> List[str]:
    lines = [l.strip("-*• ").strip() for l in response.split("\n")
             if l.strip().startswith(("-", "*", "•", "1", "2", "3"))]
    return lines[:5]


def _parse_constraints(response: str) -> List[str]:
    lines = [l.strip("-*• ").strip() for l in response.split("\n")
             if l.strip().startswith(("-", "*", "•", "1", "2", "3"))]
    return lines[:8]


# ── Node functions ──────────────────────────────────────────────────────────

async def node_step1_analyze(state: WorkflowState) -> Dict:
    """STEP_1: LLM problem understanding + parallel RAG for datasets & algorithms."""
    problem = state["problem"]

    # LLM analysis via LangChain (tracked by LangGraph for streaming)
    prompt_text = step1_handler({"problem": problem})
    lc_messages = [
        {"role": "system", "content": WORKFLOW_SYSTEM_PROMPT},
        {"role": "user", "content": prompt_text},
    ]
    response = await _llm.ainvoke(lc_messages)
    analysis = response.content
    coverage = _parse_coverage(analysis)
    clean = re.sub(r'```coverage\s*\n\{.*?\}\s*\n```\n?', '', analysis, flags=re.DOTALL).strip()

    # Extract domain keywords (small helper call, non-streamed)
    try:
        kw_msgs = [
            {"role": "system", "content": "You are a technical keyword extractor. Be concise and precise."},
            {"role": "user", "content": (
                f"User problem: {problem}\n\n"
                f"Analysis: {analysis[:1500]}\n\n"
                "Extract 3-8 key technical terms or phrases (in English) that best describe "
                "this optimization problem for searching a knowledge base. "
                "Output only the keywords separated by spaces, no other text."
            )},
        ]
        kw_response = await _kw_llm.ainvoke(kw_msgs)
        keywords = kw_response.content.strip() or problem
    except Exception:
        keywords = problem

    # Parallel RAG
    dataset_docs, algorithm_docs = await asyncio.gather(
        hybrid_search_with_rerank(f"dataset for {keywords}", top_k=10, doc_type="datasets"),
        hybrid_search_with_rerank(f"algorithm for {keywords}", top_k=10, doc_type="algorithms"),
    )
    algorithm_docs = [d for d in algorithm_docs if "example" not in d.get("name", "").lower()]

    capabilities = _analyze_capabilities(algorithm_docs)
    hint = _format_hint(capabilities)

    skip_steps = []
    if coverage.get("variables"):
        skip_steps.append("step3_variables")
    if coverage.get("objectives"):
        skip_steps.append("step4_objectives")
    if coverage.get("constraints"):
        skip_steps.append("step5_constraints")

    logger.info("STEP_1 done: %d datasets, %d algorithms, skip=%s",
                len(dataset_docs), len(algorithm_docs), skip_steps)

    return {
        "analysis": clean,
        "datasets": dataset_docs,
        "algorithms": algorithm_docs,
        "capabilities": capabilities,
        "capabilities_hint": hint,
        "coverage": coverage,
        "skip_steps": skip_steps,
        "step_results": {"STEP_1_PROBLEM": clean},
        "variables": [],
        "objectives": [],
        "constraints": [],
    }


async def node_step2_dataset(state: WorkflowState) -> Dict:
    """STEP_2: Human-in-the-loop — wait for user to select a dataset.

    The graph pauses here. The API layer returns the interrupt payload to the
    frontend, and the frontend resumes with the user's choice.
    """
    choice = interrupt({
        "action": "select_dataset",
        "datasets": state["datasets"],
        "algorithms": state["algorithms"],
        "capabilities_hint": state["capabilities_hint"],
    })
    return {
        "selected_dataset": choice,
        "step_results": {"STEP_2_DATASET": f"User selected dataset: {choice}"},
    }


async def _llm_call(prompt_text: str) -> str:
    """Helper: invoke ChatOpenAI with workflow system prompt."""
    msgs = [
        {"role": "system", "content": WORKFLOW_SYSTEM_PROMPT},
        {"role": "user", "content": prompt_text},
    ]
    resp = await _llm.ainvoke(msgs)
    return resp.content


async def node_step3_variables(state: WorkflowState) -> Dict:
    """STEP_3: LLM derives decision variables from problem + dataset context."""
    context = {
        "problem_summary": state.get("analysis", ""),
        "dataset": state.get("selected_dataset", ""),
    }
    response = await _llm_call(step3_handler(context))
    return {
        "variables": _parse_variables(response),
        "step_results": {"STEP_3_VARIABLE": response},
    }


async def node_step4_objectives(state: WorkflowState) -> Dict:
    """STEP_4: LLM derives objectives; interrupt for user supplement."""
    context = {
        "problem_summary": state.get("analysis", ""),
        "dataset_type": state.get("selected_dataset", ""),
        "variables": ", ".join(state.get("variables", [])),
    }
    response = await _llm_call(step4_handler(context))
    objectives = _parse_objectives(response)

    user_input = interrupt({
        "action": "input_objectives",
        "llm_output": response,
        "parsed": objectives,
        "capabilities_hint": state.get("capabilities_hint", ""),
    })

    if user_input and user_input.strip():
        objectives.append(f"[User] {user_input.strip()}")

    return {
        "objectives": objectives,
        "step_results": {"STEP_4_OBJECTIVE": response},
    }


async def node_step5_constraints(state: WorkflowState) -> Dict:
    """STEP_5: LLM derives constraints; interrupt for user supplement."""
    context = {
        "problem_summary": state.get("analysis", ""),
        "dataset": state.get("selected_dataset", ""),
        "variables": ", ".join(state.get("variables", [])),
        "objectives": ", ".join(state.get("objectives", [])),
    }
    response = await _llm_call(step5_handler(context))
    constraints = _parse_constraints(response)

    user_input = interrupt({
        "action": "input_constraints",
        "llm_output": response,
        "parsed": constraints,
        "capabilities_hint": state.get("capabilities_hint", ""),
    })

    if user_input and user_input.strip():
        constraints.append(f"[User] {user_input.strip()}")

    return {
        "constraints": constraints,
        "step_results": {"STEP_5_CONSTRAINT": response},
    }


async def node_step6_classify(state: WorkflowState) -> Dict:
    """STEP_6: Classify the problem based on accumulated state."""
    context = {
        "variables": ", ".join(state.get("variables", [])),
        "objectives": ", ".join(state.get("objectives", [])),
        "constraints": ", ".join(state.get("constraints", [])),
    }
    response = await _llm_call(step6_handler(context))
    classification = response.strip().split("\n")[0][:120]
    return {
        "classification": classification,
        "step_results": {"STEP_6_CLASSIFY": response},
    }


async def node_step7_algo(state: WorkflowState) -> Dict:
    """STEP_7: RAG search for algorithms + metadata validation + LLM recommendation."""
    classification = state.get("classification", "")
    objectives_list = state.get("objectives", [])
    constraints_list = state.get("constraints", [])
    objective_count = len(objectives_list)
    objectives_str = ", ".join(objectives_list)

    # ── 1. 推断 problem_type 和 variable_type ──
    problem_type = _infer_problem_type(classification)
    variable_type = _infer_variable_type(constraints_list)

    # ── 2. RAG 检索候选算法 ──
    docs = await hybrid_search_with_rerank(
        f"{classification} {objectives_str}",
        top_k=10, doc_type="algorithms",
    )
    docs = [d for d in docs if "example" not in d.get("name", "").lower()]

    # ── 3. 用 schema 校验每个候选 ──
    validated: List[dict] = []
    for doc in docs:
        try:
            meta = AlgorithmMetadata.from_dict(doc.get("chunk_metadata", {}))
        except Exception:
            continue  # skip docs without valid metadata schema

        result = validate_algorithm_for_problem(
            meta,
            problem_type=problem_type,
            variable_type=variable_type,
            objective_count=objective_count,
            user_constraints=constraints_list,
        )
        validated.append({
            "name": meta.name,
            "chinese_name": meta.chinese_name,
            "summary": doc.get("summary", ""),
            "score": max(0, 100 - 20 * len(result.errors) - 5 * len(result.warnings)),
            "passed": result.passed,
            "errors": result.errors,
            "warnings": result.warnings,
            "missing_params": result.missing_params,
            "parameters": [p.to_dict() for p in meta.parameters],
        })

    validated.sort(key=lambda x: x["score"], reverse=True)
    recommended = validated[0]["name"] if validated else (docs[0]["name"] if docs else "")

    logger.info("STEP_7: %d candidates → %d passed validation", len(docs),
                sum(1 for v in validated if v["passed"]))

    # ── 4. LLM 生成推荐文本 ──
    context = {
        "classification": classification,
        "dataset": state.get("selected_dataset", ""),
        "constraints": ", ".join(constraints_list),
    }
    response = await _llm_call(step7_handler(context))

    return {
        "recommended_algorithm": recommended,
        "validated_algorithms": validated,
        "step_results": {"STEP_7_ALGO": response},
    }


def _infer_problem_type(classification: str) -> Optional[ProblemType]:
    """Map classification text to a ProblemType enum."""
    mapping = {
        "vehicle routing": ProblemType.VRP,
        "vrp": ProblemType.VRP, "cvrp": ProblemType.CVRP, "vrptw": ProblemType.VRPTW,
        "tsp": ProblemType.TSP,
        "job shop": ProblemType.JSP, "jsp": ProblemType.JSP,
        "flow shop": ProblemType.FSP, "fsp": ProblemType.FSP,
        "open shop": ProblemType.OSP,
        "scheduling": ProblemType.SCHEDULING,
        "knapsack": ProblemType.KNAPSACK,
        "bin packing": ProblemType.BIN_PACKING,
        "continuous": ProblemType.CONTINUOUS,
    }
    lower = classification.lower()
    for key, pt in mapping.items():
        if key in lower:
            return pt
    return None


def _infer_variable_type(constraints: List[str]) -> Optional[VariableType]:
    """Heuristic: if permutation/sequence/discrete mentioned → permutation else continuous."""
    text = " ".join(constraints).lower()
    if any(w in text for w in ["permutation", "排列", "sequence", "顺序", "route", "路径", "discrete", "离散"]):
        return VariableType.PERMUTATION
    if any(w in text for w in ["binary", "二进制", "0-1"]):
        return VariableType.BINARY
    return VariableType.CONTINUOUS


# ── Conditional routing ─────────────────────────────────────────────────────

def route_after_step3(state: WorkflowState) -> str:
    return "step5_constraints" if "step4_objectives" in state.get("skip_steps", []) else "step4_objectives"


def route_after_step4(state: WorkflowState) -> str:
    return "step6_classify" if "step5_constraints" in state.get("skip_steps", []) else "step5_constraints"


# ── Build graph ─────────────────────────────────────────────────────────────

def build_workflow_graph():
    """Build and compile the 7-step optimization workflow as a StateGraph."""
    builder = StateGraph(WorkflowState)

    builder.add_node("step1_analyze", node_step1_analyze)
    builder.add_node("step2_dataset", node_step2_dataset)
    builder.add_node("step3_variables", node_step3_variables)
    builder.add_node("step4_objectives", node_step4_objectives)
    builder.add_node("step5_constraints", node_step5_constraints)
    builder.add_node("step6_classify", node_step6_classify)
    builder.add_node("step7_algo", node_step7_algo)

    builder.add_edge(START, "step1_analyze")
    builder.add_edge("step1_analyze", "step2_dataset")
    builder.add_conditional_edges(
        "step2_dataset",
        lambda s: "step5_constraints" if "step3_variables" in s.get("skip_steps", [])
                  and "step4_objectives" in s.get("skip_steps", [])
                  else "step4_objectives" if "step3_variables" in s.get("skip_steps", [])
                  else "step3_variables",
        {"step3_variables": "step3_variables",
         "step4_objectives": "step4_objectives",
         "step5_constraints": "step5_constraints"},
    )
    builder.add_conditional_edges("step3_variables", route_after_step3, {
        "step4_objectives": "step4_objectives",
        "step5_constraints": "step5_constraints",
    })
    builder.add_conditional_edges("step4_objectives", route_after_step4, {
        "step5_constraints": "step5_constraints",
        "step6_classify": "step6_classify",
    })
    builder.add_edge("step5_constraints", "step6_classify")
    builder.add_edge("step6_classify", "step7_algo")
    builder.add_edge("step7_algo", END)

    return builder.compile(checkpointer=MemorySaver())


# Singleton instance
workflow_graph = build_workflow_graph()
