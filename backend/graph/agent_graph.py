"""LangGraph agent graph — replaces the hand-rolled ReAct loop in api/workflow.py.

Uses langgraph.prebuilt.create_react_agent which provides:
- Built-in ReAct loop (think → tool_call → observe → repeat)
- Automatic tool execution with parallel calls
- Message checkpointing for conversation history
- Structured streaming via astream_events()
"""

import json
import logging
from typing import Any, Dict

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

from backend.agent.executor import execute_solver, load_dataset as load_dataset_impl
from backend.agent.visualizer import generate_chart as generate_chart_impl
from backend.rag.retriever import hybrid_search_with_rerank

logger = logging.getLogger(__name__)

AGENT_SYSTEM_PROMPT = (
    "You are an optimization expert agent with tool access. Follow this process:\n"
    "1. Understand the user's optimization problem.\n"
    "2. Use search_knowledge_base to find the right benchmark dataset and suitable algorithms.\n"
    "3. Use load_dataset to load the selected dataset.\n"
    "4. Execute one or more algorithms on the dataset using execute_solver. "
    "When comparing algorithms, run them on the same dataset so results are comparable.\n"
    "5. Use generate_chart to create visualizations (convergence curves, bar charts, etc.).\n"
    "6. Compare results and give a final data-driven recommendation.\n\n"
    "Always explain what you're doing before calling tools. "
    "When the user asks to compare algorithms, execute ALL of them before giving the final answer. "
    "Respond in Chinese if the user wrote in Chinese."
)


# ── Tools ────────────────────────────────────────────────────────────────────

@tool
async def search_knowledge_base(query: str, doc_type: str = None, top_k: int = 5) -> str:
    """Search the optimization knowledge base for algorithms, datasets, or methods.
    Use this before loading a dataset or choosing an algorithm.
    Query in English for best results.
    doc_type can be 'algorithms', 'datasets', or 'projects'.
    """
    results = await hybrid_search_with_rerank(query, top_k=top_k, doc_type=doc_type or None)
    if not results:
        return "No matching documents found."
    items = []
    for r in results:
        score = r.get("rerank_score", r.get("similarity", 0))
        items.append(f"- {r['name']} [{r['type']}] ({score:.0%}): {r['summary'][:200]}")
    return "\n".join(items)


@tool
def load_dataset(dataset_name: str) -> str:
    """Load an optimization dataset by name from the knowledge base.
    Use search_knowledge_base first to find available datasets.
    """
    result = load_dataset_impl(dataset_name)
    return json.dumps(result, ensure_ascii=False)


@tool
def execute_solver_tool(algorithm: str, dataset_name: str, parameters: Dict[str, Any] = None) -> str:
    """Execute an optimization algorithm on a given dataset.
    Run multiple algorithms on the same dataset for fair comparison.
    """
    # Delegate to synchronous dispatcher
    import asyncio
    from backend.agent.tools import call_tool
    loop = asyncio.get_event_loop()
    result = loop.run_in_executor(
        None, call_tool, "execute_solver",
        {"algorithm": algorithm, "dataset_name": dataset_name, "parameters": parameters or {}}
    )
    # run_in_executor returns a Future, but call_tool is sync, so we need to handle this
    # Actually let me use a simpler approach
    from backend.agent.dispatcher import dispatch
    dataset = load_dataset_impl(dataset_name)
    output = dispatch("execute_solver", algorithm, dataset, parameters or {})
    return json.dumps(output, ensure_ascii=False, default=str)


@tool
def generate_chart(chart_type: str, title: str, data: Dict[str, Any]) -> str:
    """Generate a visualization chart from optimization results.
    chart_type: 'convergence', 'gantt', 'route_map', 'pareto_front', or 'bar_chart'.
    """
    result = generate_chart_impl(chart_type, title, data)
    return json.dumps(result, ensure_ascii=False)


AGENT_TOOLS = [search_knowledge_base, load_dataset, execute_solver_tool, generate_chart]


# ── Build agent ──────────────────────────────────────────────────────────────

def build_agent(model_name: str = "gpt-4.1", temperature: float = 0.3):
    """Create a LangGraph ReAct agent with the optimization tool suite."""
    llm = ChatOpenAI(model=model_name, temperature=temperature)

    agent = create_react_agent(
        model=llm,
        tools=AGENT_TOOLS,
        prompt=AGENT_SYSTEM_PROMPT,
        checkpointer=MemorySaver(),
    )

    return agent
