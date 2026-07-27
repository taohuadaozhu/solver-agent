"""Step-by-step workflow API — LangGraph-powered orchestration."""

import asyncio
import json
import logging
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from langgraph.errors import GraphInterrupt
from langgraph.types import Command

from backend.graph import workflow_graph, build_agent
from backend.agent.executor import execute_solver, load_dataset
from backend.agent.tools import call_tool
from backend.agent.visualizer import generate_chart
from backend.memory.store import (
    create_conversation,
    list_conversations,
    delete_conversation,
    add_message,
    get_messages,
    save_workflow,
    get_workflow,
    list_workflows,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("workflow")

app = FastAPI(title="Solver Agent — Workflow API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Models ─────────────────────────────────────────────────────────────────

class WorkflowRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User's optimization problem description")
    conversation_id: Optional[str] = Field(default=None, description="Conversation ID for multi-turn context")


class StepRequest(BaseModel):
    step: str = Field(..., description="Step key, e.g. STEP_1_PROBLEM")
    query: str = Field(..., min_length=1)
    state_json: Optional[str] = Field(default=None, description="Serialized WorkflowState from previous step")


class ExecuteRequest(BaseModel):
    algorithm: str
    dataset_name: str
    parameters: Optional[dict] = Field(default_factory=dict)


class ContinueRequest(BaseModel):
    query: str = Field(..., min_length=1)
    selected_dataset: str = Field(..., min_length=1)
    state_json: str = Field(..., description="Serialized WorkflowState from /workflow/start")
    skip_steps: list = Field(default_factory=list, description="Steps to auto-fill from user input")


class SaveMessageRequest(BaseModel):
    conversation_id: str
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str = Field(..., min_length=1)


class SaveWorkflowRequest(BaseModel):
    conversation_id: str
    problem: str
    state_json: str
    step_outputs: dict = Field(default_factory=dict)
    status: str = "in_progress"


class ConversationCreateRequest(BaseModel):
    title: str = ""


# ── Routes ─────────────────────────────────────────────────────────────────

@app.post("/workflow/run")
async def run_full_workflow(req: WorkflowRequest):
    """Run all 7 steps sequentially and return the complete state."""
    start = time.perf_counter()
    logger.info("Starting full workflow for: %.80s...", req.query)

    state = await run_workflow(req.query)
    elapsed = (time.perf_counter() - start) * 1000

    return {
        "state": state.to_dict(),
        "elapsed_ms": round(elapsed, 1),
    }


@app.post("/workflow/step")
async def run_single_step(req: StepRequest):
    """Run a single workflow step. Client drives the state machine."""
    start = time.perf_counter()

    # Reconstruct state from previous step
    if req.state_json:
        state_data = json.loads(req.state_json)
        state = WorkflowState(**{k: v for k, v in state_data.items() if k in WorkflowState.__dataclass_fields__})
    else:
        state = WorkflowState()

    try:
        response = await run_step(req.step, state, req.query)
    except Exception:
        logger.exception("Step %s failed", req.step)
        raise HTTPException(status_code=502, detail=f"Step {req.step} failed")

    elapsed = (time.perf_counter() - start) * 1000

    current_idx = WORKFLOW_STEPS.index(req.step) + 1 if req.step in WORKFLOW_STEPS else 0

    return {
        "step": req.step,
        "step_index": current_idx,
        "total_steps": len(WORKFLOW_STEPS),
        "response": response,
        "state": state.to_dict(),
        "elapsed_ms": round(elapsed, 1),
    }


@app.post("/workflow/execute")
async def execute_algorithm(req: ExecuteRequest):
    """Execute an algorithm on a dataset (with tool calling support)."""
    logger.info("Executing %s on %s", req.algorithm, req.dataset_name)

    loop = asyncio.get_event_loop()
    output = await loop.run_in_executor(
        None,
        execute_solver,
        req.algorithm,
        req.dataset_name,
        req.parameters,
    )

    result = output.get("execution", output)
    validation = output.get("validation", {})

    chart = None
    if result.get("convergence_curve"):
        chart = generate_chart(
            "convergence",
            f"{req.algorithm} — Convergence",
            {"convergence_curve": result["convergence_curve"]},
        )

    return {
        "execution": result,
        "validation": validation,
        "chart": chart,
    }


@app.post("/workflow/tool-call")
async def handle_tool_call(req: dict):
    """Generic tool-call endpoint — dispatch to executor or visualizer."""
    tool_name = req.get("name", "")
    arguments = req.get("arguments", {})

    if tool_name == "generate_chart":
        result = generate_chart(
            arguments.get("chart_type", "bar_chart"),
            arguments.get("title", "Chart"),
            arguments.get("data", {}),
        )
        return {"success": True, "result": result}

    result = call_tool(tool_name, arguments)
    return result


# ── Workflow: LangGraph-based 7-step state machine ─────────────────────────

_WORKFLOW_NODE_STEPS = {
    "step3_variables": "STEP_3_VARIABLE",
    "step4_objectives": "STEP_4_OBJECTIVE",
    "step5_constraints": "STEP_5_CONSTRAINT",
    "step6_classify": "STEP_6_CLASSIFY",
    "step7_algo": "STEP_7_ALGO",
}


@app.post("/workflow/start")
async def start_analysis(req: WorkflowRequest):
    """Run STEP_1 + parallel RAG, then pause for dataset selection via interrupt()."""
    start = time.perf_counter()
    logger.info("Starting graph-based analysis for: %.80s...", req.query)
    thread_id = uuid.uuid4().hex[:12]
    config = {"configurable": {"thread_id": thread_id}}

    try:
        await workflow_graph.ainvoke({"problem": req.query}, config)
    except GraphInterrupt:
        pass  # Expected — graph paused at step2_dataset interrupt
    except Exception:
        logger.exception("Workflow graph start failed")
        raise HTTPException(status_code=502, detail="Analysis failed")

    state = await workflow_graph.aget_state(config)
    cur = state.values if state else {}

    elapsed = (time.perf_counter() - start) * 1000

    return {
        "analysis": cur.get("analysis", ""),
        "datasets": cur.get("datasets", []),
        "algorithms": cur.get("algorithms", []),
        "capabilities": cur.get("capabilities", {}),
        "capabilities_hint": cur.get("capabilities_hint", ""),
        "state": {**cur, "thread_id": thread_id},
        "coverage": cur.get("coverage", {}),
        "skip_steps": [s.replace("step3_variables", "STEP_3_VARIABLE")
                        .replace("step4_objectives", "STEP_4_OBJECTIVE")
                        .replace("step5_constraints", "STEP_5_CONSTRAINT")
                       for s in cur.get("skip_steps", [])],
        "elapsed_ms": round(elapsed, 1),
    }


@app.post("/workflow/stream")
async def stream_continue(req: ContinueRequest):
    """SSE streaming: resume graph from dataset selection, stream LLM tokens."""
    try:
        state_data = json.loads(req.state_json)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid state_json")

    thread_id = state_data.get("thread_id", uuid.uuid4().hex[:12])
    config = {"configurable": {"thread_id": thread_id}}

    async def event_generator():
        current_step = None
        step_buffer = []

        try:
            async for event in workflow_graph.astream_events(
                Command(resume=req.selected_dataset),
                config, version="v2",
            ):
                kind = event["event"]
                name = event.get("name", "")

                if kind == "on_chain_start" and name in _WORKFLOW_NODE_STEPS:
                    current_step = _WORKFLOW_NODE_STEPS[name]
                    step_buffer = []
                    yield _sse_str("step_start", {"step": current_step})

                elif kind == "on_chat_model_stream" and current_step:
                    chunk = event["data"]["chunk"]
                    if chunk.content:
                        step_buffer.append(chunk.content)
                        yield _sse_str("step_chunk", {"step": current_step, "text": chunk.content})

                elif kind == "on_chain_end" and name in _WORKFLOW_NODE_STEPS:
                    output = "".join(step_buffer)
                    yield _sse_str("step_done", {"step": current_step, "output": output})
                    current_step = None

            # Get final state
            final_state = await workflow_graph.aget_state(config)
            if final_state and final_state.values:
                yield _sse_str("done", {"state": final_state.values})

        except GraphInterrupt:
            # Graph paused again (objectives/constraints interrupt — not used in current frontend flow)
            pass
        except Exception as e:
            logger.exception("Workflow stream failed")
            yield _sse_str("error", {"error": str(e)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.post("/workflow/continue")
async def continue_analysis(req: ContinueRequest):
    """Non-streaming continue — delegates to graph (kept for backward compat)."""
    try:
        state_data = json.loads(req.state_json)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid state_json")

    thread_id = state_data.get("thread_id", uuid.uuid4().hex[:12])
    config = {"configurable": {"thread_id": thread_id}}

    try:
        result = await workflow_graph.ainvoke(Command(resume=req.selected_dataset), config)
        return {"state": result, "elapsed_ms": 0}
    except Exception:
        logger.exception("Continue workflow failed")
        raise HTTPException(status_code=502, detail="Workflow continuation failed")


@app.post("/workflow/auto-continue")
async def auto_continue_analysis(req: ContinueRequest):
    """Redirect to /workflow/stream (backward compat)."""
    return await stream_continue(req)


def _sse_str(event: str, data: dict) -> str:
    """Format a dict as an SSE message string."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# ── Agent Mode: LangGraph create_react_agent ─────────────────────────────────

@app.post("/workflow/agent-stream")
async def agent_stream(req: WorkflowRequest):
    """SSE streaming agent loop powered by LangGraph's create_react_agent."""

    conv_id = req.conversation_id or uuid.uuid4().hex[:12]
    config = {"configurable": {"thread_id": conv_id}}

    agent = build_agent()

    async def event_generator():
        round_num = 0
        charts = []
        pending_tool_starts = []

        try:
            async for event in agent.astream_events(
                {"messages": [{"role": "user", "content": req.query}]},
                config, version="v2",
            ):
                kind = event["event"]
                name = event.get("name", "")

                # --- Thinking round start ---
                if kind == "on_chat_model_start":
                    round_num += 1
                    yield _sse_str("thinking", {"round": round_num})

                # --- Tool calls (batch all starts in a round) ---
                elif kind == "on_tool_start":
                    pending_tool_starts.append({
                        "id": event["run_id"],
                        "name": name,
                        "arguments": event["data"].get("input", {}),
                    })

                # --- Tool results (first end flushes buffered starts) ---
                elif kind == "on_tool_end":
                    if pending_tool_starts:
                        yield _sse_str("tool_calls", {"calls": pending_tool_starts})
                        pending_tool_starts = []

                    output = event["data"].get("output", "")
                    try:
                        result = json.loads(output) if isinstance(output, str) else output
                    except (json.JSONDecodeError, TypeError):
                        result = {"output": str(output)}

                    yield _sse_str("tool_result", {
                        "tool_call_id": event["run_id"],
                        "name": name,
                        "result": result,
                    })

                    # Track charts
                    if name == "generate_chart":
                        try:
                            cd = json.loads(output) if isinstance(output, str) else output
                            if cd.get("type"):
                                charts.append(cd)
                        except Exception:
                            pass

                # --- Agent finished ---
                elif kind == "on_chain_end" and name == "agent":
                    state = await agent.aget_state(config)
                    if state and state.values:
                        messages = state.values.get("messages", [])
                        last_msg = messages[-1] if messages else None
                        final_text = last_msg.content if last_msg and hasattr(last_msg, 'content') else ""
                        yield _sse_str("llm_response", {"content": final_text})
                    yield _sse_str("done", {"charts": charts})

        except Exception as e:
            logger.exception("Agent stream failed")
            yield _sse_str("error", {"error": str(e)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ── Memory / Conversation ─────────────────────────────────────────────────

@app.post("/memory/conversations")
async def new_conversation(req: ConversationCreateRequest):
    conv_id = create_conversation(req.title)
    return {"conversation_id": conv_id}


@app.get("/memory/conversations")
async def get_conversations():
    return {"conversations": list_conversations()}


@app.delete("/memory/conversations/{conv_id}")
async def remove_conversation(conv_id: str):
    delete_conversation(conv_id)
    return {"status": "deleted"}


@app.post("/memory/messages")
async def save_message(req: SaveMessageRequest):
    add_message(req.conversation_id, req.role, req.content)
    return {"status": "saved"}


@app.get("/memory/conversations/{conv_id}/messages")
async def load_messages(conv_id: str):
    return {"messages": get_messages(conv_id)}


# ── Memory / Workflow ─────────────────────────────────────────────────────


@app.post("/memory/workflows")
async def persist_workflow(req: SaveWorkflowRequest):
    state = json.loads(req.state_json)
    wf_id = save_workflow(req.conversation_id, req.problem, state, req.step_outputs, req.status)
    return {"workflow_id": wf_id}


@app.get("/memory/workflows/{wf_id}")
async def load_workflow(wf_id: str):
    wf = get_workflow(wf_id)
    if not wf:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return wf


@app.get("/memory/conversations/{conv_id}/workflows")
async def conversation_workflows(conv_id: str):
    return {"workflows": list_workflows(conv_id)}


# ── Health ─────────────────────────────────────────────────────────────────

@app.get("/workflow/health")
async def health():
    return {"status": "ok", "version": "0.5.0", "engine": "langgraph"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.api.workflow:app", host="0.0.0.0", port=8001, reload=True)
