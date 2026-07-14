"""Step-by-step workflow API — drives the 7-step state machine with tool calling."""

import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.agent.workflow import (
    run_step,
    run_workflow,
    start_workflow,
    continue_workflow,
    stream_workflow_continue,
    WorkflowState,
    WORKFLOW_STEPS,
)
from backend.agent.tools import TOOLS, call_tool
from backend.agent.visualizer import generate_chart
from backend.agent.executor import execute_solver, load_dataset
from backend.llm.openai_client import chat_completion, chat_completion_with_tools
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


@app.post("/workflow/start")
async def start_analysis(req: WorkflowRequest):
    """Run STEP_1 problem understanding + return dataset options for user selection."""
    start = time.perf_counter()
    logger.info("Starting analysis for: %.80s...", req.query)

    try:
        result = await start_workflow(req.query)
    except Exception:
        logger.exception("Start workflow failed")
        raise HTTPException(status_code=502, detail="Analysis failed")

    elapsed = (time.perf_counter() - start) * 1000

    return {
        "analysis": result["analysis"],
        "datasets": result["datasets"],
        "state": result["state"],
        "coverage": result.get("coverage", {}),
        "skip_steps": result.get("skip_steps", []),
        "elapsed_ms": round(elapsed, 1),
    }


@app.post("/workflow/continue")
async def continue_analysis(req: ContinueRequest):
    """Continue workflow from STEP_2 with user-selected dataset, run through STEP_7."""
    start = time.perf_counter()
    logger.info("Continuing workflow with dataset=%s", req.selected_dataset)

    try:
        state_data = json.loads(req.state_json)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid state_json")

    try:
        state = await continue_workflow(state_data, req.selected_dataset, req.query)
    except Exception:
        logger.exception("Continue workflow failed")
        raise HTTPException(status_code=502, detail="Workflow continuation failed")

    elapsed = (time.perf_counter() - start) * 1000

    return {
        "state": state.to_dict(),
        "elapsed_ms": round(elapsed, 1),
    }


@app.post("/workflow/auto-continue")
async def auto_continue_analysis(req: ContinueRequest):
    """Continue workflow from dataset selection, auto-filling pre-covered steps."""
    start = time.perf_counter()
    logger.info("Auto-continue: dataset=%s, skip=%s", req.selected_dataset, req.skip_steps)

    try:
        state_data = json.loads(req.state_json)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid state_json")

    state = WorkflowState(problem=req.query, dataset_name=req.selected_dataset,
                          step_results=state_data.get("step_results", {}))

    # Run STEP_2 first (record dataset choice)
    state.step_results["STEP_2_DATASET"] = f"User selected dataset: {req.selected_dataset}"
    state.dataset_name = req.selected_dataset

    # Auto-fill skipped steps from the original query
    skip_map = {
        "STEP_3_VARIABLE": ("STEP_3_VARIABLE",
            f"Variables extracted from user input: the problem description already specified the decision variables."),
        "STEP_4_OBJECTIVE": ("STEP_4_OBJECTIVE",
            f"Objectives extracted from user input: the user already described what to optimize."),
        "STEP_5_CONSTRAINT": ("STEP_5_CONSTRAINT",
            f"Constraints extracted from user input: constraints were already provided in the problem description."),
    }
    for step_key in req.skip_steps:
        if step_key in skip_map:
            s, msg = skip_map[step_key]
            state.step_results[s] = msg

    # Run remaining non-skipped steps
    all_steps = ["STEP_3_VARIABLE", "STEP_4_OBJECTIVE", "STEP_5_CONSTRAINT", "STEP_6_CLASSIFY", "STEP_7_ALGO"]
    for step in all_steps:
        if step in req.skip_steps:
            continue
        try:
            await run_step(step, state, req.query)
        except Exception as e:
            logger.exception("Step %s failed", step)
            state.step_results[step] = f"Error: {e}"

    elapsed = (time.perf_counter() - start) * 1000

    return {
        "state": state.to_dict(),
        "elapsed_ms": round(elapsed, 1),
    }


# ── SSE Streaming ──────────────────────────────────────────────────────────

@app.post("/workflow/stream")
async def stream_continue(req: ContinueRequest):
    """SSE streaming endpoint — pushes step_start, step_chunk, step_done, done events."""

    try:
        state_data = json.loads(req.state_json)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid state_json")

    async def event_generator():
        try:
            async for event_str in stream_workflow_continue(
                req.query, req.selected_dataset, state_data, req.skip_steps or [],
            ):
                yield event_str
        except Exception as e:
            logger.exception("Stream failed")
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


def _sse_str(event: str, data: dict) -> str:
    """Format a dict as an SSE message string."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# ── Agent Loop (LLM-driven multi-tool calling) ──────────────────────────────

AGENT_SYSTEM_PROMPT = (
    "You are an optimization expert agent with tool access. Follow this process:\n"
    "1. Understand the user's optimization problem.\n"
    "2. Use load_dataset to find the right benchmark dataset.\n"
    "3. Execute one or more algorithms on the dataset using execute_solver. "
    "When comparing algorithms, run them on the same dataset so results are comparable.\n"
    "4. Use generate_chart to create visualizations (convergence curves, bar charts, etc.).\n"
    "5. Compare results and give a final data-driven recommendation.\n\n"
    "Always explain what you're doing before calling tools. "
    "When the user asks to compare algorithms, execute ALL of them before giving the final answer. "
    "Respond in Chinese if the user wrote in Chinese."
)


@app.post("/workflow/agent-stream")
async def agent_stream(req: WorkflowRequest):
    """SSE streaming agent loop: LLM decides tools → execute in parallel → repeat → final answer."""

    async def event_generator():
        messages = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": req.query},
        ]

        charts = []
        max_rounds = 8

        for round_num in range(max_rounds):
            yield _sse_str("thinking", {"round": round_num + 1})

            try:
                msg = await chat_completion_with_tools(messages, TOOLS)
            except Exception as e:
                logger.exception("Agent LLM call failed at round %d", round_num + 1)
                yield _sse_str("error", {"error": f"LLM error: {e}"})
                return

            # No tool calls → final answer
            if not msg.tool_calls:
                yield _sse_str("llm_response", {"content": msg.content or ""})
                yield _sse_str("done", {"charts": charts})
                return

            # Broadcast what tools the LLM wants to call
            calls_info = []
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    args = {}
                calls_info.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": args,
                })
            yield _sse_str("tool_calls", {"calls": calls_info})

            # Append assistant message (may have content + tool_calls)
            messages.append(msg.model_dump())

            # Execute all tool calls in parallel
            async def run_one(tc):
                try:
                    args = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    args = {}
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, call_tool, tc.function.name, args)
                return tc.id, tc.function.name, result

            results = await asyncio.gather(*[run_one(tc) for tc in msg.tool_calls])

            for tool_id, tool_name, result in results:
                yield _sse_str("tool_result", {
                    "tool_call_id": tool_id,
                    "name": tool_name,
                    "result": result,
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "content": json.dumps(result, ensure_ascii=False),
                })

                if tool_name == "generate_chart" and result.get("success"):
                    charts.append(result["result"])

        yield _sse_str("error", {"error": "Reached maximum agent rounds without final answer"})

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
    return {"status": "ok", "version": "0.4.0", "steps": WORKFLOW_STEPS}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.api.workflow:app", host="0.0.0.0", port=8001, reload=True)
