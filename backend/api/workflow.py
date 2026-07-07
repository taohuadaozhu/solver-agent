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

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.agent.workflow import (
    run_step,
    run_workflow,
    WorkflowState,
    WORKFLOW_STEPS,
)
from backend.agent.tools import TOOLS, call_tool
from backend.agent.visualizer import generate_chart
from backend.agent.executor import execute_solver, load_dataset
from backend.llm.openai_client import chat_completion

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
    result = await loop.run_in_executor(
        None,
        execute_solver,
        req.algorithm,
        req.dataset_name,
        req.parameters,
    )

    chart = None
    if result.get("convergence_curve"):
        chart = generate_chart(
            "convergence",
            f"{req.algorithm} — Convergence",
            {"convergence_curve": result["convergence_curve"]},
        )

    return {
        "execution": result,
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


@app.get("/workflow/health")
async def health():
    return {"status": "ok", "version": "0.2.0", "steps": WORKFLOW_STEPS}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.api.workflow:app", host="0.0.0.0", port=8001, reload=True)
