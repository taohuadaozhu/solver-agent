"""Tool definitions for OpenAI function calling — solver execution and visualization."""

import json
import logging
from typing import Any, Dict, List

from backend.agent.executor import execute_solver, load_dataset

logger = logging.getLogger(__name__)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "load_dataset",
            "description": "Load an optimization dataset by name from the knowledge base.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_name": {
                        "type": "string",
                        "description": "Name of the dataset, e.g. 'Berlin52 TSP Dataset' or 'FT10 Job Shop Scheduling Dataset'.",
                    },
                },
                "required": ["dataset_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_solver",
            "description": "Execute an optimization algorithm on a given dataset and return results including objective values, runtime, and solution quality.",
            "parameters": {
                "type": "object",
                "properties": {
                    "algorithm": {
                        "type": "string",
                        "description": "Algorithm name, e.g. 'GA-Permutation', 'NSGA-II', 'ACO'.",
                    },
                    "dataset_name": {
                        "type": "string",
                        "description": "Dataset to run the algorithm on.",
                    },
                    "parameters": {
                        "type": "object",
                        "description": "Algorithm hyperparameters as key-value pairs.",
                        "additionalProperties": True,
                    },
                },
                "required": ["algorithm", "dataset_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_chart",
            "description": "Generate a visualization chart from optimization results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "chart_type": {
                        "type": "string",
                        "enum": ["convergence", "gantt", "route_map", "pareto_front", "bar_chart"],
                        "description": "Type of chart to generate.",
                    },
                    "title": {
                        "type": "string",
                        "description": "Chart title.",
                    },
                    "data": {
                        "type": "object",
                        "description": "Chart data as a JSON-serializable object.",
                    },
                },
                "required": ["chart_type", "title", "data"],
            },
        },
    },
]

from backend.agent.visualizer import generate_chart

TOOL_HANDLERS = {
    "load_dataset": load_dataset,
    "execute_solver": execute_solver,
    "generate_chart": generate_chart,
}


def call_tool(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatch a tool call to the appropriate handler."""
    handler = TOOL_HANDLERS.get(tool_name)
    if handler is None:
        return {"error": f"Unknown tool: {tool_name}"}

    try:
        result = handler(**arguments)
        return {"success": True, "result": result}
    except Exception as e:
        logger.exception("Tool %s failed", tool_name)
        return {"success": False, "error": str(e)}
