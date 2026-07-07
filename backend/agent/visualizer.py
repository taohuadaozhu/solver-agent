"""Chart generator — creates plotly-compatible JSON for the frontend."""

import logging
import random
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def generate_chart(chart_type: str, title: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a Plotly-compatible chart spec from execution results."""
    if chart_type == "convergence":
        return _convergence_chart(title, data)
    elif chart_type == "gantt":
        return _gantt_chart(title, data)
    elif chart_type == "route_map":
        return _route_map(title, data)
    elif chart_type == "bar_chart":
        return _bar_chart(title, data)
    elif chart_type == "pareto_front":
        return _pareto_front(title, data)
    else:
        return {"error": f"Unknown chart type: {chart_type}"}


def _convergence_chart(title: str, data: Dict[str, Any]) -> Dict[str, Any]:
    curve = data.get("convergence_curve", [1000, 500, 300, 200, 150, 120, 100])
    iterations = list(range(1, len(curve) + 1))
    return {
        "type": "convergence",
        "data": [
            {
                "x": iterations,
                "y": curve,
                "type": "scatter",
                "mode": "lines+markers",
                "name": "Best Objective",
                "line": {"color": "#2563eb", "width": 2},
            }
        ],
        "layout": {
            "title": title,
            "xaxis": {"title": "Iteration"},
            "yaxis": {"title": "Objective Value"},
        },
    }


def _gantt_chart(title: str, data: Dict[str, Any]) -> Dict[str, Any]:
    tasks = data.get("tasks", [])
    if not tasks:
        # Generate demo gantt data
        tasks = []
        for i in range(6):
            start = sum(random.randint(3, 15) for _ in range(i))
            duration = random.randint(5, 20)
            tasks.append({
                "task": f"Job {i+1}",
                "machine": f"M{random.randint(1, 4)}",
                "start": start,
                "duration": duration,
            })

    machines = sorted(set(t["machine"] for t in tasks))
    colors = ["#2563eb", "#dc2626", "#16a34a", "#ca8a04", "#7c3aed", "#0891b2"]

    traces = []
    for i, m in enumerate(machines):
        m_tasks = [t for t in tasks if t["machine"] == m]
        traces.append({
            "x": [t["duration"] for t in m_tasks],
            "y": [t["task"] for t in m_tasks],
            "type": "bar",
            "orientation": "h",
            "name": m,
            "marker": {"color": colors[i % len(colors)]},
        })

    return {
        "type": "gantt",
        "data": traces,
        "layout": {
            "title": title,
            "barmode": "stack",
            "xaxis": {"title": "Time"},
            "yaxis": {"title": "Jobs"},
        },
    }


def _route_map(title: str, data: Dict[str, Any]) -> Dict[str, Any]:
    points = data.get("points", [])
    if not points:
        rng = random.Random(42)
        points = [
            {"name": f"City {i+1}", "x": rng.uniform(0, 100), "y": rng.uniform(0, 100)}
            for i in range(10)
        ]

    xs = [p["x"] for p in points]
    ys = [p["y"] for p in points]
    labels = [p["name"] for p in points]

    return {
        "type": "route_map",
        "data": [
            {
                "x": xs,
                "y": ys,
                "type": "scatter",
                "mode": "lines+markers+text",
                "name": "Route",
                "text": labels,
                "textposition": "top center",
                "line": {"color": "#2563eb", "width": 2},
                "marker": {"size": 10},
            }
        ],
        "layout": {
            "title": title,
            "xaxis": {"title": "X"},
            "yaxis": {"title": "Y"},
        },
    }


def _bar_chart(title: str, data: Dict[str, Any]) -> Dict[str, Any]:
    labels = data.get("labels", ["Objective 1", "Objective 2", "Objective 3", "Objective 4"])
    values = data.get("values", [random.uniform(50, 100) for _ in labels])

    return {
        "type": "bar_chart",
        "data": [
            {
                "x": labels,
                "y": values,
                "type": "bar",
                "marker": {"color": "#2563eb"},
                "name": title,
            }
        ],
        "layout": {
            "title": title,
            "xaxis": {"title": ""},
            "yaxis": {"title": "Value"},
        },
    }


def _pareto_front(title: str, data: Dict[str, Any]) -> Dict[str, Any]:
    points = data.get("points", [])
    if not points:
        rng = random.Random(123)
        points = [
            {"f1": rng.uniform(0, 1), "f2": rng.uniform(0, 1)}
            for _ in range(30)
        ]

    return {
        "type": "pareto_front",
        "data": [
            {
                "x": [p["f1"] for p in points],
                "y": [p["f2"] for p in points],
                "type": "scatter",
                "mode": "markers",
                "name": "Solutions",
                "marker": {"size": 8, "color": "#2563eb", "opacity": 0.7},
            }
        ],
        "layout": {
            "title": title,
            "xaxis": {"title": "Objective 1 (minimize)"},
            "yaxis": {"title": "Objective 2 (minimize)"},
        },
    }
