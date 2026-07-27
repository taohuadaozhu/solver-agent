"""Batch upgrade all algorithm metadata.json to the new schema format.

Converts:
  - problemType from ["Continuous Optimization"] → ["VRP","Scheduling",...]
  - objectiveType from ["Single Objective"] array → "single" string
  - variableType from ["Continuous"] → ["continuous","discrete"] etc.
  - Adds constraints stub (supported/required/optional)
  - Removes businessScenario
  - Creates parameter_schema.json with defaults (if missing)

Run:  python -m backend.rag.migrate_metadata
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

KB_ROOT = ROOT_DIR / "knowledge-base"

# ── Mapping tables ──────────────────────────────────────────────────────────

# Heuristic: algorithm family → likely problem types
FAMILY_TO_PROBLEM: Dict[str, List[str]] = {
    "Evolutionary": ["VRP", "TSP", "JSP", "FSP", "Scheduling",
                     "Knapsack", "Continuous Optimization"],
    "Swarm":       ["VRP", "TSP", "Scheduling", "Continuous Optimization"],
    "Bayesian":    ["Continuous Optimization"],
    "Mathematical": ["Continuous Optimization", "Combinatorial"],
    "Surrogate":   ["Continuous Optimization"],
    "Deep Learning": ["Continuous Optimization", "Combinatorial"],
    "Local Search": ["VRP", "TSP", "JSP", "FSP", "Scheduling"],
}

# Heuristic: name keywords → variable types
NAME_TO_VARIABLE: Dict[str, List[str]] = {
    "permutation": ["permutation", "discrete"],
    "tsp":         ["permutation", "discrete"],
    "vrp":         ["permutation", "discrete"],
    "scheduling":  ["discrete", "permutation"],
    "binary":      ["binary"],
    "discrete":    ["discrete"],
    "continuous":  ["continuous"],
    "combinatorial": ["discrete", "permutation"],
}

# Heuristic: name keywords → constraint support
NAME_TO_CONSTRAINTS: Dict[str, dict] = {
    "vrp":     {"supported": ["capacity", "time_windows"],
                "required": [], "optional": ["capacity", "time_windows"]},
    "cvrp":    {"supported": ["capacity"],
                "required": ["capacity"], "optional": []},
    "time window": {"supported": ["time_windows"],
                    "required": [], "optional": ["time_windows"]},
    "jsp":     {"supported": ["precedence", "setup_time", "machine_eligibility"],
                "required": [], "optional": ["precedence", "setup_time"]},
    "scheduling": {"supported": ["precedence", "setup_time"],
                   "required": [], "optional": ["precedence", "setup_time"]},
}

COMMON_PARAMS = {
    "population_size": {
        "type": "integer", "required": False, "default": 50,
        "range": {"min": 10, "max": 500},
        "description": "Population / swarm size"
    },
    "num_generations": {
        "type": "integer", "required": False, "default": 200,
        "range": {"min": 10, "max": 2000},
        "description": "Maximum number of generations / iterations"
    },
}


def _normalize_objective_type(raw) -> str:
    """Convert ["Single Objective"] → "single", ["Multi Objective"] → "multi"."""
    if isinstance(raw, str):
        raw = [raw]
    if not raw:
        return "single"
    first = str(raw[0]).lower()
    if "multi" in first:
        return "multi"
    return "single"


def _infer_problem_types(name: str, family: List[str], old_problem: List[str]) -> List[str]:
    """Infer business problem types from algorithm family + name + old data."""
    # If old data already has business types (not "Continuous Optimization"), keep them
    non_continuous = [p for p in old_problem
                      if p.lower() not in ("continuous optimization",
                                            "permutation optimization",
                                            "combinatorial optimization")]
    if non_continuous:
        return non_continuous

    # Infer from family
    for fam in family:
        if fam in FAMILY_TO_PROBLEM:
            return FAMILY_TO_PROBLEM[fam]

    # Infer from name keywords
    name_lower = name.lower()
    for keyword, types in NAME_TO_PROBLEM_FALLBACK.items():
        if keyword in name_lower:
            return types

    return ["Continuous Optimization"]


def _infer_variable_types(name: str, old_vars: List[str]) -> List[str]:
    """Infer variable types from name keywords."""
    name_lower = name.lower()
    for keyword, vtypes in NAME_TO_VARIABLE.items():
        if keyword in name_lower:
            return vtypes
    # Fallback from old data
    return old_vars if old_vars else ["continuous"]


def _infer_constraints(name: str) -> dict:
    """Infer constraint support from name keywords."""
    name_lower = name.lower()
    for keyword, cons in NAME_TO_CONSTRAINTS.items():
        if keyword in name_lower:
            return cons
    return {"supported": [], "required": [], "optional": []}


def _infer_params_from_name(name: str, family: List[str]) -> dict:
    """Generate default parameter_schema based on algorithm family."""
    params = {}
    has_population = any(f.lower() in ("evolutionary", "swarm") for f in family)
    if has_population:
        params["population_size"] = {
            "type": "integer", "required": False, "default": 50,
            "range": {"min": 10, "max": 500},
            "description": "Population or swarm size"
        }
        params["num_generations"] = {
            "type": "integer", "required": False, "default": 200,
            "range": {"min": 10, "max": 2000},
            "description": "Maximum number of generations"
        }
    return params


# Additional fallback for problem type inference
NAME_TO_PROBLEM_FALLBACK = {
    "vrp":     ["VRP", "CVRP", "VRPTW"],
    "tsp":     ["TSP"],
    "job shop": ["JSP"],
    "jsp":     ["JSP"],
    "flow shop": ["FSP"],
    "fsp":     ["FSP"],
    "open shop": ["OSP"],
    "knapsack": ["Knapsack"],
    "scheduling": ["Scheduling"],
    "routing": ["VRP", "TSP"],
    "vehicle": ["VRP", "CVRP"],
    "logistics": ["VRP"],
}


# ── Main migration ──────────────────────────────────────────────────────────

def migrate_one(meta_path: Path, dry_run: bool = False) -> int:
    """Migrate a single metadata.json.  Returns 1 if changed, 0 if skipped."""
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    name = data.get("name", meta_path.parent.name)
    family = data.get("algorithmFamily", [])

    # Check if already migrated (has constraints with dict structure)
    existing_cons = data.get("constraints", {})
    if isinstance(existing_cons, dict) and "supported" in existing_cons:
        return 0  # already in new format

    algo_dir = meta_path.parent

    # Build new metadata
    new_meta = {
        "name": name,
        "chineseName": data.get("chineseName", ""),
        "description": data.get("description", ""),
        "problemType": _infer_problem_types(
            name, family, data.get("problemType", [])
        ),
        "objectiveType": _normalize_objective_type(data.get("objectiveType", [])),
        "variableType": _infer_variable_types(
            name, data.get("variableType", [])
        ),
        "algorithmFamily": family,
        "constraints": _infer_constraints(name),
        "tags": [],
    }

    # Build parameter_schema if missing
    param_path = algo_dir / "parameter_schema.json"
    if not param_path.exists():
        params = _infer_params_from_name(name, family)
        if params and not dry_run:
            param_path.write_text(
                json.dumps(params, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

    if not dry_run:
        meta_path.write_text(
            json.dumps(new_meta, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    return 1


def run(dry_run: bool = False):
    """Migrate all algorithm README directories under knowledge-base."""
    algo_base = KB_ROOT / "algorithms"
    total = 0
    changed = 0
    skipped = 0

    for readme_path in sorted(algo_base.rglob("README.md")):
        meta_path = readme_path.parent / "metadata.json"
        if not meta_path.exists():
            continue
        total += 1
        try:
            result = migrate_one(meta_path, dry_run=dry_run)
            if result > 0:
                changed += 1
                if dry_run:
                    print(f"  [DRY-RUN] Would migrate: {meta_path.relative_to(KB_ROOT)}")
            else:
                skipped += 1
        except Exception as exc:
            print(f"  ERROR {meta_path.relative_to(KB_ROOT)}: {exc}")

    print(f"\nTotal: {total} | Migrated: {changed} | Skipped (already new): {skipped}")
    if dry_run:
        print("DRY-RUN mode — no files were modified. Remove --dry-run to apply.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Batch upgrade algorithm metadata.json to new schema format."
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview changes without writing files.")
    args = parser.parse_args()
    run(dry_run=args.dry_run)
