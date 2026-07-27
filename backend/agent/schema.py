"""Algorithm metadata and parameter schemas for solver validation.

Standardises the metadata.json format for every algorithm in the knowledge
base and provides a validation engine to check if a problem description
matches an algorithm's capabilities before calling the solver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union


# ── Enums ────────────────────────────────────────────────────────────────────

class ProblemType(str, Enum):
    VRP = "VRP"
    CVRP = "CVRP"
    VRPTW = "VRPTW"
    TSP = "TSP"
    JSP = "JSP"                # Job Shop Scheduling
    FSP = "FSP"                # Flow Shop Scheduling
    OSP = "OSP"                # Open Shop Scheduling
    SCHEDULING = "Scheduling"  # generic / other scheduling
    KNAPSACK = "Knapsack"
    BIN_PACKING = "Bin Packing"
    CONTINUOUS = "Continuous Optimization"
    COMBINATORIAL = "Combinatorial"
    OTHER = "Other"


class ObjectiveType(str, Enum):
    SINGLE = "single"
    MULTI = "multi"


class VariableType(str, Enum):
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    PERMUTATION = "permutation"
    BINARY = "binary"
    MIXED = "mixed"


class ParamType(str, Enum):
    INT = "int"
    FLOAT = "float"
    STRING = "string"
    BOOL = "bool"


# ── Parameter definition ────────────────────────────────────────────────────

@dataclass
class ParameterDef:
    """Definition of a single solver parameter."""
    name: str
    type: ParamType
    default: Any
    description: str = ""
    required: bool = False
    min_val: Optional[Union[int, float]] = None
    max_val: Optional[Union[int, float]] = None
    choices: Optional[List[Any]] = None      # for enum-like params
    condition: Optional[str] = None          # human-readable when this param is needed

    def validate(self, value: Any) -> Optional[str]:
        """Return error message if value is invalid, None if ok."""
        if self.type == ParamType.INT:
            if not isinstance(value, int):
                return f"{self.name}: expected int, got {type(value).__name__}"
            if self.min_val is not None and value < self.min_val:
                return f"{self.name}: {value} < min {self.min_val}"
            if self.max_val is not None and value > self.max_val:
                return f"{self.name}: {value} > max {self.max_val}"
        elif self.type == ParamType.FLOAT:
            if not isinstance(value, (int, float)):
                return f"{self.name}: expected float, got {type(value).__name__}"
            if self.min_val is not None and value < self.min_val:
                return f"{self.name}: {value} < min {self.min_val}"
            if self.max_val is not None and value > self.max_val:
                return f"{self.name}: {value} > max {self.max_val}"
        elif self.type == ParamType.STRING and self.choices:
            if value not in self.choices:
                return f"{self.name}: {value} not in {self.choices}"
        return None

    def to_dict(self) -> dict:
        d = {
            "name": self.name, "type": self.type.value, "default": self.default,
            "description": self.description, "required": self.required,
        }
        if self.min_val is not None:
            d["min"] = self.min_val
        if self.max_val is not None:
            d["max"] = self.max_val
        if self.choices:
            d["choices"] = self.choices
        if self.condition:
            d["condition"] = self.condition
        return d

    @classmethod
    def from_dict(cls, data: dict, name: str = "") -> ParameterDef:
        """Parse from flat dict e.g. {"name": "pop", "type": "int", ...}."""
        return cls(
            name=data.get("name", name),
            type=ParamType(data["type"]),
            default=data.get("default"),
            description=data.get("description", ""),
            required=data.get("required", False),
            min_val=data.get("min"),
            max_val=data.get("max"),
            choices=data.get("choices"),
            condition=data.get("condition"),
        )

    @classmethod
    def from_nested(cls, name: str, data: dict) -> ParameterDef:
        """Parse from nested format.  Supports both styles:

        Simple:  {"pop": {"type":"integer", "default":50, "range":{"min":10,"max":500}}}
        JSON Schema: {"properties":{"pop":{"type":"integer","minimum":4,"default":100}},
                        "required":["pop"]}
        """
        # Normalise type: "integer" | "number" | "float" | "string" | "boolean"
        raw_type = str(data.get("type", "string")).lower()
        if raw_type in ("integer", "int"):
            ptype = ParamType.INT
        elif raw_type in ("number", "float", "double"):
            ptype = ParamType.FLOAT
        elif raw_type in ("boolean", "bool"):
            ptype = ParamType.BOOL
        else:
            ptype = ParamType.STRING

        # Normalise range
        rng = data.get("range", {})
        min_v = data.get("min") if "min" in data else (rng.get("min") if rng else None)
        max_v = data.get("max") if "max" in data else (rng.get("max") if rng else None)
        if min_v is None:
            min_v = data.get("minimum")
        if max_v is None:
            max_v = data.get("maximum")

        # Normalise description
        desc = data.get("description", "") or data.get("title", "")

        # Choices / enum
        choices = data.get("choices") or data.get("enum")

        return cls(
            name=name,
            type=ptype,
            default=data.get("default"),
            description=desc,
            required=data.get("required", False),
            min_val=min_v,
            max_val=max_v,
            choices=choices,
            condition=data.get("condition"),
        )


# ── Algorithm metadata ──────────────────────────────────────────────────────

@dataclass
class AlgorithmMetadata:
    """Structured metadata for one algorithm, parsed from metadata.json.

    Can be constructed from two files:
      metadata.json        — name, problemType, objectiveType, constraints, etc.
      parameter_schema.json — per-parameter type/default/range/description
    """
    name: str
    chinese_name: str = ""
    description: str = ""
    problem_types: List[ProblemType] = field(default_factory=list)
    objective_type: ObjectiveType = ObjectiveType.SINGLE
    variable_types: List[VariableType] = field(default_factory=list)
    algorithm_family: List[str] = field(default_factory=list)
    constraints: Dict[str, List[str]] = field(default_factory=dict)
    #    {"supported": ["time_windows", "capacity"],
    #     "required": ["capacity"],          // must be present in problem
    #     "optional": ["time_windows"]}      // can be present, not required
    parameters: List[ParameterDef] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    def supported_constraints(self) -> List[str]:
        return self.constraints.get("supported", [])

    def required_constraints(self) -> List[str]:
        return self.constraints.get("required", [])

    def optional_constraints(self) -> List[str]:
        return self.constraints.get("optional", [])

    def supports_problem_type(self, pt: ProblemType) -> bool:
        return pt in self.problem_types or ProblemType.OTHER in self.problem_types

    def supports_variable_type(self, vt: VariableType) -> bool:
        return vt in self.variable_types

    def get_param(self, name: str) -> Optional[ParameterDef]:
        for p in self.parameters:
            if p.name == name:
                return p
        return None

    @classmethod
    def from_dict(cls, data: dict, param_schema: Optional[dict] = None) -> AlgorithmMetadata:
        """Parse from metadata.json dict, optionally merging external parameter_schema.

        Supports three parameter formats:
          1. Flat list:      "parameters": [{"name":"pop","type":"int",...}, ...]
          2. Nested simple:  "parameters": {"pop": {"type":"integer",...}, ...}
          3. JSON Schema:    separate file with {"properties":{...}, "required":[...]}
        """
        def _parse_enums(values, enum_cls):
            if not values:
                return []
            return [enum_cls(v) for v in values if v in enum_cls._value2member_map_]

        params = []
        merged_schema = param_schema or data.get("parameters", {})

        if isinstance(merged_schema, dict):
            # Check for JSON Schema format (has "properties" key at top level)
            if "properties" in merged_schema:
                top_required = merged_schema.get("required", [])
                if isinstance(top_required, list):
                    top_required = set(top_required)
                else:
                    top_required = set()

                for pname, pdata in merged_schema["properties"].items():
                    pdata = dict(pdata)
                    pdata["required"] = pname in top_required
                    params.append(ParameterDef.from_nested(pname, pdata))
            else:
                # Nested simple format: {"pop": {"type":"integer",...}}
                for pname, pdata in merged_schema.items():
                    if isinstance(pdata, dict) and "type" in pdata:
                        params.append(ParameterDef.from_nested(pname, pdata))
        elif isinstance(merged_schema, list):
            params = [ParameterDef.from_dict(p) for p in merged_schema]

        return cls(
            name=data.get("name", ""),
            chinese_name=data.get("chineseName", "") or data.get("chinese_name", ""),
            description=data.get("description", ""),
            problem_types=_parse_enums(data.get("problemType", []), ProblemType),
            objective_type=ObjectiveType(data.get("objectiveType", "single")),
            variable_types=_parse_enums(data.get("variableType", []), VariableType),
            algorithm_family=data.get("algorithmFamily", []),
            constraints=data.get("constraints", {}),
            parameters=params,
            tags=data.get("tags", []),
        )

    @classmethod
    def from_directory(cls, algo_dir: str) -> AlgorithmMetadata:
        """Load from metadata.json + parameter_schema.json in a directory.

        Args:
            algo_dir: Path to an algorithm directory containing both JSON files.
        """
        import json as _json
        from pathlib import Path

        base = Path(algo_dir)
        meta_path = base / "metadata.json"
        param_path = base / "parameter_schema.json"

        meta = _json.loads(meta_path.read_text(encoding="utf-8"))
        param = None
        if param_path.exists():
            param = _json.loads(param_path.read_text(encoding="utf-8"))

        return cls.from_dict(meta, param_schema=param)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "chineseName": self.chinese_name,
            "description": self.description,
            "problemType": [pt.value for pt in self.problem_types],
            "objectiveType": self.objective_type.value,
            "variableType": [vt.value for vt in self.variable_types],
            "algorithmFamily": self.algorithm_family,
            "constraints": self.constraints,
            "parameters": [p.to_dict() for p in self.parameters],
            "tags": self.tags,
        }


# ── Validation result ───────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    passed: bool
    errors: List[str] = field(default_factory=list)     # blocking issues
    warnings: List[str] = field(default_factory=list)   # non-blocking concerns
    missing_params: List[str] = field(default_factory=list)
    invalid_params: List[str] = field(default_factory=list)

    def merge(self, other: ValidationResult) -> ValidationResult:
        return ValidationResult(
            passed=self.passed and other.passed,
            errors=self.errors + other.errors,
            warnings=self.warnings + other.warnings,
            missing_params=self.missing_params + other.missing_params,
            invalid_params=self.invalid_params + other.invalid_params,
        )


# ── Validation engine ────────────────────────────────────────────────────────

def validate_algorithm_for_problem(
    algo: AlgorithmMetadata,
    problem_type: Optional[ProblemType],
    variable_type: Optional[VariableType],
    objective_count: int,
    user_constraints: List[str],
    user_params: Optional[Dict[str, Any]] = None,
) -> ValidationResult:
    """Check whether an algorithm can handle the given problem specification.

    Returns a ValidationResult with pass/fail and specific issues.
    """
    result = ValidationResult(passed=True)

    # 1. Problem type match
    if problem_type and not algo.supports_problem_type(problem_type):
        result.errors.append(
            f"Algorithm '{algo.name}' does not support problem type '{problem_type.value}'. "
            f"Supported: {[p.value for p in algo.problem_types]}"
        )
        result.passed = False

    # 2. Variable type match
    if variable_type and not algo.supports_variable_type(variable_type):
        result.errors.append(
            f"Algorithm '{algo.name}' requires variable type in "
            f"{[v.value for v in algo.variable_types]}, but problem has '{variable_type.value}'"
        )
        result.passed = False

    # 3. Objective count match
    if algo.objective_type == ObjectiveType.SINGLE and objective_count > 1:
        result.errors.append(
            f"Algorithm '{algo.name}' is single-objective but problem has {objective_count} objectives. "
            f"Use a multi-objective algorithm (NSGA-II, MOEA/D, etc.) instead."
        )
        result.passed = False

    # 4. Required constraints present
    required = algo.required_constraints()
    for req in required:
        if not any(req.lower() in c.lower() for c in user_constraints):
            result.errors.append(
                f"Algorithm '{algo.name}' requires constraint '{req}' but it is not present "
                f"in the problem description. Available constraints: {user_constraints}"
            )
            result.passed = False

    # 5. Warn about unsupported constraints
    supported = algo.supported_constraints()
    for c in user_constraints:
        if not any(c.lower() in s.lower() for s in supported):
            result.warnings.append(
                f"Constraint '{c}' is not in algorithm '{algo.name}' supported list: {supported}. "
                f"The solver may ignore or mishandle it."
            )

    # 6. Parameter validation
    if user_params:
        for name, value in user_params.items():
            param_def = algo.get_param(name)
            if param_def is None:
                continue  # unknown params are silently ignored
            err = param_def.validate(value)
            if err:
                result.invalid_params.append(err)

    # Check required params
    for p in algo.parameters:
        if p.required and (user_params is None or p.name not in user_params):
            result.missing_params.append(
                f"'{algo.name}' requires parameter '{p.name}' ({p.type.value}, "
                f"default={p.default}): {p.description}"
            )

    return result


def rank_algorithms(
    algorithms: List[AlgorithmMetadata],
    problem_type: Optional[ProblemType],
    variable_type: Optional[VariableType],
    objective_count: int,
    user_constraints: List[str],
) -> List[tuple]:
    """Sort algorithms by fitness for a given problem.  Returns (algo, score, result).

    Score: starts at 100, loses points for each mismatch.  Higher = better fit.
    """
    scored = []
    for algo in algorithms:
        result = validate_algorithm_for_problem(
            algo, problem_type, variable_type, objective_count, user_constraints,
        )
        score = 100
        score -= 20 * len(result.errors)
        score -= 5 * len(result.warnings)
        score -= 3 * len(result.missing_params)
        scored.append((algo, max(0, score), result))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored
