from typing import Dict, Optional

STEP_PROMPTS = {
    "STEP_1_PROBLEM": (
        "STEP 1 — PROBLEM UNDERSTANDING:\n"
        "Please read the user's problem description and extract the core optimization task.\n"
        "Ask for the domain, key actors, resources, and the main performance challenge.\n"
        "If the description is unclear, request a concise problem statement with concrete constraints.\n"
    ),
    "STEP_2_DATASET": (
        "STEP 2 — DATASET SELECTION:\n"
        "Based on the described problem, identify the most appropriate dataset type or benchmark instance.\n"
        "Specify whether the user needs routing, scheduling, production, or planning data.\n"
        "If possible, mention the concrete dataset name or file format needed.\n"
    ),
    "STEP_3_VARIABLE": (
        "STEP 3 — DECISION VARIABLES:\n"
        "List the decision variables that the optimization should control.\n"
        "For example: assignments, sequences, routes, start times, resource allocations, or machine selections.\n"
        "Explain why each variable is important to the user problem.\n"
    ),
    "STEP_4_OBJECTIVE": (
        "STEP 4 — OBJECTIVE DEFINITION:\n"
        "Identify the objective(s) that the optimization should pursue.\n"
        "Common objectives include minimizing cost, time, distance, energy, or maximizing throughput and utilization.\n"
        "If the problem is multi-objective, state the main trade-offs clearly.\n"
    ),
    "STEP_5_CONSTRAINT": (
        "STEP 5 — CONSTRAINT IDENTIFICATION:\n"
        "List the constraints and business rules that the solution must satisfy.\n"
        "Examples: resource capacities, time windows, sequence restrictions, machine availability, safety limits.\n"
        "Be precise about hard constraints versus soft preferences.\n"
    ),
    "STEP_6_CLASSIFY": (
        "STEP 6 — PROBLEM CLASSIFICATION:\n"
        "Classify the problem into one or more known optimization categories.\n"
        "Possible categories include: vehicle routing, job shop scheduling, open shop scheduling, flow shop scheduling, production planning, or general combinatorial optimization.\n"
        "Use the problem description, variables, objectives, and constraints to choose the best category.\n"
    ),
    "STEP_7_ALGO": (
        "STEP 7 — ALGORITHM RECOMMENDATION:\n"
        "Recommend the most suitable optimization algorithms for the classified problem type.\n"
        "Consider dataset scale, objective type, constraint complexity, and required solution quality.\n"
        "Mention both primary algorithms and useful hybrid or local search techniques.\n"
    ),
}


def build_prompt(query: str, context_chunks: list):
    context_text = '\n\n'.join([chunk.get('content', '') for chunk in context_chunks])
    return f"""Given the following context from the knowledge base, answer the user query.\n\nContext:\n{context_text}\n\nQuestion:\n{query}\n"""


def get_step_prompt(step: str, context: Optional[Dict] = None) -> str:
    template = STEP_PROMPTS.get(step)
    if not template:
        raise ValueError(f'Unknown step: {step}')

    if not context:
        return template

    details = []
    if step == "STEP_1_PROBLEM":
        details.append(f"User problem description: {context.get('problem', '')}")
    if step == "STEP_2_DATASET":
        details.append(f"Problem summary: {context.get('problem_summary', '')}")
    if step == "STEP_3_VARIABLE":
        details.append(f"Dataset or problem type: {context.get('dataset_type', '')}")
    if step == "STEP_4_OBJECTIVE":
        details.append(f"Variables: {context.get('variables', '')}")
    if step == "STEP_5_CONSTRAINT":
        details.append(f"Objectives: {context.get('objectives', '')}")
    if step == "STEP_6_CLASSIFY":
        details.append(f"Constraints: {context.get('constraints', '')}")
    if step == "STEP_7_ALGO":
        details.append(
            f"Classification: {context.get('classification', '')}, "
            f"dataset: {context.get('dataset', '')}, "
            f"constraints: {context.get('constraints', '')}"
        )

    if details:
        return template + "\n\nContext details:\n" + "\n".join(details)
    return template


def step1_handler(context: Optional[Dict] = None) -> str:
    return get_step_prompt("STEP_1_PROBLEM", context)


def step2_handler(context: Optional[Dict] = None) -> str:
    return get_step_prompt("STEP_2_DATASET", context)


def step3_handler(context: Optional[Dict] = None) -> str:
    return get_step_prompt("STEP_3_VARIABLE", context)


def step4_handler(context: Optional[Dict] = None) -> str:
    return get_step_prompt("STEP_4_OBJECTIVE", context)


def step5_handler(context: Optional[Dict] = None) -> str:
    return get_step_prompt("STEP_5_CONSTRAINT", context)


def step6_handler(context: Optional[Dict] = None) -> str:
    return get_step_prompt("STEP_6_CLASSIFY", context)


def step7_handler(context: Optional[Dict] = None) -> str:
    return get_step_prompt("STEP_7_ALGO", context)


def get_step_handler(step: str):
    if step == "STEP_1_PROBLEM":
        return step1_handler

    if step == "STEP_2_DATASET":
        return step2_handler

    if step == "STEP_3_VARIABLE":
        return step3_handler

    if step == "STEP_4_OBJECTIVE":
        return step4_handler

    if step == "STEP_5_CONSTRAINT":
        return step5_handler

    if step == "STEP_6_CLASSIFY":
        return step6_handler

    if step == "STEP_7_ALGO":
        return step7_handler

    raise ValueError(f'Unknown step: {step}')
