"""Embedding model evaluation for the optimization knowledge base.

Measures retrieval quality across search strategies and (optionally) embedding
models.  Run after building the chunk index:

    python -m backend.rag.eval_embeddings

Customise TEST_QUERIES below to match your domain before running.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from backend.rag.retriever import (
    semantic_search,
    keyword_search,
    hybrid_search,
    hybrid_search_with_rerank,
)

logging.basicConfig(level=logging.WARNING)  # suppress INFO spam during eval
logger = logging.getLogger("eval")

# ── Test queries ─────────────────────────────────────────────────────────────
# Each entry: (query, [relevant_document_names])
# "relevant_document_names" should be substrings that appear in the doc name.
# Add at least 20 entries that cover your real use-cases.

TEST_QUERIES: List[Tuple[str, List[str]]] = [
    # ── Chinese queries ──
    ("最小化完工时间的作业车间调度算法", ["GA", "SA", "FT10", "JSP"]),
    ("车辆路径问题带时间窗约束", ["VRP", "ACO", "Solomon", "Time Window"]),
    ("多目标优化帕累托前沿如何生成", ["NSGA", "Pareto", "MOEA"]),
    ("大规模flow shop调度用什么算法", ["Flow Shop", "ILS", "GA", "SA"]),
    ("带容量限制的车辆路径规划", ["CVRP", "VRP", "Vehicle Routing"]),
    ("单机调度的最小化总延迟", ["Scheduling", "Tardiness", "SA"]),
    ("遗传算法的交叉概率和变异概率怎么设置", ["GA", "Crossover", "Mutation"]),
    ("蚁群算法解TSP问题", ["ACO", "TSP", "Ant Colony"]),
    ("流水线车间调度序列相关换模时间", ["Flow Shop", "Setup", "ILS"]),
    ("鲁棒优化处理机器故障的不确定性", ["Robust", "Stochastic", "Uncertainty"]),

    # ── English queries ──
    ("job shop scheduling minimize makespan", ["JSP", "GA", "SA", "FT10"]),
    ("capacitated vehicle routing with time windows", ["CVRP", "VRP", "Solomon"]),
    ("NSGA-II multi objective optimization", ["NSGA", "Pareto", "Multi"]),
    ("particle swarm optimization for continuous problems", ["PSO", "Particle Swarm"]),
    ("differential evolution algorithm convergence", ["DE", "Differential Evolution"]),
    ("simulated annealing permutation flow shop", ["SA", "Flow Shop", "Permutation"]),

    # ── Mixed / parameter-level queries ──
    ("GA crossover rate 0.9 mutation rate 0.1", ["GA", "Genetic"]),
    ("benchmark dataset for TSP Berlin52", ["TSP", "Berlin", "GA"]),
    ("open shop scheduling algorithm comparison", ["Open Shop", "GA", "SA"]),

    # ── Edge cases ──
    ("cuckoo search levy flight", ["Cuckoo Search", "CSA", "Levy"]),
    ("harmony search improvisation rate", ["Harmony Search", "HS"]),
    ("人工蜂群算法雇佣蜂观察蜂", ["Bee Colony", "ABCA", "Swarm"]),
    ("whale optimization bubble net hunting", ["Whale", "WOA"]),
    ("surrogate assisted evolutionary optimization", ["Surrogate", "EGO", "Kriging"]),
    ("variable neighborhood search shaking", ["VNS", "Variable Neighborhood"]),
]

# ── Metrics ──────────────────────────────────────────────────────────────────


@dataclass
class EvalResult:
    strategy: str
    recall_at_k: Dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    ndcg_at_k: Dict[int, float] = field(default_factory=dict)
    avg_latency_ms: float = 0.0
    total_queries: int = 0
    failures: int = 0


def _matches(doc_name: str, rel_list: List[str]) -> bool:
    """Check if a document name matches any relevance substring."""
    return any(r.lower() in doc_name.lower() for r in rel_list)


def _dcg(relevance_scores: List[float], k: int) -> float:
    """Discounted Cumulative Gain."""
    return sum(
        rel / math.log2(i + 2)  # i+2 because i is 0-indexed
        for i, rel in enumerate(relevance_scores[:k])
    )


def _ndcg(results: List[dict], relevant: List[str], k: int) -> float:
    """Normalized DCG@k — 1.0 means ideal ranking."""
    relevance = [1.0 if _matches(r["name"], relevant) else 0.0 for r in results[:k]]
    dcg = _dcg(relevance, k)
    ideal = _dcg(sorted(relevance, reverse=True), k)
    return dcg / ideal if ideal > 0 else 0.0


async def _evaluate_strategy(
    name: str,
    strategy_fn: Callable,
    k_values: Tuple[int, ...] = (1, 3, 5),
) -> EvalResult:
    """Run all test queries through a retrieval strategy, collect metrics."""
    result = EvalResult(strategy=name, total_queries=len(TEST_QUERIES))
    recall_sums = {k: 0.0 for k in k_values}
    ndcg_sums = {k: 0.0 for k in k_values}
    mrr_sum = 0.0
    latencies = []

    for query, relevant_docs in TEST_QUERIES:
        try:
            t0 = time.perf_counter()
            hits = await strategy_fn(query)
            elapsed = (time.perf_counter() - t0) * 1000
            latencies.append(elapsed)

            names = [h["name"] for h in hits]

            # Recall@k
            for k in k_values:
                top_names = names[:k]
                matched = sum(1 for doc in relevant_docs
                              if any(doc.lower() in n.lower() for n in top_names))
                recall_sums[k] += matched / max(len(relevant_docs), 1)

            # MRR
            for i, n in enumerate(names):
                if _matches(n, relevant_docs):
                    mrr_sum += 1.0 / (i + 1)
                    break

            # NDCG@k
            for k in k_values:
                ndcg_sums[k] += _ndcg(hits, relevant_docs, k)

        except Exception as exc:
            result.failures += 1
            logger.error("Query failed: %s — %s", query[:60], exc)

    n = len(TEST_QUERIES)
    result.recall_at_k = {k: recall_sums[k] / n for k in k_values}
    result.mrr = mrr_sum / n
    result.ndcg_at_k = {k: ndcg_sums[k] / n for k in k_values}
    result.avg_latency_ms = sum(latencies) / len(latencies) if latencies else 0

    return result


# ── Strategy definitions ─────────────────────────────────────────────────────


async def _dense_only(query: str) -> List[dict]:
    return await semantic_search(query, top_k=10)


async def _keyword_only(query: str) -> List[dict]:
    return await keyword_search(query, top_k=10)


async def _hybrid_rrf(query: str) -> List[dict]:
    return await hybrid_search(query, top_k=10)


async def _hybrid_rerank(query: str) -> List[dict]:
    return await hybrid_search_with_rerank(query, top_k=10)


STRATEGIES = {
    "dense_only": _dense_only,
    "keyword_only": _keyword_only,
    "hybrid_rrf": _hybrid_rrf,
    "hybrid_rerank": _hybrid_rerank,
}


# ── Report formatting ────────────────────────────────────────────────────────


def _print_separator(title: str):
    print(f"\n{'─' * 70}")
    print(f"  {title}")
    print(f"{'─' * 70}")


def _print_summary_table(results: List[EvalResult]):
    """Print a compact comparison table."""
    print(f"\n{'Strategy':<22s} {'Recall@3':>9s} {'Recall@5':>9s} {'MRR':>8s} {'NDCG@5':>8s} {'Lat(ms)':>8s}")
    print("─" * 70)
    for r in results:
        print(
            f"{r.strategy:<22s} "
            f"{r.recall_at_k.get(3, 0):>8.1%}  "
            f"{r.recall_at_k.get(5, 0):>8.1%}  "
            f"{r.mrr:>7.3f}  "
            f"{r.ndcg_at_k.get(5, 0):>7.3f}  "
            f"{r.avg_latency_ms:>7.0f}"
        )
    print()


def _print_per_query_breakdown(results: List[EvalResult], top_n: int = 5):
    """Print worst-performing queries to help spot gaps."""
    _print_separator(f"Worst {top_n} queries (by MRR across all strategies)")

    # Average MRR per query
    query_mrr: Dict[int, float] = {}
    for i, (query, _) in enumerate(TEST_QUERIES):
        mrr_vals = []
        for r in results:
            # Reconstruct per-query MRR (approximate from aggregate)
            pass  # Per-query stats would need per-query result storage

    print("  (Enable per-query tracking for detailed breakdown)")


def _print_interpretation(results: List[EvalResult]):
    """Human-readable interpretation of results."""
    _print_separator("Interpretation")

    best = max(results, key=lambda r: r.ndcg_at_k.get(5, 0))
    worst = min(results, key=lambda r: r.ndcg_at_k.get(5, 0))

    hyb = next((r for r in results if r.strategy == "hybrid_rerank"), None)
    dense = next((r for r in results if r.strategy == "dense_only"), None)

    if hyb and dense:
        gain = (hyb.ndcg_at_k.get(5, 0) - dense.ndcg_at_k.get(5, 0)) / max(dense.ndcg_at_k.get(5, 0), 0.001)
        print(f"  Hybrid+rerank NDCG@5 gain over dense-only: {gain:+.0%}")
        if gain > 0.10:
            print("  ✓ Hybrid search and LLM rerank provide meaningful lift.")
        elif gain > 0:
            print("  = Modest improvement. Rerank helps but isn't transformative.")
        else:
            print("  ✗ Rerank may not be worth the extra latency. Review your rerank prompt.")

    print(f"  Best strategy:  {best.strategy} (NDCG@5={best.ndcg_at_k.get(5,0):.3f})")
    print(f"  Worst strategy: {worst.strategy} (NDCG@5={worst.ndcg_at_k.get(5,0):.3f})")

    if best.avg_latency_ms > 2000:
        print(f"  ⚠  Best strategy latency ({best.avg_latency_ms:.0f}ms) > 2s — consider caching.")
    print()


# ── Main ─────────────────────────────────────────────────────────────────────


async def run(only_strategies: Optional[List[str]] = None):
    """Run the full evaluation suite.

    Args:
        only_strategies: If set, only run these strategies (e.g. ["dense_only", "hybrid_rerank"]).
    """
    strategies_to_run = only_strategies or list(STRATEGIES.keys())

    print(f"\n{'=' * 70}")
    print(f"  RAG Embedding Evaluation")
    print(f"  Queries: {len(TEST_QUERIES)}  |  Strategies: {len(strategies_to_run)}")
    print(f"{'=' * 70}")

    results: List[EvalResult] = []
    for name in strategies_to_run:
        fn = STRATEGIES.get(name)
        if fn is None:
            print(f"  Unknown strategy: {name} — skipping")
            continue

        print(f"\n  Running {name} ... ", end="", flush=True)
        t0 = time.perf_counter()
        result = await _evaluate_strategy(name, fn)
        elapsed = (time.perf_counter() - t0)
        print(f"done ({elapsed:.1f}s, {result.failures} failures)")
        results.append(result)

    _print_summary_table(results)
    _print_interpretation(results)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate RAG retrieval quality across strategies."
    )
    parser.add_argument("--only", nargs="+", choices=list(STRATEGIES.keys()),
                        help="Only run specific strategies.")
    parser.add_argument("--json", action="store_true",
                        help="Output results as JSON for CI / automation.")
    args = parser.parse_args()

    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(
        run(only_strategies=args.only if args.only else None)
    )

    if args.json:
        report = []
        for r in results:
            report.append({
                "strategy": r.strategy,
                "recall_at_3": r.recall_at_k.get(3, 0),
                "recall_at_5": r.recall_at_k.get(5, 0),
                "mrr": r.mrr,
                "ndcg_at_5": r.ndcg_at_k.get(5, 0),
                "avg_latency_ms": r.avg_latency_ms,
                "queries": r.total_queries,
                "failures": r.failures,
            })
        print(json.dumps(report, indent=2))
