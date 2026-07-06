from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import networkx as nx

from MPC4plus.experiments.evaluation.metrics import (
    Phase4RunMetrics,
    compute_phase4_run_metrics,
)


@dataclass
class EndToEndRunMetrics:
    graph_id: str
    selector_name: str
    num_nodes: int
    num_edges: int
    num_phase4_steps: int
    num_trace_samples: int
    num_C_before_phase4: int
    num_C_after_phase4: int
    num_M_C_edges: int
    covered_vertices: int
    coverage_ratio: float
    phase4_solve_seconds: float
    phase5_solve_seconds: float
    total_solve_seconds: float


def compute_end_to_end_run_metrics(
    *,
    phase4_result: Any,
    phase5_result: Any,
    selector_name: str,
    phase4_solve_seconds: float = 0.0,
    phase5_solve_seconds: float = 0.0,
    total_solve_seconds: float | None = None,
) -> EndToEndRunMetrics:
    """
    Compute end-to-end metrics for one selector route.

    This extends Phase4RunMetrics by adding:
      - covered_vertices
      - coverage_ratio
      - phase4_solve_seconds
      - phase5_solve_seconds
      - total_solve_seconds

    For the user's current phase5_runner.py, the Phase 5 result is expected
    to be an MPC4PlusResult-like object whose final solution is stored in
    `solution_graph`.
    """
    phase4_metrics: Phase4RunMetrics = compute_phase4_run_metrics(
        result=phase4_result,
        selector_name=selector_name,
    )

    covered_vertex_set = extract_covered_vertices(phase5_result)
    covered_vertices = len(covered_vertex_set)

    num_nodes = phase4_metrics.num_nodes
    coverage_ratio = 0.0 if num_nodes == 0 else float(covered_vertices) / float(num_nodes)

    phase4_solve_seconds = float(phase4_solve_seconds)
    phase5_solve_seconds = float(phase5_solve_seconds)
    if total_solve_seconds is None:
        total_solve_seconds = phase4_solve_seconds + phase5_solve_seconds
    total_solve_seconds = float(total_solve_seconds)

    return EndToEndRunMetrics(
        graph_id=phase4_metrics.graph_id,
        selector_name=phase4_metrics.selector_name,
        num_nodes=phase4_metrics.num_nodes,
        num_edges=phase4_metrics.num_edges,
        num_phase4_steps=phase4_metrics.num_phase4_steps,
        num_trace_samples=phase4_metrics.num_trace_samples,
        num_C_before_phase4=phase4_metrics.num_C_before_phase4,
        num_C_after_phase4=phase4_metrics.num_C_after_phase4,
        num_M_C_edges=phase4_metrics.num_M_C_edges,
        covered_vertices=covered_vertices,
        coverage_ratio=coverage_ratio,
        phase4_solve_seconds=phase4_solve_seconds,
        phase5_solve_seconds=phase5_solve_seconds,
        total_solve_seconds=total_solve_seconds,
    )


def extract_covered_vertices(phase5_result: Any) -> set[int]:
    """
    Robustly extract final covered vertices from a Phase 5 result object/dict.

    Current priority order:
      1) solution_graph
      2) covered_vertices / solution_vertices
      3) solution_paths / paths
    """
    if isinstance(phase5_result, dict):
        if "solution_graph" in phase5_result:
            return _vertices_from_graph(phase5_result["solution_graph"])
        if "covered_vertices" in phase5_result:
            return _coerce_vertex_set(phase5_result["covered_vertices"])
        if "solution_vertices" in phase5_result:
            return _coerce_vertex_set(phase5_result["solution_vertices"])
        if "solution_paths" in phase5_result:
            return vertices_from_paths(phase5_result["solution_paths"])
        if "paths" in phase5_result:
            return vertices_from_paths(phase5_result["paths"])
    else:
        if hasattr(phase5_result, "solution_graph"):
            return _vertices_from_graph(getattr(phase5_result, "solution_graph"))
        if hasattr(phase5_result, "covered_vertices"):
            return _coerce_vertex_set(getattr(phase5_result, "covered_vertices"))
        if hasattr(phase5_result, "solution_vertices"):
            return _coerce_vertex_set(getattr(phase5_result, "solution_vertices"))
        if hasattr(phase5_result, "solution_paths"):
            return vertices_from_paths(getattr(phase5_result, "solution_paths"))
        if hasattr(phase5_result, "paths"):
            return vertices_from_paths(getattr(phase5_result, "paths"))

    raise ValueError(
        "Unable to extract covered vertices from phase5_result. "
        "Expected one of: solution_graph, covered_vertices, solution_vertices, solution_paths, paths."
    )


def vertices_from_paths(paths: Iterable[Iterable[int]]) -> set[int]:
    vertices: set[int] = set()
    for path in paths:
        for u in path:
            vertices.add(int(u))
    return vertices


def _coerce_vertex_set(vertices: Iterable[int]) -> set[int]:
    return {int(u) for u in vertices}


def _vertices_from_graph(graph: Any) -> set[int]:
    if isinstance(graph, nx.Graph):
        return {int(u) for u in graph.nodes()}
    if hasattr(graph, "nodes"):
        return {int(u) for u in graph.nodes()}
    raise ValueError(f"solution_graph is not a graph-like object: {type(graph)}")