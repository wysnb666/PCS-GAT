from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Phase4RunMetrics:
    graph_id: str
    selector_name: str
    num_nodes: int
    num_edges: int
    num_phase4_steps: int
    num_trace_samples: int
    num_C_before_phase4: int
    num_C_after_phase4: int
    num_M_C_edges: int


def compute_phase4_run_metrics(
    *,
    result: Any,
    selector_name: str,
) -> Phase4RunMetrics:
    """
    Compute lightweight comparable metrics for a Phase 4 pipeline run.

    This is intentionally a first-stage evaluation layer.
    It does NOT yet evaluate final Phase 5 solution quality.
    It only summarizes Phase 4 behavior in a unified format.
    """
    trace_collector = getattr(result, "trace_collector", None)
    trace_samples = trace_collector.samples() if trace_collector is not None else []

    final_state = result.final_state
    return Phase4RunMetrics(
        graph_id=result.graph_id,
        selector_name=selector_name,
        num_nodes=result.G.number_of_nodes(),
        num_edges=result.G.number_of_edges(),
        num_phase4_steps=len(final_state.history),
        num_trace_samples=len(trace_samples),
        num_C_before_phase4=len(result.C_before_phase4),
        num_C_after_phase4=len(result.C_after_phase4),
        num_M_C_edges=len(result.M_C),
    )