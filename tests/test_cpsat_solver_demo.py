from __future__ import annotations

import pytest

pytest.importorskip("ortools")

import networkx as nx

from MPC4plus.experiments.solvers.cpsat_mpc4_solver import (
    enumerate_candidate_paths_order_4_to_7,
    solve_mpc4_with_cpsat,
)


def test_enumerate_candidate_paths_order_4_to_7_on_path5() -> None:
    G = nx.path_graph([1, 2, 3, 4, 5])

    paths = enumerate_candidate_paths_order_4_to_7(G)
    as_tuples = set(paths)

    assert (1, 2, 3, 4) in as_tuples
    assert (2, 3, 4, 5) in as_tuples
    assert (1, 2, 3, 4, 5) in as_tuples
    assert len(paths) == 3


def test_solve_mpc4_with_cpsat_on_path5() -> None:
    G = nx.path_graph([1, 2, 3, 4, 5])

    result = solve_mpc4_with_cpsat(
        G,
        graph_id="path5",
        time_limit_seconds=10.0,
        num_search_workers=1,
    )

    assert result.graph_id == "path5"
    assert result.solver_status in {"OPTIMAL", "FEASIBLE"}
    assert result.objective_value == 5
    assert result.covered_vertices == 5
    assert result.coverage_ratio == 1.0
    assert len(result.selected_paths) == 1
    assert result.selected_path_lengths == [5]


def test_solve_mpc4_with_cpsat_on_two_disjoint_4paths() -> None:
    G = nx.Graph()
    G.add_edges_from(
        [
            (1, 2), (2, 3), (3, 4),
            (5, 6), (6, 7), (7, 8),
        ]
    )

    result = solve_mpc4_with_cpsat(
        G,
        graph_id="two_4paths",
        time_limit_seconds=10.0,
        num_search_workers=1,
    )

    assert result.solver_status in {"OPTIMAL", "FEASIBLE"}
    assert result.objective_value == 8
    assert result.covered_vertices == 8
    assert result.coverage_ratio == 1.0
    assert sorted(result.selected_path_lengths) == [4, 4]