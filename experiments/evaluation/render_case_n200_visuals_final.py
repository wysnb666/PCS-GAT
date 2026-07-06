from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Iterable

# Project root should be the directory that contains the MPC4plus package.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
import networkx as nx

from MPC4plus.experiments.datasets.graph_dataset_io import (
    load_graph_records_json,
    load_split_manifest_json,
    record_to_graph,
    select_records_by_ids,
)
from MPC4plus.experiments.run_experiment import load_vertex_gnn_model
from MPC4plus.experiments.baselines.rule_runner import run_rule_phase4_pipeline
from MPC4plus.experiments.models.vertex_gnn_runner import run_vertex_gnn_phase4_pipeline
from MPC4plus.experiments.evaluation.end_to_end_compare import solve_phase5_from_phase4_result


GRAPHS_JSON = "MPC4plus/artifacts/datasets/large_graph_zeroshot_single_n200_case_s42/graphs.json"
SPLIT_JSON = "MPC4plus/artifacts/datasets/large_graph_zeroshot_single_n200_case_s42/split.json"
GRAPH_ID = "largezero_er_n200_p0p025_001"
GNN_CHECKPOINT = "MPC4plus/artifacts/models/midscale_generalization_mix_b1_s42_fix1/vertex_gnn_lookahead.pt"
CPSAT_900S_JSON = "MPC4plus/artifacts/results/large_graph_zeroshot_single_n200_case_s42/cpsat_results_900s.json"
OUTPUT_DIR = "MPC4plus/artifacts/figures/case_n200_visuals_final"
DEVICE = "cpu"   # 如果你想用 GPU，再手动改成 "cuda"
BACKEND = "ilp"
FIG_DPI = 400

BG_EDGE = "#d9d9d9"
BG_NODE = "#f2f2f2"
SOL_EDGE = "#1f2937"
SOL_NODE = "#111827"
UNCOVERED = "#e76f51"
GAINED = "#2563eb"
RULE_ONLY = "#9ca3af"
OPT_MISS = "#e76f51"
TEXT_BOX = "#ffffff"


def load_target_graph(graphs_json: str, split_json: str, graph_id: str) -> nx.Graph:
    records = load_graph_records_json(graphs_json)
    manifest = load_split_manifest_json(split_json)
    selected = select_records_by_ids(records, manifest.test_ids)
    by_id = {r.graph_id: r for r in selected}
    if graph_id not in by_id:
        raise ValueError(f"graph_id {graph_id} not found in test split")
    return record_to_graph(by_id[graph_id])


def edge_set(G: nx.Graph) -> set[tuple[int, int]]:
    return {tuple(sorted((int(u), int(v)))) for u, v in G.edges()}


def solve_rule_and_gnn(G: nx.Graph, graph_id: str) -> Dict[str, nx.Graph]:
    def rule_phase4_solver(G_sub: nx.Graph):
        return run_rule_phase4_pipeline(
            G_sub,
            graph_id=graph_id,
            backend=BACKEND,
            enable_trace=True,
        )

    gnn_model = load_vertex_gnn_model(
        checkpoint_path=GNN_CHECKPOINT,
        device=DEVICE,
    )

    def gnn_phase4_solver(G_sub: nx.Graph):
        return run_vertex_gnn_phase4_pipeline(
            G_sub,
            model=gnn_model,
            graph_id=graph_id,
            backend=BACKEND,
            enable_trace=True,
            device=DEVICE,
        )

    rule_phase4 = rule_phase4_solver(G)
    rule_phase5 = solve_phase5_from_phase4_result(rule_phase4, phase4_solver_fn=rule_phase4_solver)

    gnn_phase4 = gnn_phase4_solver(G)
    gnn_phase5 = solve_phase5_from_phase4_result(gnn_phase4, phase4_solver_fn=gnn_phase4_solver)

    return {
        "rule": rule_phase5.solution_graph,
        "gnn": gnn_phase5.solution_graph,
    }


def load_cpsat_opt_solution_graph(cpsat_json_path: str) -> nx.Graph:
    payload = json.loads(Path(cpsat_json_path).read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "results" in payload:
        row = payload["results"][0]
    elif isinstance(payload, list):
        row = payload[0]
    else:
        row = payload

    selected_paths = row.get("selected_paths", [])
    H = nx.Graph()
    for path in selected_paths:
        for u, v in zip(path[:-1], path[1:]):
            H.add_edge(int(u), int(v))
    return H


def compute_layout(G: nx.Graph) -> Dict[int, tuple[float, float]]:
    pos0 = nx.spring_layout(
        G,
        seed=17,
        k=2.2 / (G.number_of_nodes() ** 0.5),
        iterations=500,
    )
    pos = nx.kamada_kawai_layout(G, pos=pos0)
    return {int(k): (float(v[0]), float(v[1])) for k, v in pos.items()}


def draw_background(ax, G: nx.Graph, pos) -> None:
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color=BG_EDGE,
        width=0.35,
        alpha=0.22,
    )
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_size=14,
        node_color=BG_NODE,
        edgecolors="none",
        alpha=0.9,
    )


def add_info_box(ax, title: str, covered: int, total: int) -> None:
    uncovered = total - covered
    txt = f"{title}\nCovered {covered} / {total}\nUncovered {uncovered}"
    ax.text(
        0.02, 0.98, txt,
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=10.5,
        bbox=dict(boxstyle="round,pad=0.35", facecolor=TEXT_BOX, edgecolor="#cccccc", alpha=0.95),
    )


def label_nodes(ax, pos, nodes: Iterable[int], color: str) -> None:
    labels = {int(v): str(int(v)) for v in nodes}
    text_items = nx.draw_networkx_labels(
        nx.Graph(),
        pos,
        labels=labels,
        font_size=8,
        font_color=color,
        font_weight="bold",
        ax=ax,
    )
    for txt in text_items.values():
        txt.set_path_effects([pe.withStroke(linewidth=2.2, foreground="white")])


def draw_solution_panel(ax, G: nx.Graph, pos, sol: nx.Graph, *, panel_title: str) -> None:
    total = G.number_of_nodes()
    covered_nodes = sorted(int(v) for v in sol.nodes())
    uncovered_nodes = sorted(set(int(v) for v in G.nodes()) - set(covered_nodes))

    draw_background(ax, G, pos)

    nx.draw_networkx_edges(
        sol, pos, ax=ax,
        edge_color=SOL_EDGE,
        width=2.25,
        alpha=0.95,
    )
    nx.draw_networkx_nodes(
        sol, pos, ax=ax,
        nodelist=covered_nodes,
        node_size=20,
        node_color=SOL_NODE,
        edgecolors="none",
        alpha=0.96,
    )

    if uncovered_nodes:
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            nodelist=uncovered_nodes,
            node_size=90,
            node_color=UNCOVERED,
            edgecolors="black",
            linewidths=0.9,
            alpha=0.98,
        )
        label_nodes(ax, pos, uncovered_nodes, color="#7f1d1d")

    add_info_box(ax, panel_title, len(covered_nodes), total)
    ax.set_axis_off()


def draw_difference_panel(ax, G: nx.Graph, pos, rule_sol: nx.Graph, gnn_sol: nx.Graph, opt_sol: nx.Graph | None) -> None:
    draw_background(ax, G, pos)

    rule_edges = edge_set(rule_sol)
    gnn_edges = edge_set(gnn_sol)

    shared_edges = sorted(rule_edges & gnn_edges)
    rule_only_edges = sorted(rule_edges - gnn_edges)
    gnn_only_edges = sorted(gnn_edges - rule_edges)

    rule_nodes = set(int(v) for v in rule_sol.nodes())
    gnn_nodes = set(int(v) for v in gnn_sol.nodes())

    gained_nodes = sorted(gnn_nodes - rule_nodes)
    lost_nodes = sorted(rule_nodes - gnn_nodes)
    missed_vs_opt: list[int] = []
    if opt_sol is not None:
        opt_nodes = set(int(v) for v in opt_sol.nodes())
        missed_vs_opt = sorted(opt_nodes - gnn_nodes)

    if shared_edges:
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edgelist=shared_edges,
            edge_color="#cfcfcf",
            width=1.2,
            alpha=0.65,
        )

    if rule_only_edges:
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edgelist=rule_only_edges,
            edge_color=RULE_ONLY,
            width=2.0,
            alpha=0.9,
            style="dashed",
        )

    if gnn_only_edges:
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edgelist=gnn_only_edges,
            edge_color=GAINED,
            width=3.0,
            alpha=0.98,
        )

    if gained_nodes:
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            nodelist=gained_nodes,
            node_size=95,
            node_color=GAINED,
            edgecolors="black",
            linewidths=0.8,
            alpha=0.98,
        )
        label_nodes(ax, pos, gained_nodes, color="#1d4ed8")

    if lost_nodes:
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            nodelist=lost_nodes,
            node_size=72,
            node_color=RULE_ONLY,
            edgecolors="black",
            linewidths=0.7,
            alpha=0.9,
        )

    if missed_vs_opt:
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            nodelist=missed_vs_opt,
            node_size=120,
            node_color="none",
            edgecolors=OPT_MISS,
            linewidths=2.0,
            alpha=1.0,
        )
        label_nodes(ax, pos, missed_vs_opt, color="#7f1d1d")

    ax.text(
        0.02, 0.98,
        f"Difference view\nGNN gains {len(gained_nodes)} vertices\nStill missing vs OPT: {len(missed_vs_opt)}",
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=10.5,
        bbox=dict(boxstyle="round,pad=0.35", facecolor=TEXT_BOX, edgecolor="#cccccc", alpha=0.95),
    )

    handles = [
        Line2D([0], [0], color=GAINED, lw=3.0, marker='o', markersize=7,
               markerfacecolor=GAINED, markeredgecolor='black', label='GNN gains vs Rule'),
        Line2D([0], [0], color=RULE_ONLY, lw=2.0, linestyle='--', label='Rule-only segment'),
        Line2D([0], [0], color=OPT_MISS, lw=0, marker='o', markersize=9,
               markerfacecolor='none', markeredgewidth=2.0, markeredgecolor=OPT_MISS,
               label='Still missing vs OPT'),
    ]
    ax.legend(
        handles=handles,
        loc="lower left",
        fontsize=8.5,
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="#cccccc",
    )

    ax.set_axis_off()


def save_panel(fig, out_base: Path) -> None:
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", dpi=FIG_DPI)
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight", dpi=FIG_DPI)
    plt.close(fig)


def main() -> None:
    outdir = Path(OUTPUT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)

    G = load_target_graph(GRAPHS_JSON, SPLIT_JSON, GRAPH_ID)
    sols = solve_rule_and_gnn(G, GRAPH_ID)
    rule_sol = sols["rule"]
    gnn_sol = sols["gnn"]

    opt_sol = None
    cpsat_path = Path(CPSAT_900S_JSON)
    if cpsat_path.exists():
        opt_sol = load_cpsat_opt_solution_graph(str(cpsat_path))

    pos = compute_layout(G)

    fig, ax = plt.subplots(figsize=(7.2, 7.2), facecolor="white")
    draw_solution_panel(ax, G, pos, rule_sol, panel_title="Canonical Rule")
    save_panel(fig, outdir / "case_n200_rule_final")

    fig, ax = plt.subplots(figsize=(7.2, 7.2), facecolor="white")
    draw_solution_panel(ax, G, pos, gnn_sol, panel_title="Vertex GNN")
    save_panel(fig, outdir / "case_n200_gnn_final")

    fig, ax = plt.subplots(figsize=(7.2, 7.2), facecolor="white")
    draw_difference_panel(ax, G, pos, rule_sol, gnn_sol, opt_sol)
    save_panel(fig, outdir / "case_n200_diff_final")

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.8), facecolor="white")
    draw_solution_panel(axes[0], G, pos, rule_sol, panel_title="Canonical Rule")
    draw_solution_panel(axes[1], G, pos, gnn_sol, panel_title="Vertex GNN")
    draw_difference_panel(axes[2], G, pos, rule_sol, gnn_sol, opt_sol)
    plt.tight_layout(w_pad=1.6)
    save_panel(fig, outdir / "case_n200_triptych_final")

    print("[OK] Saved final publication-style figures to:", outdir)
    for name in [
        "case_n200_rule_final.pdf",
        "case_n200_rule_final.png",
        "case_n200_gnn_final.pdf",
        "case_n200_gnn_final.png",
        "case_n200_diff_final.pdf",
        "case_n200_diff_final.png",
        "case_n200_triptych_final.pdf",
        "case_n200_triptych_final.png",
    ]:
        print(" -", outdir / name)


if __name__ == "__main__":
    main()