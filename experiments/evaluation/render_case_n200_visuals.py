from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import networkx as nx

from MPC4plus.experiments.datasets.graph_dataset_io import (
    load_graph_records_json,
    load_split_manifest_json,
    record_to_graph,
    select_records_by_ids,
)
from MPC4plus.experiments.run_experiment import (
    load_vertex_gnn_model,
    load_xgb_metadata,
    load_xgb_ranker,
)
from MPC4plus.experiments.baselines.rule_runner import run_rule_phase4_pipeline
from MPC4plus.experiments.baselines.heuristic_runner import run_heuristic_phase4_pipeline
from MPC4plus.experiments.baselines.xgb_runner import run_xgb_phase4_pipeline
from MPC4plus.experiments.models.vertex_gnn_runner import run_vertex_gnn_phase4_pipeline
from MPC4plus.experiments.evaluation.end_to_end_compare import solve_phase5_from_phase4_result



GRAPHS_JSON = "MPC4plus/artifacts/datasets/large_graph_zeroshot_single_n200_case_s42/graphs.json"
SPLIT_JSON = "MPC4plus/artifacts/datasets/large_graph_zeroshot_single_n200_case_s42/split.json"
GRAPH_ID = "largezero_er_n200_p0p025_001"

XGB_MODEL = "MPC4plus/artifacts/models/midscale_generalization_mix_b1_s42_fix1/xgb_model_lookahead.json"
XGB_METADATA = "MPC4plus/artifacts/models/midscale_generalization_mix_b1_s42_fix1/xgb_metadata_lookahead.json"
GNN_CHECKPOINT = "MPC4plus/artifacts/models/midscale_generalization_mix_b1_s42_fix1/vertex_gnn_lookahead.pt"

CPSAT_900S_JSON = "MPC4plus/artifacts/results/large_graph_zeroshot_single_n200_case_s42/cpsat_results_900s.json"

OUTPUT_DIR = "MPC4plus/artifacts/figures/case_n200_visuals"
DEVICE = "cuda"   # 没有 GPU 就改成 "cpu"
BACKEND = "ilp"


# =========================
# 辅助函数
# =========================

def edge_set_of_graph(G: nx.Graph) -> set[tuple[int, int]]:
    return {tuple(sorted(e)) for e in G.edges()}


def load_target_graph(graphs_json: str, split_json: str, graph_id: str) -> nx.Graph:
    records = load_graph_records_json(graphs_json)
    manifest = load_split_manifest_json(split_json)
    selected = select_records_by_ids(records, manifest.test_ids)
    by_id = {r.graph_id: r for r in selected}
    if graph_id not in by_id:
        raise ValueError(f"graph_id {graph_id} not found in test split")
    return record_to_graph(by_id[graph_id])


def solve_rule_and_gnn(
    G: nx.Graph,
    *,
    graph_id: str,
    xgb_model_path: str,
    xgb_metadata_path: str,
    gnn_checkpoint_path: str,
    backend: str = "ilp",
    device: str = "cpu",
):
    # 这里只是为了和你当前项目结构一致，把所有 route 的 loader 都准备好。
    # 真正画图只需要 rule 和 gnn。
    xgb_metadata = load_xgb_metadata(xgb_metadata_path)
    xgb_feature_names = list(xgb_metadata.get("feature_names", []))
    xgb_label_name = str(xgb_metadata.get("label_name", "label__selected_by_rule"))
    ranker = load_xgb_ranker(
        model_path=xgb_model_path,
        feature_names=xgb_feature_names,
        label_name=xgb_label_name,
    )
    gnn_model = load_vertex_gnn_model(
        checkpoint_path=gnn_checkpoint_path,
        device=device,
    )

    def rule_phase4_solver(G_sub: nx.Graph):
        return run_rule_phase4_pipeline(
            G_sub,
            graph_id=graph_id,
            backend=backend,
            enable_trace=True,
        )

    def heuristic_phase4_solver(G_sub: nx.Graph):
        return run_heuristic_phase4_pipeline(
            G_sub,
            graph_id=graph_id,
            backend=backend,
            enable_trace=True,
        )

    def xgb_phase4_solver(G_sub: nx.Graph):
        return run_xgb_phase4_pipeline(
            G_sub,
            ranker=ranker,
            graph_id=graph_id,
            backend=backend,
            enable_trace=True,
        )

    def gnn_phase4_solver(G_sub: nx.Graph):
        return run_vertex_gnn_phase4_pipeline(
            G_sub,
            model=gnn_model,
            graph_id=graph_id,
            backend=backend,
            enable_trace=True,
            device=device,
        )

    # 真正需要的两个结果
    rule_phase4 = rule_phase4_solver(G)
    rule_phase5 = solve_phase5_from_phase4_result(
        rule_phase4,
        phase4_solver_fn=rule_phase4_solver,
    )

    gnn_phase4 = gnn_phase4_solver(G)
    gnn_phase5 = solve_phase5_from_phase4_result(
        gnn_phase4,
        phase4_solver_fn=gnn_phase4_solver,
    )

    # 可选：如果你后面要多比较 heuristic/xgb，也留在这里
    heuristic_phase4 = heuristic_phase4_solver(G)
    heuristic_phase5 = solve_phase5_from_phase4_result(
        heuristic_phase4,
        phase4_solver_fn=heuristic_phase4_solver,
    )

    xgb_phase4 = xgb_phase4_solver(G)
    xgb_phase5 = solve_phase5_from_phase4_result(
        xgb_phase4,
        phase4_solver_fn=xgb_phase4_solver,
    )

    return {
        "rule": rule_phase5.solution_graph,
        "gnn": gnn_phase5.solution_graph,
        "heuristic": heuristic_phase5.solution_graph,
        "xgb": xgb_phase5.solution_graph,
    }


def load_cpsat_opt_solution_graph(cpsat_json_path: str) -> nx.Graph:
    payload = json.loads(Path(cpsat_json_path).read_text(encoding="utf-8"))

    # 兼容你现在的 run_cpsat_pilot.py 输出格式
    # results 是长度为 1 的 list
    if "results" in payload:
        row = payload["results"][0]
    else:
        row = payload

    selected_paths = row.get("selected_paths", [])
    H = nx.Graph()
    for path in selected_paths:
        for u, v in zip(path[:-1], path[1:]):
            H.add_edge(int(u), int(v))
    return H


def draw_solution_view(
    G: nx.Graph,
    pos: dict[int, tuple[float, float]],
    solution_graph: nx.Graph,
    output_path: str,
    *,
    title: str,
):
    plt.figure(figsize=(8, 8))

    # 原图背景
    nx.draw_networkx_edges(
        G,
        pos,
        edge_color="#d9d9d9",
        width=0.5,
        alpha=0.55,
    )
    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=18,
        node_color="#f0f0f0",
        edgecolors="none",
        alpha=0.9,
    )

    # 解高亮
    sol_edges = list(solution_graph.edges())
    sol_nodes = list(solution_graph.nodes())

    nx.draw_networkx_edges(
        solution_graph,
        pos,
        edgelist=sol_edges,
        edge_color="#2b2b2b",
        width=2.3,
        alpha=0.95,
    )
    nx.draw_networkx_nodes(
        solution_graph,
        pos,
        nodelist=sol_nodes,
        node_size=28,
        node_color="#2b2b2b",
        edgecolors="none",
        alpha=0.95,
    )

    plt.title(title, fontsize=13)
    plt.axis("off")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def draw_difference_view(
    G: nx.Graph,
    pos: dict[int, tuple[float, float]],
    rule_sol: nx.Graph,
    gnn_sol: nx.Graph,
    cpsat_sol: nx.Graph | None,
    output_path: str,
    *,
    title: str,
):
    plt.figure(figsize=(8, 8))

    # 原图背景
    nx.draw_networkx_edges(
        G,
        pos,
        edge_color="#e0e0e0",
        width=0.45,
        alpha=0.5,
    )
    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=16,
        node_color="#f5f5f5",
        edgecolors="none",
        alpha=0.9,
    )

    rule_nodes = set(rule_sol.nodes())
    gnn_nodes = set(gnn_sol.nodes())

    gained_nodes = sorted(gnn_nodes - rule_nodes)
    lost_nodes = sorted(rule_nodes - gnn_nodes)

    rule_edges = edge_set_of_graph(rule_sol)
    gnn_edges = edge_set_of_graph(gnn_sol)

    gained_edges = sorted(gnn_edges - rule_edges)
    lost_edges = sorted(rule_edges - gnn_edges)

    # 先画 rule 独有（浅）
    if lost_edges:
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=lost_edges,
            edge_color="#8c8c8c",
            width=1.8,
            alpha=0.65,
            style="dashed",
        )
    if lost_nodes:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=lost_nodes,
            node_size=34,
            node_color="#8c8c8c",
            edgecolors="none",
            alpha=0.75,
        )

    # 再画 GNN 独有（深）
    if gained_edges:
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=gained_edges,
            edge_color="#111111",
            width=2.8,
            alpha=0.95,
        )
    if gained_nodes:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=gained_nodes,
            node_size=44,
            node_color="#111111",
            edgecolors="none",
            alpha=0.95,
        )

    # 可选：如果你想强调 OPT 与 GNN 的差距，也可轻微标出来
    if cpsat_sol is not None:
        opt_nodes = set(cpsat_sol.nodes())
        missing_vs_opt = sorted(opt_nodes - gnn_nodes)
        if missing_vs_opt:
            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=missing_vs_opt,
                node_size=52,
                node_color="none",
                edgecolors="#111111",
                linewidths=1.2,
                alpha=0.95,
            )

    plt.title(title, fontsize=13)
    plt.axis("off")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def main():
    G = load_target_graph(GRAPHS_JSON, SPLIT_JSON, GRAPH_ID)

    # 统一布局：三张图必须使用同一套 pos
    pos = nx.spring_layout(G, seed=42, k=1.0 / (G.number_of_nodes() ** 0.5), iterations=300)

    solutions = solve_rule_and_gnn(
        G,
        graph_id=GRAPH_ID,
        xgb_model_path=XGB_MODEL,
        xgb_metadata_path=XGB_METADATA,
        gnn_checkpoint_path=GNN_CHECKPOINT,
        backend=BACKEND,
        device=DEVICE,
    )

    rule_sol = solutions["rule"]
    gnn_sol = solutions["gnn"]

    cpsat_sol = None
    cpsat_json = Path(CPSAT_900S_JSON)
    if cpsat_json.exists():
        cpsat_sol = load_cpsat_opt_solution_graph(str(cpsat_json))

    outdir = Path(OUTPUT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)

    draw_solution_view(
        G,
        pos,
        rule_sol,
        str(outdir / "case_n200_rule.pdf"),
        title="Canonical Rule (192 covered vertices)",
    )

    draw_solution_view(
        G,
        pos,
        gnn_sol,
        str(outdir / "case_n200_gnn.pdf"),
        title="Vertex GNN (198 covered vertices)",
    )

    draw_difference_view(
        G,
        pos,
        rule_sol,
        gnn_sol,
        cpsat_sol,
        str(outdir / "case_n200_diff.pdf"),
        title="Difference view (GNN gains +6 vertices; OPT = 200)",
    )

    print("[OK] Saved:")
    print(" -", outdir / "case_n200_rule.pdf")
    print(" -", outdir / "case_n200_gnn.pdf")
    print(" -", outdir / "case_n200_diff.pdf")


if __name__ == "__main__":
    main()