from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

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


# =========================
# 路径与配置
# =========================
GRAPHS_JSON = "MPC4plus/artifacts/datasets/large_graph_zeroshot_single_n200_case_s42/graphs.json"
SPLIT_JSON = "MPC4plus/artifacts/datasets/large_graph_zeroshot_single_n200_case_s42/split.json"
GRAPH_ID = "largezero_er_n200_p0p025_001"

GNN_CHECKPOINT = "MPC4plus/artifacts/models/midscale_generalization_mix_b1_s42_fix1/vertex_gnn_lookahead.pt"
CPSAT_900S_JSON = "MPC4plus/artifacts/results/large_graph_zeroshot_single_n200_case_s42/cpsat_results_900s.json"

OUTPUT_DIR = "MPC4plus/artifacts/figures/case_n200_visuals_pubstyle4"

DEVICE = "cpu"   # 如果你想用 GPU，改成 "cuda"
BACKEND = "ilp"
FIG_DPI = 500

# 路径长度颜色（按 4 / 5 / 6 / >=7 分类）
PATH_LEN_COLORS = {
    4: "#2c7fb8",  # blue
    5: "#33a02c",  # green
    6: "#9467bd",  # purple
    7: "#e377c2",  # pink for >= 7
}

BG_EDGE_COLOR = "#cfcfcf"         # 背景边浅灰
COVERED_LABEL_COLOR = "#111111"   # 覆盖节点的内部标签颜色
UNCOVERED_NODE_COLOR = "#d62728"  # 未覆盖顶点高亮
UNCOVERED_LABEL_COLOR = "#8b0000" # 未覆盖顶点标签
RULE_ONLY_COLOR = "#9e9e9e"       # Difference 中 Rule 独有边
OPT_MISS_COLOR = "#d62728"        # Difference 中相对 OPT 仍未覆盖
TEXT_BOX_FACE = "#ffffff"
TEXT_BOX_EDGE = "#bdbdbd"


# =========================
# 数据辅助函数
# =========================
def load_target_graph(graphs_json: str, split_json: str, graph_id: str) -> nx.Graph:
    records = load_graph_records_json(graphs_json)
    manifest = load_split_manifest_json(split_json)
    selected = select_records_by_ids(records, manifest.test_ids)
    by_id = {r.graph_id: r for r in selected}
    if graph_id not in by_id:
        raise ValueError(f"graph_id {graph_id} not found in test split")
    return record_to_graph(by_id[graph_id])


def edge_key(u: int, v: int) -> Tuple[int, int]:
    a, b = int(u), int(v)
    return (a, b) if a <= b else (b, a)


def edge_set(G: nx.Graph) -> set[Tuple[int, int]]:
    return {edge_key(u, v) for u, v in G.edges()}


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


def compute_layout(G: nx.Graph) -> Dict[int, Tuple[float, float]]:
    pos0 = nx.spring_layout(
        G,
        seed=23,
        k=2.5 / (G.number_of_nodes() ** 0.5),
        iterations=600,
    )
    pos = nx.kamada_kawai_layout(G, pos=pos0)
    return {int(k): (float(v[0]), float(v[1])) for k, v in pos.items()}


# =========================
# 路径组件处理
# =========================
def component_ordered_nodes(H: nx.Graph, nodes: Iterable[int]) -> List[int]:
    sub = H.subgraph(nodes).copy()
    deg1 = [int(v) for v, d in sub.degree() if d == 1]
    if len(deg1) >= 1:
        start = min(deg1)
    else:
        start = min(int(v) for v in sub.nodes())

    order = [start]
    prev = None
    cur = start

    while True:
        nbrs = [int(x) for x in sub.neighbors(cur) if int(x) != prev]
        if not nbrs:
            break
        nxt = nbrs[0]
        order.append(nxt)
        prev, cur = cur, nxt
        if len(order) > sub.number_of_nodes() + 2:
            break

    if len(order) != sub.number_of_nodes():
        return sorted(int(v) for v in sub.nodes())
    return order


def extract_path_components(sol: nx.Graph) -> List[dict]:
    comps = []
    for nodes in nx.connected_components(sol):
        ordered = component_ordered_nodes(sol, nodes)
        length = len(ordered)
        comps.append({
            "nodes": ordered,
            "edges": [edge_key(u, v) for u, v in zip(ordered[:-1], ordered[1:])],
            "length": length,
            "color": PATH_LEN_COLORS[7] if length >= 7 else PATH_LEN_COLORS.get(length, PATH_LEN_COLORS[7]),
        })
    comps.sort(key=lambda x: (x["length"], x["nodes"][0]))
    return comps


# =========================
# 绘图辅助
# =========================
def add_text_box(ax, title: str, covered: int, total: int) -> None:
    uncovered = total - covered
    txt = f"{title}\nCovered {covered} / {total}\nUncovered {uncovered}"
    ax.text(
        0.02, 0.98, txt,
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=10.5,
        bbox=dict(boxstyle="round,pad=0.35", facecolor=TEXT_BOX_FACE, edgecolor=TEXT_BOX_EDGE, alpha=0.97),
    )


def add_length_legend(ax) -> None:
    handles = [
        Line2D([0], [0], color=PATH_LEN_COLORS[4], lw=2.0, marker="o", markersize=7,
               markerfacecolor=PATH_LEN_COLORS[4], markeredgecolor="black", label="Length 4"),
        Line2D([0], [0], color=PATH_LEN_COLORS[5], lw=2.0, marker="o", markersize=7,
               markerfacecolor=PATH_LEN_COLORS[5], markeredgecolor="black", label="Length 5"),
        Line2D([0], [0], color=PATH_LEN_COLORS[6], lw=2.0, marker="o", markersize=7,
               markerfacecolor=PATH_LEN_COLORS[6], markeredgecolor="black", label="Length 6"),
        Line2D([0], [0], color=PATH_LEN_COLORS[7], lw=2.0, marker="o", markersize=7,
               markerfacecolor=PATH_LEN_COLORS[7], markeredgecolor="black", label="Length 7"),
        Line2D([0], [0], color="none", lw=0, marker="o", markersize=8,
               markerfacecolor=UNCOVERED_NODE_COLOR, markeredgecolor="black", label="Uncovered vertex"),
    ]
    ax.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor=TEXT_BOX_EDGE,
        fontsize=9.2,
    )


def draw_background(ax, G: nx.Graph, pos) -> None:
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color=BG_EDGE_COLOR,
        width=0.4,
        alpha=0.55,
    )


def draw_center_labels(ax, pos, nodes: Iterable[int], size: int = 5) -> None:
    labels = {int(v): str(int(v)) for v in nodes}
    text_items = nx.draw_networkx_labels(
        nx.Graph(),
        pos,
        labels=labels,
        font_size=size,
        font_color=COVERED_LABEL_COLOR,
        font_weight="normal",
        ax=ax,
    )
    for txt in text_items.values():
        txt.set_path_effects([pe.withStroke(linewidth=1.6, foreground="white")])


def draw_uncovered_labels_above(ax, pos, nodes: Iterable[int]) -> None:
    for v in nodes:
        x, y = pos[int(v)]
        ax.text(
            x, y + 0.035, str(int(v)),
            fontsize=5,
            color=UNCOVERED_LABEL_COLOR,
            ha="center", va="bottom",
            path_effects=[pe.withStroke(linewidth=1.8, foreground="white")],
            zorder=20,
        )


# =========================
# 主绘图函数
# =========================
def draw_solution_panel(ax, G: nx.Graph, pos, sol: nx.Graph, title: str) -> None:
    draw_background(ax, G, pos)

    comps = extract_path_components(sol)
    covered_nodes = set(int(v) for v in sol.nodes())
    all_nodes = set(int(v) for v in G.nodes())
    uncovered_nodes = sorted(all_nodes - covered_nodes)

    # 路径按长度着色，边线适中，不要太粗
    for comp in comps:
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edgelist=comp["edges"],
            edge_color=comp["color"],
            width=1.2,
            alpha=0.98,
        )
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            nodelist=comp["nodes"],
            node_size=80,
            node_color=comp["color"],
            edgecolors="black",
            linewidths=0.6,
            alpha=0.98,
        )

    # 覆盖节点标签写在内部
    draw_center_labels(ax, pos, covered_nodes, size=5)

    # 未覆盖点高亮，标签放在顶点上方
    if uncovered_nodes:
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            nodelist=uncovered_nodes,
            node_size=90,
            node_color=UNCOVERED_NODE_COLOR,
            edgecolors="black",
            linewidths=0.7,
            alpha=0.98,
        )
        draw_uncovered_labels_above(ax, pos, uncovered_nodes)

    add_text_box(ax, title, len(covered_nodes), G.number_of_nodes())
    add_length_legend(ax)
    ax.set_axis_off()


def draw_difference_panel(ax, G: nx.Graph, pos, rule_sol: nx.Graph, gnn_sol: nx.Graph, opt_sol: nx.Graph | None) -> None:
    draw_background(ax, G, pos)

    rule_es = edge_set(rule_sol)
    gnn_es = edge_set(gnn_sol)
    gnn_only_es = gnn_es - rule_es
    rule_only_es = rule_es - gnn_es

    rule_nodes = set(int(v) for v in rule_sol.nodes())
    gnn_nodes = set(int(v) for v in gnn_sol.nodes())
    gained_nodes = sorted(gnn_nodes - rule_nodes)

    missed_vs_opt: List[int] = []
    if opt_sol is not None:
        opt_nodes = set(int(v) for v in opt_sol.nodes())
        missed_vs_opt = sorted(opt_nodes - gnn_nodes)

    # Difference 中把 GNN 独有边按它所属路径长度着色
    gnn_comps = extract_path_components(gnn_sol)
    for comp in gnn_comps:
        comp_edge_set = set(comp["edges"])
        diff_edges = sorted(comp_edge_set & gnn_only_es)
        diff_nodes = sorted(set(sum(([u, v] for u, v in diff_edges), []))) if diff_edges else []
        if diff_edges:
            nx.draw_networkx_edges(
                G, pos, ax=ax,
                edgelist=diff_edges,
                edge_color=comp["color"],
                width=1.0,
                alpha=0.98,
            )
            nx.draw_networkx_nodes(
                G, pos, ax=ax,
                nodelist=diff_nodes,
                node_size=80,
                node_color=comp["color"],
                edgecolors="black",
                linewidths=0.6,
                alpha=0.98,
            )
            draw_center_labels(ax, pos, diff_nodes, size=8)

    if rule_only_es:
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edgelist=sorted(rule_only_es),
            edge_color=RULE_ONLY_COLOR,
            width=1.0,
            style="dashed",
            alpha=0.9,
        )

    if missed_vs_opt:
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            nodelist=missed_vs_opt,
            node_size=90,
            node_color="white",
            edgecolors=OPT_MISS_COLOR,
            linewidths=1.0,
            alpha=1.0,
        )
        draw_uncovered_labels_above(ax, pos, missed_vs_opt)

    txt = f"Difference view\nPCS-GNN gains {len(gained_nodes)} vertices\nStill missing vs OPT: {len(missed_vs_opt)}"
    ax.text(
        0.02, 0.98, txt,
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=10.5,
        bbox=dict(boxstyle="round,pad=0.35", facecolor=TEXT_BOX_FACE, edgecolor=TEXT_BOX_EDGE, alpha=0.97),
    )

    legend_handles = [
        Line2D([0], [0], color=PATH_LEN_COLORS[4], lw=2.0, marker="o", markersize=7,
               markerfacecolor=PATH_LEN_COLORS[4], markeredgecolor="black", label="Diff edge in length 4 path"),
        Line2D([0], [0], color=PATH_LEN_COLORS[5], lw=2.0, marker="o", markersize=7,
               markerfacecolor=PATH_LEN_COLORS[5], markeredgecolor="black", label="Diff edge in length 5 path"),
        Line2D([0], [0], color=PATH_LEN_COLORS[6], lw=2.0, marker="o", markersize=7,
               markerfacecolor=PATH_LEN_COLORS[6], markeredgecolor="black", label="Diff edge in length 6 path"),
        Line2D([0], [0], color=PATH_LEN_COLORS[7], lw=2.0, marker="o", markersize=7,
               markerfacecolor=PATH_LEN_COLORS[7], markeredgecolor="black", label="Diff edge in length 7 path"),
        Line2D([0], [0], color=RULE_ONLY_COLOR, lw=1.3, linestyle="--", label="Rule-only segment"),
        Line2D([0], [0], color="none", lw=0, marker="o", markersize=8,
               markerfacecolor="white", markeredgewidth=1.8, markeredgecolor=OPT_MISS_COLOR,
               label="Still missing vs OPT"),
    ]
    ax.legend(
        legend_handles,
        [h.get_label() for h in legend_handles],
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor=TEXT_BOX_EDGE,
        fontsize=8.6,
    )
    ax.set_axis_off()


def save_figure(fig, out_base: Path) -> None:
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
    opt_sol = load_cpsat_opt_solution_graph(CPSAT_900S_JSON) if Path(CPSAT_900S_JSON).exists() else None

    def print_path_statistics(sol: nx.Graph, name: str) -> None:
        comps = extract_path_components(sol)
        len_counts = {4: 0, 5: 0, 6: 0, 7: 0}  # 7 代表 >=7
        long_path_lengths = []  # 记录长度 >=7 的具体顶点数

        for comp in comps:
            L = comp["length"]
            if L == 4:
                len_counts[4] += 1
            elif L == 5:
                len_counts[5] += 1
            elif L == 6:
                len_counts[6] += 1
            else:  # L >= 7
                len_counts[7] += 1
                long_path_lengths.append(L)

        print(f"\n=== {name} Path Statistics ===")
        print(f"Length 4 paths:  {len_counts[4]}")
        print(f"Length 5 paths:  {len_counts[5]}")
        print(f"Length 6 paths:  {len_counts[6]}")
        print(f"Length ≥7 paths: {len_counts[7]}")
        if long_path_lengths:
            print(f"  Vertex counts of ≥7 paths: {long_path_lengths}")
        else:
            print("  No paths with length ≥7.")

    # 在 main 中调用
    print_path_statistics(rule_sol, "Rule-based")
    print_path_statistics(gnn_sol, "GNN-based")
    if opt_sol is not None:
        print_path_statistics(opt_sol, "CP-SAT OPT")

    pos = compute_layout(G)

    # 画布加宽，给右侧图例留空间
    fig, ax = plt.subplots(figsize=(10.6, 7.2), facecolor="white")
    draw_solution_panel(ax, G, pos, rule_sol, "Canonical Rule")
    save_figure(fig, outdir / "case_n200_rule_pubstyle")

    fig, ax = plt.subplots(figsize=(10.6, 7.2), facecolor="white")
    draw_solution_panel(ax, G, pos, gnn_sol, "PCS-GNN")
    save_figure(fig, outdir / "case_n200_gnn_pubstyle")

    fig, ax = plt.subplots(figsize=(10.6, 7.2), facecolor="white")
    draw_difference_panel(ax, G, pos, rule_sol, gnn_sol, opt_sol)
    save_figure(fig, outdir / "case_n200_diff_pubstyle")

    print("[OK] Saved files to", outdir)


if __name__ == "__main__":
    main()