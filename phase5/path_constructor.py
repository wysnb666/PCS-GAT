from __future__ import annotations

import itertools
import networkx as nx

from MPC4plus.phase3.component_meta import ComponentMeta, SatelliteInfo
from MPC4plus.phase5.exact_solver import (
    brute_force_best_single_path_containing,
    union_solution_graphs,
)


def _component_containing_anchor(metas: list[ComponentMeta], anchor: int) -> ComponentMeta | None:
    for meta in metas:
        if anchor in meta.nodes:
            return meta
    return None


def _critical_satellites_for_anchor(meta: ComponentMeta, anchor: int) -> list[SatelliteInfo]:
    out: list[SatelliteInfo] = []
    for sat in meta.satellites:
        if sat.is_critical_satellite and sat.rescue_anchor == anchor:
            out.append(sat)
    return out


def _center_nodes(meta: ComponentMeta) -> set[int]:
    return set(meta.center_nodes or set())


def _anchor_neighbors_in_center(meta: ComponentMeta, anchor: int, G: nx.Graph) -> set[int]:
    """
    Center-side neighbors of the anchor, used to approximate the two local branches
    around a 2-anchor when constructing P_v.
    """
    center = _center_nodes(meta)
    out: set[int] = set()
    if anchor not in center:
        return out
    for u in center:
        if u != anchor and G.has_edge(anchor, u):
            out.add(u)
    return out


def _satellite_attachment_candidates(sat: SatelliteInfo, anchor: int) -> set[int]:
    """
    Nodes in a critical satellite most naturally attached to the anchor-side branch:
    - endpoints of rescue edges incident to the satellite
    - if unavailable, all satellite nodes
    """
    out: set[int] = set()
    for u, v in sat.rescue_edges:
        if u == anchor and v in sat.nodes:
            out.add(v)
        elif v == anchor and u in sat.nodes:
            out.add(u)

    if out:
        return out
    return set(sat.nodes)


def _candidate_required_sets(meta: ComponentMeta, anchor: int, G: nx.Graph) -> list[set[int]]:
    """
    Structured candidate required-node sets for constructing P_v.

    Refinement over the previous version:
    - explicitly tries a 'two-branch around anchor' viewpoint:
      anchor + center-neighbors + one/two critical satellites
    - still keeps all-critical-satellites and center-only candidates
    """
    center = _center_nodes(meta)
    critical_sats = _critical_satellites_for_anchor(meta, anchor)
    center_neighbors = _anchor_neighbors_in_center(meta, anchor, G)

    required_sets: list[set[int]] = []

    # Smallest center-based seed
    base = {anchor} | center
    required_sets.append(set(base))

    # Anchor + immediate center neighbors only
    if center_neighbors:
        required_sets.append({anchor} | center_neighbors)

    # Anchor + center + one critical satellite
    for sat in critical_sats:
        required_sets.append(set(base) | set(sat.nodes))

    # Anchor + center-neighbors + one critical satellite
    for sat in critical_sats:
        required_sets.append({anchor} | center_neighbors | set(sat.nodes))

    # Anchor + center + two critical satellites
    for sat1, sat2 in itertools.combinations(critical_sats, 2):
        required_sets.append(set(base) | set(sat1.nodes) | set(sat2.nodes))

    # Anchor + center-neighbors + two critical satellites
    for sat1, sat2 in itertools.combinations(critical_sats, 2):
        required_sets.append({anchor} | center_neighbors | set(sat1.nodes) | set(sat2.nodes))

    # All critical satellites
    if critical_sats:
        all_nodes = set(base)
        for sat in critical_sats:
            all_nodes.update(sat.nodes)
        required_sets.append(all_nodes)

    # More branch-oriented candidate:
    # choose one attachment node from one satellite and one from another satellite,
    # then require anchor + center + those satellites.
    if len(critical_sats) >= 2:
        for sat1, sat2 in itertools.combinations(critical_sats, 2):
            attach1 = _satellite_attachment_candidates(sat1, anchor)
            attach2 = _satellite_attachment_candidates(sat2, anchor)
            for x in attach1:
                for y in attach2:
                    required_sets.append(set(base) | set(sat1.nodes) | set(sat2.nodes) | {x, y})

    # Deduplicate while preserving order
    unique: list[set[int]] = []
    seen: set[frozenset[int]] = set()
    for rs in required_sets:
        key = frozenset(rs)
        if key not in seen:
            seen.add(key)
            unique.append(rs)
    return unique


def _best_structured_local_path(
    G: nx.Graph,
    meta: ComponentMeta,
    anchor: int,
) -> nx.Graph:
    """
    Prefer structured local constructions before falling back to the whole component.
    """
    best = nx.Graph()
    best_size = 0

    for required in _candidate_required_sets(meta, anchor, G):
        sub = G.subgraph(sorted(required)).copy()
        path = brute_force_best_single_path_containing(
            sub,
            required_vertices={anchor},
            min_vertices=5,
        )
        if path.number_of_nodes() > best_size:
            best = path
            best_size = path.number_of_nodes()

    return best


def construct_Pv_for_anchor(
    G: nx.Graph,
    metas: list[ComponentMeta],
    anchor: int,
) -> nx.Graph:
    """
    Construct a 5+-path P_v for one v in R_c.

    Current refined strategy:
    1. try a structured local construction around the 2-anchor using:
       - center nodes
       - center-side neighbors
       - critical satellites whose rescue-anchor is v
    2. if that fails, fall back to the whole critical component containing v.
    """
    meta = _component_containing_anchor(metas, anchor)
    if meta is None:
        return nx.Graph()

    best_local = _best_structured_local_path(G, meta, anchor)
    if best_local.number_of_nodes() >= 5:
        return best_local

    sub_full = G.subgraph(sorted(meta.nodes)).copy()
    path = brute_force_best_single_path_containing(
        sub_full,
        required_vertices={anchor},
        min_vertices=5,
    )
    return path


def construct_all_Pv(
    G: nx.Graph,
    metas: list[ComponentMeta],
    Rc: set[int],
) -> nx.Graph:
    graphs = []
    for anchor in sorted(Rc):
        pv = construct_Pv_for_anchor(G, metas, anchor)
        if pv.number_of_nodes() > 0:
            graphs.append(pv)
    return union_solution_graphs(graphs)