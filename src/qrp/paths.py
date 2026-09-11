"""Candidate path generation.

The placement model chooses, for each user pair, one path from a candidate
set. A path here is a *logical* path: a sequence of nodes that will host
repeaters, joined by elementary links. Nodes lying between two consecutive
repeaters are bypassed and act as optical switches, which is the paper's own
reading of what an unselected candidate site does.

Pouryousef enumerates k shortest paths with k between 1000 and 4000 and notes
this is a computational rather than principled choice. No affordable
candidate set is exact for the network problem either, and this module does
not claim otherwise.

Why no reduction is exact
-------------------------
Under Eq. (2) the utility of a single path depends only on its hop count and
its worst link, so for one user pair on its own the paths worth considering
form the Pareto frontier over (hops, worst link). The network problem is
different. Repeater memory (Eq. 9) and the repeater budget (Eq. 12) are shared
between pairs, so a path that is worse on its own can be the one the global
optimum needs: it may pass through repeaters another pair already uses, or
avoid a node whose memory is taken. The low-W*p rate models and the
waiting-time analyses also depend on every link, not only the worst one.
tests/test_candidate_paths.py contains a network where dropping a locally
dominated path loses a user pair.

Strategies
----------
``candidates`` (default)
    The union of three families, all simple paths by construction:

    1. The exact Pareto frontier over (hops, worst link). For each distinct
       link length L, a breadth-first search over links no longer than L
       gives the fewest hops achievable; every L at which that number drops
       is a frontier point. At each point the generator keeps up to
       ``tie_cap`` paths per hop count, from the minimum up to ``hop_slack``
       extra hops, so equal-quality alternatives through different repeaters
       survive.
    2. The ``k_simple`` shortest simple paths by total length.
    3. The ``k_simple`` shortest simple paths by summed loss weight
       10^(alpha l / 10), roughly the expected attempts per link, which is
       what the low-W*p rate models and the waiting time respond to.

    Families 2 and 3 count only paths within ``max_hops``, and the limit acts
    inside Yen's search rather than as a filter on its output. The cheapest
    paths by loss use many short links; filtered afterwards, the first 120 on
    the 40 km site grid were all over 20 hops and the family kept nothing.

    Nothing is dropped for being dominated. The set is a heuristic, checked
    against exhaustive enumeration on small random networks in the tests.

``legacy``
    The generator used for every result committed before September 2026:
    one bottleneck-optimal and one shortest-total path per exact hop count
    from a dynamic program over (hops, node), then dominance pruning. It
    discards ties, loses hop counts whose best walk revisits a node, and can
    drop globally necessary paths. Kept only to reproduce those results.

``exhaustive``
    Every simple path within the hop limit. For small validation graphs only;
    it refuses to run past ``max_exhaustive`` paths per pair.
"""

from __future__ import annotations

import heapq
import itertools
import math
from collections import deque
from dataclasses import dataclass

import networkx as nx

from .topology import Topology

STRATEGIES = ("candidates", "legacy", "exhaustive")


@dataclass(frozen=True)
class CandidatePath:
    """One logical path between two end nodes.

    Attributes
    ----------
    source, target
        The end nodes.
    nodes
        Full node sequence including both endpoints. Consecutive entries are
        joined by one elementary link.
    link_lengths_km
        Length of each elementary link. ``len(link_lengths_km) == hops``.
    repeaters
        Interior nodes, which must host a repeater for this path to be usable.
    """

    source: str
    target: str
    nodes: tuple[str, ...]
    link_lengths_km: tuple[float, ...]
    repeaters: tuple[str, ...]

    @property
    def hops(self) -> int:
        return len(self.link_lengths_km)

    @property
    def max_link_km(self) -> float:
        return max(self.link_lengths_km)

    @property
    def total_km(self) -> float:
        return sum(self.link_lengths_km)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"CandidatePath({self.source}->{self.target}, h={self.hops}, "
            f"max_link={self.max_link_km:.0f}km, n_rep={len(self.repeaters)})"
        )


# --------------------------------------------------------------------------
# Logical link graph
# --------------------------------------------------------------------------


def build_logical_graph(topology: Topology, max_link_km: float) -> nx.Graph:
    """Graph whose edges are elementary links, bypassing intermediate nodes.

    An edge (u, v) exists when the physical fibre distance between u and v is
    at most ``max_link_km``. Its length is that fibre distance.

    ``max_link_km`` is a modelling cap, not physics. Without it every node
    pair would be joined and the search would consider links so lossy that no
    solver time should be spent on them. Set it well above the distance at
    which the link success probability becomes negligible.
    """
    physical = topology.graph
    logical = nx.Graph()
    logical.add_nodes_from(physical.nodes(data=True))

    for source in physical.nodes:
        lengths, paths = nx.single_source_dijkstra(
            physical, source, cutoff=max_link_km, weight="length_km"
        )
        for target, distance in lengths.items():
            if target == source or logical.has_edge(source, target):
                continue
            bypassed = tuple(paths[target][1:-1])
            logical.add_edge(source, target, length_km=distance, bypassed=bypassed)

    return logical


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------


def enumerate_paths(
    topology: Topology,
    pairs: list[tuple[str, str]],
    max_link_km: float = 300.0,
    max_hops: int = 20,
    logical: nx.Graph | None = None,
    strategy: str = "candidates",
    tie_cap: int = 4,
    hop_slack: int = 1,
    k_simple: int = 6,
    loss_db_per_km: float = 0.35,
    max_exhaustive: int = 200_000,
) -> dict[tuple[str, str], list[CandidatePath]]:
    """Build the candidate path set for every demand pair.

    Paths are independent of hardware, so this runs once and is reused across
    every point of a sweep. See the module docstring for what each strategy
    keeps and why none of them is exact for the network problem.
    ``loss_db_per_km`` only shapes the loss-weighted family of the default
    strategy; it does not change any physics.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; choose from {STRATEGIES}")

    if logical is None:
        logical = build_logical_graph(topology, max_link_km)
    elif strategy == "candidates":
        logical = logical.copy()

    if strategy == "candidates":
        nx.set_edge_attributes(
            logical,
            {(u, v): 10.0 ** (loss_db_per_km * d["length_km"] / 10.0)
             for u, v, d in logical.edges(data=True)},
            "loss_weight",
        )

    result: dict[tuple[str, str], list[CandidatePath]] = {}
    for source, target in pairs:
        if strategy == "legacy":
            node_lists = _legacy_node_lists(logical, source, target, max_hops)
            candidates = _to_paths(logical, source, target, node_lists, legacy_order=True)
            candidates = _drop_dominated(candidates)
        elif strategy == "exhaustive":
            node_lists = _exhaustive_node_lists(logical, source, target, max_hops, max_exhaustive)
            candidates = _to_paths(logical, source, target, node_lists)
        else:
            node_lists = _candidate_node_lists(
                logical, source, target, max_hops, tie_cap, hop_slack, k_simple
            )
            candidates = _to_paths(logical, source, target, node_lists)
        result[(source, target)] = candidates

    return result


def _to_paths(logical, source, target, node_lists, legacy_order: bool = False):
    seen: set[tuple[str, ...]] = set()
    out: list[CandidatePath] = []
    for nodes in node_lists:
        key = tuple(nodes)
        if len(key) < 2 or key in seen:
            continue
        seen.add(key)
        lengths = tuple(logical[a][b]["length_km"] for a, b in zip(key[:-1], key[1:]))
        out.append(
            CandidatePath(
                source=source,
                target=target,
                nodes=key,
                link_lengths_km=lengths,
                repeaters=tuple(key[1:-1]),
            )
        )
    if legacy_order:
        out.sort(key=lambda p: (p.hops, p.max_link_km))
    else:
        out.sort(key=lambda p: (p.hops, p.max_link_km, p.total_km, p.nodes))
    return out


# --------------------------------------------------------------------------
# Default strategy
# --------------------------------------------------------------------------


def _candidate_node_lists(logical, source, target, max_hops, tie_cap, hop_slack, k_simple):
    out: list[tuple[str, ...]] = []

    # 1. The Pareto frontier over (hops, worst link), with ties and slack.
    for threshold, fewest in _frontier(logical, source, target, max_hops):
        within = nx.subgraph_view(
            logical,
            filter_edge=lambda u, v, t=threshold: logical[u][v]["length_km"] <= t + 1e-9,
        )
        ceiling = min(fewest + hop_slack, max_hops)
        per_hops: dict[int, int] = {}
        limit = tie_cap * (hop_slack + 1) * 3
        for nodes in _shortest_simple(within, source, target, weight=None, limit=limit):
            hops = len(nodes) - 1
            if hops > ceiling:
                break
            if per_hops.get(hops, 0) >= tie_cap:
                continue
            per_hops[hops] = per_hops.get(hops, 0) + 1
            out.append(tuple(nodes))

    # 2 and 3. Shortest simple paths within the hop limit, by total length and
    # by summed loss. The limit has to act inside the search: the cheapest
    # paths by loss use many short links, and filtering an unlimited ranking
    # afterwards kept nothing at all at 40 km site spacing.
    for weight in ("length_km", "loss_weight"):
        out.extend(_k_shortest_within_hops(logical, source, target, weight, k_simple, max_hops))

    return out


def _frontier(logical, source, target, max_hops) -> list[tuple[float, int]]:
    """Pareto points (worst link, fewest hops) over simple paths, in order."""
    lengths = sorted({d["length_km"] for _, _, d in logical.edges(data=True)})
    points: list[tuple[float, int]] = []
    best = math.inf
    for threshold in lengths:
        hops = _fewest_hops(logical, source, target, threshold, max_hops)
        if hops is not None and hops < best:
            best = hops
            points.append((threshold, hops))
            if hops == 1:
                break
    return points


def _fewest_hops(logical, source, target, threshold, max_hops) -> int | None:
    """Breadth-first search over links no longer than ``threshold``."""
    depth = {source: 0}
    queue = deque([source])
    while queue:
        u = queue.popleft()
        if depth[u] >= max_hops:
            continue
        for v, data in logical[u].items():
            if v in depth or data["length_km"] > threshold + 1e-9:
                continue
            if v == target:
                return depth[u] + 1
            depth[v] = depth[u] + 1
            queue.append(v)
    return None


def _shortest_simple(graph, source, target, weight, limit):
    """Up to ``limit`` simple paths in order of hops or weight (Yen)."""
    try:
        yield from itertools.islice(
            nx.shortest_simple_paths(graph, source, target, weight=weight), limit
        )
    except nx.NetworkXNoPath:
        return


def _k_shortest_within_hops(graph, source, target, weight, k, max_hops):
    """The ``k`` cheapest simple paths with at most ``max_hops`` hops.

    Yen's algorithm with a hop-limited spur search. Every weight is positive,
    so the cheapest walk within a hop limit never revisits a node and each
    spur search is exact. Equal costs go to fewer hops, then to node order,
    so the result does not depend on the order the graph stores its edges.
    """
    first = _cheapest_within_hops(graph, source, target, weight, max_hops, frozenset(), frozenset())
    if first is None:
        return []
    accepted = [first]
    seen = {first}
    heap: list = []
    while len(accepted) < k:
        last = accepted[-1]
        for i in range(len(last) - 1):
            root = last[: i + 1]
            banned_next = frozenset(
                p[i + 1] for p in accepted if len(p) > i + 1 and p[: i + 1] == root
            )
            spur = _cheapest_within_hops(
                graph, root[-1], target, weight, max_hops - i, frozenset(root[:-1]), banned_next
            )
            if spur is None:
                continue
            path = root[:-1] + spur
            if path in seen:
                continue
            seen.add(path)
            heapq.heappush(heap, (_rounded_cost(graph, path, weight), len(path), path))
        if not heap:
            break
        accepted.append(heapq.heappop(heap)[2])
    return accepted


def _rounded_cost(graph, nodes, weight) -> float:
    """Path cost to 12 significant figures, so float noise does not break ties."""
    total = math.fsum(graph[a][b][weight] for a, b in zip(nodes[:-1], nodes[1:]))
    return float(f"{total:.12g}")


def _cheapest_within_hops(graph, source, target, weight, max_hops, banned, banned_next):
    """Cheapest path from source to target with at most ``max_hops`` hops.

    Bellman-Ford by layers: layer h holds the nodes whose cheapest cost fell
    when h hops were allowed, so only those need relaxing at layer h + 1.
    ``banned`` nodes are never entered, and the first hop may not go to a node
    in ``banned_next``. Returns a node tuple or None.
    """
    if max_hops < 1:
        return None
    best = {source: 0.0}
    frontier = {source: 0.0}
    layers: list[dict] = []
    found = 0
    for hops in range(1, max_hops + 1):
        improved: dict = {}
        parent: dict = {}
        for u, cost_u in frontier.items():
            for v, data in graph[u].items():
                if v == source or v in banned or (hops == 1 and v in banned_next):
                    continue
                cost = cost_u + data[weight]
                if cost >= best.get(v, math.inf) * (1.0 - 1e-12):
                    continue
                held = improved.get(v)
                if (held is None or cost < held * (1.0 - 1e-12)
                        or (cost <= held * (1.0 + 1e-12) and str(u) < str(parent[v]))):
                    improved[v] = cost
                    parent[v] = u
        if not improved:
            break
        best.update(improved)
        layers.append(parent)
        if target in improved:
            found = hops
        frontier = {v: c for v, c in improved.items() if v != target}
    if not found:
        return None
    nodes = [target]
    for hops in range(found, 0, -1):
        nodes.append(layers[hops - 1][nodes[-1]])
    nodes.reverse()
    if nodes[0] != source or len(set(nodes)) != len(nodes):
        return None
    return tuple(nodes)


# --------------------------------------------------------------------------
# Exhaustive strategy
# --------------------------------------------------------------------------


def _exhaustive_node_lists(logical, source, target, max_hops, max_paths):
    out = []
    for nodes in nx.all_simple_paths(logical, source, target, cutoff=max_hops):
        out.append(tuple(nodes))
        if len(out) > max_paths:
            raise ValueError(
                f"more than {max_paths} simple paths between {source} and {target}; "
                "exhaustive enumeration is meant for small validation graphs"
            )
    return out


# --------------------------------------------------------------------------
# Legacy strategy, kept to reproduce committed results
# --------------------------------------------------------------------------


def _legacy_node_lists(logical, source, target, max_hops):
    out = []
    for dp in (_bottleneck_dp, _shortest_total_dp):
        value, parent = dp(logical, source, max_hops)
        for hops in range(1, max_hops + 1):
            if (hops, target) not in value:
                continue
            nodes = _reconstruct(parent, target, hops)
            if nodes is None or nodes[0] != source:
                continue
            out.append(tuple(nodes))
    return out


def _reconstruct(parent: dict, target: str, hops: int) -> list[str] | None:
    """Walk the predecessor table back to the source. None if not simple."""
    nodes = [target]
    node = target
    for h in range(hops, 0, -1):
        previous = parent.get((h, node))
        if previous is None:
            return None
        nodes.append(previous)
        node = previous
    nodes.reverse()
    if len(set(nodes)) != len(nodes):
        return None
    return nodes


def _bottleneck_dp(
    logical: nx.Graph, source: str, max_hops: int
) -> tuple[dict, dict]:
    """Legacy: minimise the longest link over walks of exactly h hops."""
    value: dict[tuple[int, str], float] = {(0, source): 0.0}
    parent: dict[tuple[int, str], str] = {}

    for h in range(1, max_hops + 1):
        for u in logical.nodes:
            base = value.get((h - 1, u))
            if base is None:
                continue
            for v in logical.neighbors(u):
                candidate = max(base, logical[u][v]["length_km"])
                key = (h, v)
                if candidate < value.get(key, math.inf) - 1e-12:
                    value[key] = candidate
                    parent[key] = u
    return value, parent


def _shortest_total_dp(
    logical: nx.Graph, source: str, max_hops: int
) -> tuple[dict, dict]:
    """Legacy: minimise total length over walks of exactly h hops."""
    value: dict[tuple[int, str], float] = {(0, source): 0.0}
    parent: dict[tuple[int, str], str] = {}

    for h in range(1, max_hops + 1):
        for u in logical.nodes:
            base = value.get((h - 1, u))
            if base is None:
                continue
            for v in logical.neighbors(u):
                candidate = base + logical[u][v]["length_km"]
                key = (h, v)
                if candidate < value.get(key, math.inf) - 1e-12:
                    value[key] = candidate
                    parent[key] = u
    return value, parent


def _drop_dominated(candidates: list[CandidatePath]) -> list[CandidatePath]:
    """Legacy: remove paths beaten on both hop count and worst link.

    This is safe for one user pair on its own under Eq. (2), and NOT safe for
    the network problem, where shared repeater memory and the repeater budget
    can make a dominated path the one the optimum needs.
    """
    kept: list[CandidatePath] = []
    for candidate in candidates:
        dominated = any(
            other.hops <= candidate.hops
            and other.max_link_km <= candidate.max_link_km + 1e-9
            and (
                other.hops < candidate.hops
                or other.max_link_km < candidate.max_link_km - 1e-9
            )
            for other in candidates
            if other is not candidate
        )
        if not dominated:
            kept.append(candidate)
    return kept


def path_statistics(paths: dict[tuple[str, str], list[CandidatePath]]) -> str:
    """One-line summary for the run log."""
    counts = [len(v) for v in paths.values()]
    if not counts:
        return "no candidate paths"
    hop_range = [p.hops for v in paths.values() for p in v]
    return (
        f"{sum(counts)} candidate paths over {len(counts)} pairs "
        f"({min(counts)} to {max(counts)} per pair), "
        f"hop counts {min(hop_range)} to {max(hop_range)}"
    )
