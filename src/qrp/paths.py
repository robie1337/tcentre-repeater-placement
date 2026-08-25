"""Candidate path enumeration.

The placement model needs, for each user pair, a set of candidate paths to
choose between. A path here is a *logical* path: a sequence of nodes that
will host repeaters, joined by elementary links. Nodes lying between two
consecutive repeaters are bypassed and act as optical switches, which is the
paper's own reading of what an unselected candidate site does.

Pouryousef enumerates k shortest paths with k between 1000 and 4000 and notes
this is a computational rather than principled choice. We can do better,
because for this objective the optimal path per hop count is computable
exactly.

Why an exact DP works
---------------------
For a path with h links, Eq. (5) fixes the fidelity: it depends on h and on
the hardware, not on which links were chosen. Eq. (2) makes the rate
proportional to p_min, the success probability of the worst link. Utility is
increasing in rate, so among all paths with exactly h links the best one is
whichever maximises p_min. Since p is decreasing in length, that is the path
minimising its longest link: a bottleneck shortest path with a hop
constraint, solvable exactly by dynamic programming.

So enumerating one bottleneck-optimal path per hop count gives the exact
per-pair optimum for every h. The model still has to choose between hop
counts, and the memory and budget constraints couple pairs together, so we
also keep the minimum-total-length path per hop count as an alternative.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import networkx as nx

from .topology import Topology


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
    pair would be joined and the DP would consider links so lossy that no
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
# Hop-constrained dynamic programs
# --------------------------------------------------------------------------


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
    """Minimise the longest link on a path of exactly h hops.

    Returns (value, parent) where value[(h, v)] is the smallest achievable
    maximum link length on an h-hop path from ``source`` to ``v``.
    """
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
    """Minimise total path length subject to exactly h hops."""
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


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------


def enumerate_paths(
    topology: Topology,
    pairs: list[tuple[str, str]],
    max_link_km: float = 300.0,
    max_hops: int = 20,
    logical: nx.Graph | None = None,
) -> dict[tuple[str, str], list[CandidatePath]]:
    """Build the candidate path set for every demand pair.

    For each pair and each hop count up to ``max_hops`` this returns the
    bottleneck-optimal path (exactly utility-optimal for that hop count) and,
    when different, the minimum-total-length path.

    Paths are independent of hardware, so this runs once and is reused across
    every point of the sweep. That is what keeps several thousand solves
    cheap.
    """
    if logical is None:
        logical = build_logical_graph(topology, max_link_km)

    result: dict[tuple[str, str], list[CandidatePath]] = {}

    for source, target in pairs:
        seen: set[tuple[str, ...]] = set()
        candidates: list[CandidatePath] = []

        for dp in (_bottleneck_dp, _shortest_total_dp):
            value, parent = dp(logical, source, max_hops)
            for hops in range(1, max_hops + 1):
                if (hops, target) not in value:
                    continue
                nodes = _reconstruct(parent, target, hops)
                if nodes is None or nodes[0] != source:
                    continue
                key = tuple(nodes)
                if key in seen:
                    continue
                seen.add(key)
                lengths = tuple(
                    logical[a][b]["length_km"] for a, b in zip(nodes[:-1], nodes[1:])
                )
                candidates.append(
                    CandidatePath(
                        source=source,
                        target=target,
                        nodes=key,
                        link_lengths_km=lengths,
                        repeaters=tuple(nodes[1:-1]),
                    )
                )

        candidates.sort(key=lambda p: (p.hops, p.max_link_km))
        result[(source, target)] = _drop_dominated(candidates)

    return result


def _drop_dominated(candidates: list[CandidatePath]) -> list[CandidatePath]:
    """Remove paths that no hardware setting could ever prefer.

    Path A dominates path B when A has no more hops and no longer a worst
    link, with at least one strictly better. Fidelity falls with hop count
    and rate falls with both hop count and worst-link length, so A beats B on
    utility at every width and every hardware point. A also needs no more
    repeaters, since a path with h hops uses h-1 of them, so it cannot lose
    on the budget constraint either.

    Ties are kept: two paths with the same hop count and worst link may use
    different repeaters, and which one is cheaper depends on what the rest of
    the network is doing.
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
