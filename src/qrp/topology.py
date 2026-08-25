"""Network topologies: the CA9 graph and the paper's own test cases.

The nine city graph is the project's own contribution as a dataset. Distances
are fibre-route estimates from vault note 37, which builds them from
great-circle distances scaled by the 1.1 to 1.4 factor that real routes carry,
cross-checked against the Bell wholesale network map where rights of way are
shared.

The operator does not publish PoP locations, so candidate repeater sites are placed
at uniform spacing along each corridor rather than at named facilities. That
is stated as a limitation in the report rather than hidden.

The graph is deliberately in two components. The eastern and western halves
of the nine city set are separated by roughly 2000 km of fibre through
Thunder Bay with no intermediate CA9 city in our node set. Set
``bridge=True`` to add that edge.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations

import networkx as nx

# --------------------------------------------------------------------------
# Cities
# --------------------------------------------------------------------------

#: (latitude, longitude) for the nine end nodes. Used only for drawing.
CITY_COORDS: dict[str, tuple[float, float]] = {
    "Vancouver": (49.2827, -123.1207),
    "Kamloops": (50.6745, -120.3273),
    "Kelowna": (49.8880, -119.4960),
    "Calgary": (51.0447, -114.0719),
    "Edmonton": (53.5461, -113.4938),
    "Saskatoon": (52.1332, -106.6700),
    "Toronto": (43.6532, -79.3832),
    "Ottawa": (45.4215, -75.6972),
    "Montreal": (45.5019, -73.5674),
}

WEST = ("Vancouver", "Kamloops", "Kelowna", "Calgary", "Edmonton", "Saskatoon")
EAST = ("Toronto", "Ottawa", "Montreal")

#: Fibre-route length estimates in km (vault note 37, cross-checked against
#: the corridor list in note 28).
CITY_EDGES: dict[tuple[str, str], float] = {
    ("Vancouver", "Kamloops"): 360.0,
    ("Vancouver", "Kelowna"): 390.0,
    ("Kamloops", "Kelowna"): 190.0,
    ("Kamloops", "Calgary"): 600.0,
    ("Kelowna", "Calgary"): 610.0,
    ("Calgary", "Edmonton"): 300.0,
    ("Calgary", "Saskatoon"): 640.0,
    ("Edmonton", "Saskatoon"): 530.0,
    ("Toronto", "Ottawa"): 450.0,
    ("Ottawa", "Montreal"): 200.0,
    ("Toronto", "Montreal"): 540.0,
}

#: The east-west link, off by default. Vault note 37 lists it as a bridge
#: placeholder rather than a verified carrier route.
BRIDGE_EDGE: tuple[tuple[str, str], float] = (("Saskatoon", "Toronto"), 2000.0)

#: Demand weights from vault note 37. Heavier pairs are the corridors the carrier
#: actually markets. Everything unlisted is weight 1.
DEMAND_WEIGHTS: dict[frozenset[str], int] = {
    frozenset({"Vancouver", "Calgary"}): 3,
    frozenset({"Toronto", "Montreal"}): 3,
    frozenset({"Toronto", "Ottawa"}): 3,
    frozenset({"Calgary", "Edmonton"}): 2,
    frozenset({"Vancouver", "Kelowna"}): 2,
    frozenset({"Ottawa", "Montreal"}): 2,
}


# --------------------------------------------------------------------------
# Graph construction
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Topology:
    """A physical graph with end nodes and candidate repeater sites.

    ``graph`` carries every node. Nodes have a ``kind`` attribute of either
    ``"city"`` or ``"site"``; edges have a ``length_km`` attribute and are
    only ever between consecutive nodes along one corridor.
    """

    graph: nx.Graph
    cities: tuple[str, ...]
    sites: tuple[str, ...]
    name: str
    spacing_km: float

    @property
    def n_candidates(self) -> int:
        return len(self.sites)

    def demand_pairs(self, within_component: bool = True) -> list[tuple[str, str]]:
        """All city pairs that could be served, optionally same-component only."""
        pairs = []
        components = list(nx.connected_components(self.graph))

        def component_of(node: str) -> int:
            for i, comp in enumerate(components):
                if node in comp:
                    return i
            raise KeyError(node)

        for a, b in combinations(sorted(self.cities), 2):
            if within_component and component_of(a) != component_of(b):
                continue
            pairs.append((a, b))
        return pairs

    def demand_weight(self, a: str, b: str) -> int:
        return DEMAND_WEIGHTS.get(frozenset({a, b}), 1)


def _insert_sites(
    graph: nx.Graph,
    u: str,
    v: str,
    length_km: float,
    spacing_km: float,
    label: str,
) -> list[str]:
    """Split one corridor into segments no longer than ``spacing_km``.

    A corridor of length L gets ceil(L / spacing) - 1 interior sites, evenly
    spaced, so every resulting segment is at most ``spacing_km`` long.
    """
    n_segments = max(1, math.ceil(length_km / spacing_km))
    n_sites = n_segments - 1
    if n_sites == 0:
        graph.add_edge(u, v, length_km=length_km, corridor=(u, v))
        return []

    seg_length = length_km / n_segments
    site_names = [f"{label}:{i + 1}" for i in range(n_sites)]
    for name in site_names:
        graph.add_node(name, kind="site", corridor=(u, v))

    chain = [u, *site_names, v]
    for a, b in zip(chain[:-1], chain[1:]):
        graph.add_edge(a, b, length_km=seg_length, corridor=(u, v))
    return site_names


def build_ca9(
    spacing_km: float = 80.0,
    bridge: bool = False,
) -> Topology:
    """Build the nine city CA9 graph.

    Parameters
    ----------
    spacing_km
        Maximum distance between adjacent candidate repeater sites. The
        default of 80 km is the PoP spacing assumed throughout the vault.
    bridge
        Include the ~2000 km Saskatoon to Toronto edge, connecting the two
        halves of the network. Off by default because that route is a
        placeholder rather than a verified carrier corridor.
    """
    graph = nx.Graph()
    for city in CITY_COORDS:
        graph.add_node(city, kind="city", coords=CITY_COORDS[city])

    edges = dict(CITY_EDGES)
    if bridge:
        (bu, bv), blen = BRIDGE_EDGE
        edges[(bu, bv)] = blen

    sites: list[str] = []
    for (u, v), length in edges.items():
        label = f"{u[:3]}-{v[:3]}".upper()
        sites.extend(_insert_sites(graph, u, v, length, spacing_km, label))

    return Topology(
        graph=graph,
        cities=tuple(CITY_COORDS),
        sites=tuple(sites),
        # Display name, used in figure titles. Kept neutral so that figures
        # can go into slides without naming the carrier.
        name=f"Nine-city carrier network{' (bridged)' if bridge else ''}",
        spacing_km=spacing_km,
    )


def build_dumbbell(
    backbone_km: float,
    n_candidates: int = 10,
    name: str | None = None,
) -> Topology:
    """The paper's dumbbell test case.

    Two end nodes joined by one backbone with equally spaced candidate
    repeater locations. Section IV of the paper uses this to show that below
    roughly 40 km of backbone no repeater is placed at all. Week 1 validation
    reproduces that.
    """
    graph = nx.Graph()
    graph.add_node("A", kind="city", coords=(0.0, 0.0))
    graph.add_node("B", kind="city", coords=(0.0, 1.0))

    n_segments = n_candidates + 1
    seg_length = backbone_km / n_segments
    site_names = [f"R{i + 1}" for i in range(n_candidates)]
    for site in site_names:
        graph.add_node(site, kind="site", corridor=("A", "B"))

    chain = ["A", *site_names, "B"]
    for a, b in zip(chain[:-1], chain[1:]):
        graph.add_edge(a, b, length_km=seg_length, corridor=("A", "B"))

    return Topology(
        graph=graph,
        cities=("A", "B"),
        sites=tuple(site_names),
        name=name or f"dumbbell-{backbone_km:g}km",
        spacing_km=seg_length,
    )


def build_line(
    total_km: float,
    n_candidates: int,
    name: str = "line",
) -> Topology:
    """A two-endpoint chain. Convenience wrapper used by the tests."""
    return build_dumbbell(total_km, n_candidates, name=name)


# --------------------------------------------------------------------------
# The paper's own SURFnet instance, from the authors' published data
# --------------------------------------------------------------------------

#: The four end nodes used in the SURFnet experiments. Rabbie et al., whose
#: instance Pouryousef reuses, name Delft, Enschede, Groningen and Maastricht.
SURFNET_END_NODES = ("Delft", "Enschede", "Groningen", "Maastricht")


def build_surfnet(
    gml_path,
    end_nodes: tuple[str, ...] = SURFNET_END_NODES,
) -> tuple[Topology, dict[str, str]]:
    """Load the authors' own SURFnet topology from their published GML.

    The file is ``data/SurfnetCore.gml`` in github.com/pooryousefshahrooz/
    q_net_planning. It carries a measured fibre length on every edge, so this
    is the real network rather than a straight-line stand-in.

    Every node is a candidate repeater location, which is what the paper says
    it does for SURFnet. Returns the topology and a map from the requested
    end-node names to the graph's own node identifiers.
    """
    raw = nx.read_gml(gml_path, label="id")

    graph = nx.Graph()
    labels: dict[int, str] = {}
    for node, data in raw.nodes(data=True):
        label = str(data.get("label", node))
        labels[node] = label
        graph.add_node(node, kind="site", label=label)

    for u, v, data in raw.edges(data=True):
        length = float(data.get("length", 0.0))
        if length <= 0:
            continue
        graph.add_edge(u, v, length_km=length, corridor=(u, v))

    # Match the requested end nodes against the file's own labels.
    resolved: dict[str, str] = {}
    for wanted in end_nodes:
        for node, label in labels.items():
            if wanted.lower() in label.lower():
                resolved[wanted] = node
                graph.nodes[node]["kind"] = "city"
                break

    cities = tuple(resolved.values())
    sites = tuple(n for n in graph.nodes if n not in set(cities))

    return (
        Topology(
            graph=graph,
            cities=cities,
            sites=sites,
            name="SURFnet core (authors' own data)",
            spacing_km=0.0,
        ),
        resolved,
    )


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def describe(topology: Topology) -> str:
    """One-paragraph summary, printed at the top of every script run."""
    graph = topology.graph
    components = nx.number_connected_components(graph)
    lengths = [d["length_km"] for _, _, d in graph.edges(data=True)]
    total = sum(lengths)
    return (
        f"{topology.name}: {len(topology.cities)} end nodes, "
        f"{topology.n_candidates} candidate sites, "
        f"{graph.number_of_edges()} segments, "
        f"{components} component(s), "
        f"{total:.0f} km of fibre, "
        f"segment length {min(lengths):.1f} to {max(lengths):.1f} km"
    )
