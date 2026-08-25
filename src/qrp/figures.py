"""Figures for the report.

Everything is written as vector PDF plus a PNG for slides. Captions belong in
the report, not here, but each function returns the path it wrote so the
scripts can log it.

House style: one accent colour, no chart junk, no colour used decoratively.
Sequential data uses viridis, which stays readable in greyscale and for the
common colour vision deficiencies.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .hardware import LABELS, LOG_SCALED
from .topology import CITY_COORDS, Topology

FIG_DIR = Path(__file__).resolve().parents[2] / "figures"

INK = "#111111"
MUTED = "#6B7280"
ACCENT = "#1C7293"
WARN = "#C71F37"

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
    }
)


def _save(fig, stem: str, out_dir: Path | None = None) -> Path:
    directory = out_dir or FIG_DIR
    directory.mkdir(parents=True, exist_ok=True)
    pdf = directory / f"{stem}.pdf"
    fig.savefig(pdf)
    fig.savefig(directory / f"{stem}.png")
    plt.close(fig)
    return pdf


# --------------------------------------------------------------------------
# Topology
# --------------------------------------------------------------------------


def node_positions(topology: Topology) -> dict[str, tuple[float, float]]:
    """Longitude/latitude for every node, interpolating sites along corridors."""
    positions: dict[str, tuple[float, float]] = {
        city: (lon, lat) for city, (lat, lon) in CITY_COORDS.items()
    }

    corridors: dict[tuple[str, str], list[str]] = {}
    for node, data in topology.graph.nodes(data=True):
        if data.get("kind") != "site":
            continue
        corridors.setdefault(data["corridor"], []).append(node)

    for (u, v), sites in corridors.items():
        sites.sort(key=lambda n: int(n.rsplit(":", 1)[1]))
        x0, y0 = positions[u]
        x1, y1 = positions[v]
        for i, site in enumerate(sites, start=1):
            t = i / (len(sites) + 1)
            positions[site] = (x0 + t * (x1 - x0), y0 + t * (y1 - y0))
    return positions


#: Hand-placed label offsets, because the western cities sit close together
#: and automatic placement overlaps them.
_LABEL_OFFSETS: dict[str, tuple[float, float, str, str]] = {
    "Vancouver": (-7, -3, "right", "center"),
    "Kamloops": (0, 9, "center", "bottom"),
    "Kelowna": (2, -11, "left", "top"),
    "Calgary": (7, -6, "left", "top"),
    "Edmonton": (0, 9, "center", "bottom"),
    "Saskatoon": (8, 2, "left", "center"),
    "Toronto": (-8, -2, "right", "center"),
    "Ottawa": (-7, 6, "right", "bottom"),
    "Montreal": (8, 2, "left", "center"),
}


def plot_topology(
    topology: Topology,
    chosen_repeaters: list[str] | None = None,
    stem: str = "fig_topology",
    out_dir: Path | None = None,
    title: str | None = None,
) -> Path:
    """The nine city graph, optionally with the selected repeaters marked.

    Disconnected components are drawn in their own panels. The western and
    eastern halves of the network are about 2000 km apart with no CA9 city
    between them in our node set, and drawing them on one axis wastes most of
    the figure on empty prairie.
    """
    import networkx as nx

    positions = node_positions(topology)
    chosen = set(chosen_repeaters or [])
    components = sorted(
        nx.connected_components(topology.graph),
        key=lambda c: -min(positions[n][0] for n in c),
    )
    components = list(reversed(components))

    widths = []
    for comp in components:
        xs = [positions[n][0] for n in comp]
        widths.append(max(max(xs) - min(xs), 1.0))

    fig, axes = plt.subplots(
        1, len(components),
        figsize=(7.6, 3.9),
        gridspec_kw={"width_ratios": widths, "wspace": 0.22},
    )
    if len(components) == 1:
        axes = [axes]

    total_sites = total_used = 0

    for ax, comp in zip(axes, components):
        for u, v in topology.graph.edges:
            if u not in comp:
                continue
            ax.plot(
                [positions[u][0], positions[v][0]],
                [positions[u][1], positions[v][1]],
                color="#D4D0C8", lw=1.0, zorder=1,
            )

        site_x, site_y, used_x, used_y = [], [], [], []
        for site in topology.sites:
            if site not in comp:
                continue
            x, y = positions[site]
            (used_x if site in chosen else site_x).append(x)
            (used_y if site in chosen else site_y).append(y)
        total_sites += len(site_x)
        total_used += len(used_x)

        ax.scatter(site_x, site_y, s=9, color=MUTED, alpha=0.45, zorder=2)
        if used_x:
            ax.scatter(used_x, used_y, s=40, color=ACCENT, zorder=4,
                       edgecolor="white", linewidth=0.6)

        for city in topology.cities:
            if city not in comp:
                continue
            x, y = positions[city]
            colour = WARN if city in chosen else INK
            ax.scatter([x], [y], s=46, color=colour, marker="s", zorder=5,
                       edgecolor="white", linewidth=0.7)
            dx, dy, ha, va = _LABEL_OFFSETS.get(city, (0, 8, "center", "bottom"))
            ax.annotate(city, (x, y), textcoords="offset points", xytext=(dx, dy),
                        ha=ha, va=va, fontsize=8, color=INK)

        ax.set_xlabel("Longitude")
        ax.margins(0.18)
        ax.tick_params(labelsize=7)

    axes[0].set_ylabel("Latitude")

    handles = [
        plt.Line2D([], [], marker="s", ls="", color=INK, ms=6, label="end node"),
        plt.Line2D([], [], marker="o", ls="", color=MUTED, alpha=0.5, ms=4,
                   label=f"candidate site ({total_sites})"),
    ]
    if total_used:
        handles.append(
            plt.Line2D([], [], marker="o", ls="", color=ACCENT, ms=7,
                       label=f"repeater selected ({total_used})")
        )
    if any(c in chosen for c in topology.cities):
        handles.append(
            plt.Line2D([], [], marker="s", ls="", color=WARN, ms=6,
                       label="end node hosting a repeater")
        )

    fig.legend(handles=handles, frameon=False, fontsize=8,
               loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(
        title or f"{topology.name}: {topology.n_candidates} candidate sites "
                 f"at {topology.spacing_km:.0f} km spacing",
        x=0.02, ha="left", fontsize=10,
    )
    return _save(fig, stem, out_dir)


# --------------------------------------------------------------------------
# Phase diagram
# --------------------------------------------------------------------------


def plot_phase_diagram(
    frame: pd.DataFrame,
    x_name: str,
    y_name: str,
    value: str = "utility",
    stem: str = "fig_phase_diagram",
    out_dir: Path | None = None,
    contour: str | None = "served_pairs",
    title: str | None = None,
) -> Path:
    """Heat map of one output over a two-parameter grid.

    The optional contour overlays the boundary where the network stops being
    able to serve every user pair, which is the sharpest feature in the data.
    """
    pivot = frame.pivot_table(index=y_name, columns=x_name, values=value, aggfunc="mean")
    xs = pivot.columns.to_numpy(dtype=float)
    ys = pivot.index.to_numpy(dtype=float)
    grid = pivot.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    mesh = ax.pcolormesh(xs, ys, grid, shading="nearest", cmap="viridis")
    bar = fig.colorbar(mesh, ax=ax)
    bar.set_label(_value_label(value))
    bar.outline.set_visible(False)

    if contour and contour in frame.columns:
        cpivot = frame.pivot_table(index=y_name, columns=x_name, values=contour, aggfunc="mean")
        cgrid = cpivot.to_numpy(dtype=float)
        top = np.nanmax(cgrid)
        if np.nanmin(cgrid) < top:
            # Label the contour with the number it actually traces. This is
            # the best the grid reaches, which is not necessarily every pair:
            # a two-dimensional slice holds the other parameters at their
            # midpoints, so the full network may need a corner the slice
            # never visits.
            total = frame["total_pairs"].max() if "total_pairs" in frame.columns else None
            if total is not None and top >= total:
                label = f"all {int(total)} pairs served"
            else:
                label = f"best in this slice: {top:.0f}"
                if total is not None:
                    label += f" of {int(total)}"
            lines = ax.contour(xs, ys, cgrid, levels=[top - 0.5], colors=[WARN], linewidths=1.6)
            ax.clabel(lines, fmt={top - 0.5: label}, fontsize=7)

    if x_name in LOG_SCALED:
        ax.set_xscale("log")
    if y_name in LOG_SCALED:
        ax.set_yscale("log")
    ax.set_xlabel(LABELS.get(x_name, x_name))
    ax.set_ylabel(LABELS.get(y_name, y_name))
    ax.set_title(title or f"{_value_label(value)} over the T centre sweep box", loc="left")
    return _save(fig, stem, out_dir)


def _value_label(value: str) -> str:
    return {
        "utility": "Network utility  $\\sum \\log_2(R(F-\\frac{1}{2}))$",
        "served_pairs": "User pairs served",
        "n_repeaters": "Repeaters placed",
        "mean_hops": "Mean hop count",
        "mean_fidelity": "Mean end-to-end fidelity",
    }.get(value, value)


# --------------------------------------------------------------------------
# Sobol indices
# --------------------------------------------------------------------------


def plot_sobol(
    result,
    stem: str = "fig_sobol_indices",
    out_dir: Path | None = None,
    title: str | None = None,
) -> Path:
    """Horizontal bars for S_i and S_Ti with their confidence intervals."""
    frame = result.to_frame()
    y = np.arange(len(frame))
    height = 0.38

    fig, ax = plt.subplots(figsize=(6.6, 0.7 * len(frame) + 1.8))
    ax.barh(y + height / 2, frame["ST"], height, xerr=frame["ST_conf"],
            color=ACCENT, label="total effect $S_{Ti}$",
            error_kw={"ecolor": MUTED, "lw": 0.9})
    ax.barh(y - height / 2, frame["S1"], height, xerr=frame["S1_conf"],
            color="#E8A904", label="first order $S_i$",
            error_kw={"ecolor": MUTED, "lw": 0.9})

    ax.set_yticks(y)
    ax.set_yticklabels(frame["parameter"])
    ax.invert_yaxis()
    ax.set_xlabel("Share of variance in " + _value_label(result.output_name))
    ax.set_title(
        title or f"Which hardware parameter moves the result "
                 f"({result.n_samples} solves)",
        loc="left",
    )
    ax.legend(frameon=False, fontsize=8)
    ax.axvline(0.0, color=MUTED, lw=0.8)
    return _save(fig, stem, out_dir)


# --------------------------------------------------------------------------
# One-dimensional scans
# --------------------------------------------------------------------------


def plot_line_scan(
    frame: pd.DataFrame,
    x_name: str,
    values: tuple[str, ...] = ("utility", "served_pairs", "n_repeaters"),
    stem: str = "fig_line_scan",
    out_dir: Path | None = None,
    title: str | None = None,
    mark_x: float | None = None,
    mark_label: str = "",
) -> Path:
    """Stacked panels for a one-parameter scan."""
    frame = frame.sort_values(x_name)
    fig, axes = plt.subplots(len(values), 1, figsize=(6.4, 1.9 * len(values)), sharex=True)
    if len(values) == 1:
        axes = [axes]

    for ax, value in zip(axes, values):
        ax.plot(frame[x_name], frame[value], color=ACCENT, lw=1.6, marker="o", ms=2.6)
        ax.set_ylabel(_value_label(value), fontsize=8)
        if mark_x is not None:
            ax.axvline(mark_x, color=WARN, lw=1.1, ls="--")
        ax.grid(True, color="#EEEEEE", lw=0.7)

    if x_name in LOG_SCALED:
        axes[-1].set_xscale("log")
    axes[-1].set_xlabel(LABELS.get(x_name, x_name))
    if mark_x is not None and mark_label:
        axes[0].annotate(mark_label, (mark_x, axes[0].get_ylim()[1]),
                         textcoords="offset points", xytext=(4, -12),
                         color=WARN, fontsize=8)
    axes[0].set_title(title or f"Scan over {LABELS.get(x_name, x_name)}", loc="left")
    return _save(fig, stem, out_dir)


def plot_validation(
    backbone_frame: pd.DataFrame,
    stem: str = "fig_validation_threshold",
    out_dir: Path | None = None,
) -> Path:
    """Repeaters placed against backbone length, for the 40 km check."""
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.plot(backbone_frame["backbone_km"], backbone_frame["n_repeaters"],
            color=ACCENT, lw=1.6, marker="o", ms=3.4)
    ax.axhline(0.5, color=MUTED, lw=0.8, ls=":")
    ax.set_xlabel("Backbone length (km)")
    ax.set_ylabel("Repeaters placed")
    ax.set_title("Below a threshold length the optimiser places no repeater", loc="left")
    ax.grid(True, color="#EEEEEE", lw=0.7)

    positive = backbone_frame[backbone_frame["n_repeaters"] > 0]
    if not positive.empty:
        threshold = positive["backbone_km"].min()
        ax.axvline(threshold, color=WARN, lw=1.2, ls="--")
        ax.annotate(f"first repeater at {threshold:.0f} km",
                    (threshold, ax.get_ylim()[1] * 0.75),
                    textcoords="offset points", xytext=(6, 0),
                    color=WARN, fontsize=8)
    return _save(fig, stem, out_dir)
