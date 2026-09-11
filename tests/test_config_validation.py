"""Inputs that would silently produce a wrong answer must raise instead."""

from __future__ import annotations

import pytest

from qrp.hardware import (
    POURYOUSEF_BASELINE,
    TCENTRE_MEASURED,
    TCENTRE_MIDRANGE,
    TCENTRE_PROJECTED,
    Hardware,
)
from qrp.model import NetworkConfig
from qrp.paths import enumerate_paths
from qrp.sweep import SweepContext, evaluate
from qrp.topology import build_dumbbell, build_surfnet


class TestNetworkConfig:
    def test_one_path_per_pair_is_the_only_supported_value(self):
        with pytest.raises(ValueError, match="paths_per_pair"):
            NetworkConfig(paths_per_pair=2)

    def test_unknown_rate_model(self):
        with pytest.raises(ValueError, match="rate_model"):
            NetworkConfig(rate_model="optimistic")

    def test_defaults_are_valid(self):
        NetworkConfig()


class TestHardware:
    @pytest.mark.parametrize("preset", [POURYOUSEF_BASELINE, TCENTRE_MEASURED, TCENTRE_MIDRANGE,
                                        TCENTRE_PROJECTED])
    def test_presets_are_valid(self, preset):
        preset.with_()

    @pytest.mark.parametrize("field, value", [
        ("link_fidelity", 1.1),
        ("link_fidelity", -0.1),
        ("swap_success", 1.5),
        ("gate_fidelity", 1.2),
        ("measurement_fidelity", -0.01),
        ("eta_emission", 2.0),
        ("eta_detection", float("nan")),
        ("swap_werner", 1.01),
        ("alpha_db_per_km", -0.35),
        ("alpha_db_per_km", float("inf")),
        ("generation_rate_hz", 0.0),
        ("generation_rate_hz", -5.0),
        ("t_repeater_memory_s", 0.0),
        ("t_endnode_memory_s", -1.0),
        ("t_endnode_memory_s", float("nan")),
    ])
    def test_invalid_values_are_rejected(self, field, value):
        with pytest.raises(ValueError, match=field):
            Hardware(**{field: value})

    def test_infinite_coherence_is_allowed(self):
        Hardware(t_repeater_memory_s=float("inf"), t_endnode_memory_s=float("inf"))


def _write_gml(path, labels):
    nodes = "\n".join(f'  node [ id {i} label "{label}" ]' for i, label in enumerate(labels))
    edges = "\n".join(f"  edge [ source {i} target {i + 1} length 50.0 ]"
                      for i in range(len(labels) - 1))
    path.write_text(f"graph [\n{nodes}\n{edges}\n]\n", encoding="utf-8")


class TestSurfnetLoader:
    LABELS = ["Delft", "Delft North", "Enschede", "Groningen", "Maastricht"]

    def test_exact_label_wins_over_a_longer_one(self, tmp_path):
        gml = tmp_path / "net.gml"
        _write_gml(gml, self.LABELS)
        topo, resolved = build_surfnet(gml)
        assert topo.graph.nodes[resolved["Delft"]]["label"] == "Delft"
        assert len(topo.cities) == 4

    def test_missing_name_raises(self, tmp_path):
        gml = tmp_path / "net.gml"
        _write_gml(gml, self.LABELS)
        with pytest.raises(ValueError, match="matches no label"):
            build_surfnet(gml, end_nodes=("Delft", "Utrecht"))

    def test_ambiguous_name_raises(self, tmp_path):
        gml = tmp_path / "net.gml"
        _write_gml(gml, self.LABELS)
        with pytest.raises(ValueError, match="ambiguous"):
            build_surfnet(gml, end_nodes=("Del",))


def test_sweep_rows_record_the_model_choices():
    topo = build_dumbbell(100.0, n_candidates=3)
    paths = enumerate_paths(topo, topo.demand_pairs(), max_link_km=101.0, max_hops=4)
    context = SweepContext(topology=topo, paths=paths,
                           config=NetworkConfig(require_all_pairs=False),
                           base_hardware=POURYOUSEF_BASELINE, max_link_km=101.0, max_hops=4)
    record = evaluate(context, {})
    for key in ("backend", "mip_gap", "rate_model", "coherence_model", "path_strategy",
                "spacing_km", "max_link_km", "max_hops", "width_grid"):
        assert key in record
    assert record["max_hops"] == 4
    assert record["width_grid"] == ";".join(str(w) for w in NetworkConfig().widths())
