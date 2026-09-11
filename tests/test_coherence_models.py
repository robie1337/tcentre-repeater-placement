"""Coherence models inside the placement model, not bolted on in scripts."""

from __future__ import annotations

import math

import pytest

from qrp import physics, waiting
from qrp.hardware import TCENTRE_MIDRANGE
from qrp.model import COHERENCE_MODELS, NetworkConfig, build_candidates, solve_placement
from qrp.paths import enumerate_paths
from qrp.topology import build_dumbbell

#: Short enough that waiting matters on a 400 km dumbbell, long enough that
#: the paper's own propagation gates still pass some paths.
HARDWARE = TCENTRE_MIDRANGE.with_(t_repeater_memory_s=6e-3, t_endnode_memory_s=6e-3)
MEASURED_T2 = TCENTRE_MIDRANGE.with_(t_repeater_memory_s=0.112, t_endnode_memory_s=0.112)


@pytest.fixture(scope="module")
def paths():
    topo = build_dumbbell(400.0, n_candidates=10)
    return topo, enumerate_paths(topo, topo.demand_pairs(), max_link_km=150.0, max_hops=11)


def by_key(candidates):
    return {(c.pair, c.path.nodes, c.width): c for c in candidates}


def config(model: str) -> NetworkConfig:
    return NetworkConfig(require_all_pairs=False, coherence_model=model)


def test_paper_is_the_default(paths):
    _, p = paths
    default = by_key(build_candidates(p, HARDWARE, NetworkConfig(require_all_pairs=False)))
    explicit = by_key(build_candidates(p, HARDWARE, config("paper")))
    assert default.keys() == explicit.keys()
    assert all(default[k].utility == explicit[k].utility for k in default)


def test_waiting_gate_drops_exactly_the_combinations_that_wait_too_long(paths):
    _, p = paths
    paper = by_key(build_candidates(p, HARDWARE, config("paper")))
    gated = by_key(build_candidates(p, HARDWARE, config("waiting_gate")))
    assert gated.keys() <= paper.keys()
    removed = 0
    for key, cand in paper.items():
        probs = [physics.link_success(length, HARDWARE.alpha_db_per_km)
                 for length in cand.path.link_lengths_km]
        rates = waiting.link_ready_rates(cand.path.link_lengths_km, probs, cand.width,
                                         HARDWARE.generation_rate_hz)
        storage = waiting.expected_storage_time(rates, cand.tau_e2e_s)
        assert (key in gated) == (storage <= HARDWARE.t_repeater_memory_s)
        removed += key not in gated
    assert 0 < removed < len(paper)


def test_decay_only_lowers_fidelity_and_the_pessimistic_bound_lowers_it_more(paths):
    # At 6 ms every decayed combination falls to F <= 1/2, so use the measured
    # coherence, where some survive and the comparison has something to check.
    _, p = paths
    paper = by_key(build_candidates(p, MEASURED_T2, config("paper")))
    opt = by_key(build_candidates(p, MEASURED_T2, config("decay_optimistic")))
    pess = by_key(build_candidates(p, MEASURED_T2, config("decay_pessimistic")))
    assert pess
    assert pess.keys() <= opt.keys() <= paper.keys()
    for key, cand in pess.items():
        assert cand.fidelity <= opt[key].fidelity + 1e-12
        assert opt[key].fidelity <= paper[key].fidelity + 1e-12
        assert cand.rate == paper[key].rate
        assert 0.0 < cand.decay_factor <= opt[key].decay_factor <= 1.0


def test_decay_weakens_as_coherence_grows(paths):
    _, p = paths
    shorter = MEASURED_T2.with_(t_repeater_memory_s=0.04, t_endnode_memory_s=0.04)
    short = by_key(build_candidates(p, shorter, config("decay_optimistic")))
    long = by_key(build_candidates(p, MEASURED_T2, config("decay_optimistic")))
    common = short.keys() & long.keys()
    assert common
    assert all(long[k].fidelity >= short[k].fidelity for k in common)
    assert len(long) >= len(short)


def test_infinite_coherence_makes_every_model_the_paper_model(paths):
    _, p = paths
    hw = HARDWARE.with_(t_repeater_memory_s=math.inf, t_endnode_memory_s=math.inf)
    paper = by_key(build_candidates(p, hw, config("paper")))
    for model in COHERENCE_MODELS:
        other = by_key(build_candidates(p, hw, config(model)))
        assert other.keys() == paper.keys()
        assert all(other[k].utility == pytest.approx(paper[k].utility) for k in paper)


@pytest.mark.parametrize("model", COHERENCE_MODELS)
def test_every_model_solves(paths, model):
    topo, p = paths
    result = solve_placement(topo, p, MEASURED_T2, config(model), mip_gap=0.0)
    assert result.status == "optimal"


def test_unknown_coherence_model_raises():
    with pytest.raises(ValueError, match="coherence_model"):
        NetworkConfig(coherence_model="hopeful")
