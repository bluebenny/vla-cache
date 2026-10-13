"""V' in-module statistics vs. the reduction of V's full attention maps, on the forked LLaMA."""

import copy

import pytest
import torch
from helpers import SEQ_LEN, as_vla, build_tiny_llama, run_lm
from transformers import DynamicCache

from experiments.robot.vla_cache_sdpa import (
    VPrimeLlamaSdpaAttention,
    attn_stats_from_attentions,
    install_vprime,
)
from experiments.robot.vla_cache_utils import get_layer_mask_schedule

REUSABLE = torch.arange(1, 257, 2)  # odd vision positions


def _stats_equal(a, b):
    return all(torch.equal(x, y) for x, y in zip(a.entropies, b.entropies)) and torch.equal(
        a.text_vision_score, b.text_vision_score
    )


def _prev_and_cur_embeds(seed, dtype):
    g = torch.Generator().manual_seed(seed)
    prev = torch.randn(1, SEQ_LEN, 64, generator=g).to(dtype)
    cur = prev.clone()
    moved = torch.arange(2, 257, 4)  # vision rows that changed between frames
    cur[:, moved] = torch.randn(1, len(moved), 64, generator=g).to(dtype)
    return prev, cur


def _run_vprime(lm, collector, embeds, cache):
    collector.reset()
    collector.enabled = True
    out = run_lm(lm, embeds, cache, output_attentions=False)
    collector.enabled = False
    return out, collector.pop()


@pytest.fixture(params=[torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
def setup(request):
    dtype = request.param
    lm = build_tiny_llama(dtype=dtype)
    collector = install_vprime(as_vla(lm), attn_compute="eager_debug")
    prev, cur = _prev_and_cur_embeds(0, dtype)
    return lm, collector, prev, cur


def test_install_swaps_classes_and_checks_fork(setup):
    lm, collector, _, _ = setup
    assert all(type(layer.self_attn) is VPrimeLlamaSdpaAttention for layer in lm.model.layers)
    assert "_update_causal_mask" in vars(lm.model)


def test_prefill_without_reuse_matches_eager_exactly(setup):
    lm, collector, prev, _ = setup
    v_out = run_lm(lm, prev, DynamicCache(), output_attentions=True)  # V / B1: eager fallback
    p_out, stats = _run_vprime(lm, collector, prev, DynamicCache())

    assert len(v_out.attentions) == len(lm.model.layers) + 1  # + final cache_position
    assert _stats_equal(stats, attn_stats_from_attentions(v_out.attentions))
    assert torch.equal(p_out.logits, v_out.logits)


def test_reuse_prefill_matches_eager_exactly(setup):
    lm, collector, prev, cur = setup
    # The fork re-wraps the passed cache; like `generate`, continue from the returned one
    v_prev = run_lm(lm, prev, DynamicCache(), output_attentions=True)
    prev_cache = v_prev.past_key_values

    lm.config.reusable_patches = REUSABLE
    lm.config.proportion_attn_var = get_layer_mask_schedule(v_prev.attentions)
    try:
        v_cache, p_cache = copy.deepcopy(prev_cache), copy.deepcopy(prev_cache)
        v_out = run_lm(lm, cur, v_cache, output_attentions=True)
        p_out, stats = _run_vprime(lm, collector, cur, p_cache)
    finally:
        lm.config.reusable_patches = None
        lm.config.proportion_attn_var = None

    # Some rows were really dropped, and the statistics saw the same pruned rows as V
    assert len(v_out.attentions[-1]) < SEQ_LEN
    assert _stats_equal(stats, attn_stats_from_attentions(v_out.attentions))
    assert torch.equal(p_out.logits, v_out.logits)
    v_cache, p_cache = v_out.past_key_values, p_out.past_key_values
    assert p_cache.get_seq_length() == v_cache.get_seq_length() == SEQ_LEN
    for layer in range(len(lm.model.layers)):
        assert torch.equal(p_cache.key_cache[layer], v_cache.key_cache[layer])
        assert torch.equal(p_cache.value_cache[layer], v_cache.value_cache[layer])


def test_sdpa_statistics_are_close_to_eager():
    lm = build_tiny_llama(dtype=torch.float32)
    collector = install_vprime(as_vla(lm), attn_compute="sdpa")
    prev, _ = _prev_and_cur_embeds(1, torch.float32)
    v_out = run_lm(lm, prev, DynamicCache(), output_attentions=True)
    _, stats = _run_vprime(lm, collector, prev, DynamicCache())
    reference = attn_stats_from_attentions(v_out.attentions)
    for a, b in zip(stats.entropies, reference.entropies):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(stats.text_vision_score, reference.text_vision_score, rtol=1e-4, atol=1e-6)


def test_collector_off_keeps_original_sdpa_outputs():
    """Without collection, the copied forward is the fork's LlamaSdpaAttention.forward."""
    reference = build_tiny_llama(dtype=torch.float32)
    lm = build_tiny_llama(dtype=torch.float32)
    install_vprime(as_vla(lm))
    prev, _ = _prev_and_cur_embeds(2, torch.float32)
    assert torch.equal(
        run_lm(lm, prev, DynamicCache(), output_attentions=False).logits,
        run_lm(reference, prev, DynamicCache(), output_attentions=False).logits,
    )
