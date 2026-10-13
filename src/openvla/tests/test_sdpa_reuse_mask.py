"""Reuse prefill on SDPA: the causal mask must be explicit, otherwise the pruned queries are masked wrongly."""

import copy

import torch
from helpers import SEQ_LEN, as_vla, build_tiny_llama, run_lm
from transformers import DynamicCache

from experiments.robot.vla_cache_sdpa import install_vprime

REUSABLE = torch.arange(1, 257, 2)
PROPORTIONS = torch.zeros(32)
PROPORTIONS[[2, 6, 9, 11]] = torch.tensor([0.3, 0.6, 0.6, 1.0])


def _reuse_prefill(lm, embeds, cache, output_attentions):
    lm.config.reusable_patches = REUSABLE
    lm.config.proportion_attn_var = PROPORTIONS
    try:
        return run_lm(lm, embeds, cache, output_attentions)
    finally:
        lm.config.reusable_patches = None
        lm.config.proportion_attn_var = None


def _setup():
    lm = build_tiny_llama(dtype=torch.float32)
    g = torch.Generator().manual_seed(0)
    prev = torch.randn(1, SEQ_LEN, 64, generator=g)
    cur = prev + 0.5 * torch.randn(1, SEQ_LEN, 64, generator=g)
    # The fork re-wraps the passed cache; like `generate`, continue from the returned one
    cache = run_lm(lm, prev, DynamicCache(), output_attentions=False).past_key_values
    return lm, cur, cache


def test_vprime_reuse_prefill_matches_v_up_to_sdpa_numerics():
    lm, cur, cache = _setup()
    install_vprime(as_vla(lm))
    v_out = _reuse_prefill(lm, cur, copy.deepcopy(cache), output_attentions=True)  # V: eager
    p_out = _reuse_prefill(lm, cur, copy.deepcopy(cache), output_attentions=False)  # V': SDPA + explicit mask
    v_cache, p_cache = v_out.past_key_values, p_out.past_key_values

    torch.testing.assert_close(p_out.logits, v_out.logits, rtol=1e-4, atol=1e-4)
    for layer in range(len(lm.model.layers)):
        torch.testing.assert_close(p_cache.key_cache[layer], v_cache.key_cache[layer], rtol=1e-4, atol=1e-4)
    # The reuse dropped tokens and the cache still covers the whole prompt
    assert p_out.logits.shape[1] == SEQ_LEN - len(REUSABLE)
    assert p_cache.get_seq_length() == SEQ_LEN


def test_fork_sdpa_shortcut_is_wrong_for_reuse_prefill():
    """Regression check for the pitfall `install_vprime` fixes: the fork alone uses SDPA's top-left `is_causal`."""
    lm, cur, cache = _setup()
    v_out = _reuse_prefill(lm, cur, copy.deepcopy(cache), output_attentions=True)
    fork_out = _reuse_prefill(lm, cur, copy.deepcopy(cache), output_attentions=False)
    assert (fork_out.logits - v_out.logits).abs().max() > 1e-2
