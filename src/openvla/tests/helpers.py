"""Shared builders for the CPU tests."""

import importlib.util
import os
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
SEQ_LEN = 292  # <BOS> + 256 vision patches + 35 text tokens, like an OpenVLA LIBERO prompt


def load_module_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_public_vla_cache_utils():
    return load_module_from_path(
        "vla_cache_utils_public", os.path.join(TESTS_DIR, "reference", "vla_cache_utils_public.py")
    )


def random_attention_tuple(seed, num_layers=32, heads=4, seq_len=SEQ_LEN, pruned=None, dtype=torch.bfloat16):
    """V-style attention tuple: per-layer causal maps [1, heads, q, kv] + the final query positions.
    With `pruned`, the rows of those positions are dropped from layer 2 on (like `LlamaModel` does)."""
    g = torch.Generator().manual_seed(seed)
    full = torch.arange(seq_len)
    positions = full
    attentions = []
    for layer in range(num_layers):
        if pruned is not None and layer == 2:
            positions = full[~torch.isin(full, pruned)]
        scores = torch.randn(1, heads, len(positions), seq_len, generator=g) * 3
        scores = scores.masked_fill(full[None, :] > positions[:, None], float("-inf"))
        attentions.append(scores.softmax(-1).to(dtype))
    attentions.append(positions)
    return tuple(attentions)


def random_image(seed):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8))


def build_tiny_llama(dtype=torch.float32, seed=0, num_layers=32, attn_implementation="sdpa", qk_scale=8.0):
    """Bare forked LlamaForCausalLM with the VLA-Cache config fields OpenVLA sets."""
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    config = LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=128,
        max_position_embeddings=1024,
        attn_implementation=attn_implementation,
    )
    lm = LlamaForCausalLM(config)
    with torch.no_grad():
        for layer in lm.model.layers:
            layer.self_attn.q_proj.weight.mul_(qk_scale)
            layer.self_attn.k_proj.weight.mul_(qk_scale)
    lm = lm.to(dtype).eval()
    lm.config.proportion_attn_var = None
    lm.config.reusable_patches = None
    return lm


def as_vla(lm):
    """`install_vprime` only needs `.language_model`."""
    return SimpleNamespace(language_model=lm)


def run_lm(lm, embeds, cache, output_attentions):
    """Prefill like `PrismaticForConditionalGeneration` (its 2D attention mask is one token short)."""
    with torch.no_grad():
        return lm(
            inputs_embeds=embeds,
            attention_mask=torch.ones(1, embeds.shape[1] - 1, dtype=torch.long),
            past_key_values=cache,
            use_cache=True,
            output_attentions=output_attentions,
            return_dict=True,
        )
