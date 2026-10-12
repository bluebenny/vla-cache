"""
vla_cache_sdpa.py

Inference modes B0 / B1 / V / V' and the V' implementation (VLA-Cache on the SDPA attention path).

    B0 : VLA-Cache off, attention outputs off, LLM loaded with `attn_implementation="sdpa"`.
    B1 : public baseline (`use_vla_cache=False`); `output_attentions=True` makes the SDPA layers fall back to eager.
    V  : public VLA-Cache (`use_vla_cache=True`), same eager fallback as B1.
    Vp : V' -- the reuse decisions of V, computed on the SDPA path of B0.

V needs the full attention maps of the previous prefill, but only reduces them to a few numbers: the text->vision
relation of layer `ATTN_LAYER_ID` and one entropy scalar per layer. V' computes exactly those numbers inside the
attention modules during the prefill (same ops and dtypes as the eager attention) and never returns the maps.

The reuse itself (token dropping at `LlamaModel.pruning_loc`, partial KV writes, cache crop) is the forked
`transformers` code (see pyproject.toml), which does not depend on the attention class -- with one exception:
with SDPA and `output_attentions=False` its `_update_causal_mask` returns None for the reuse prefill, so the pruned
query rows are never sliced out of the mask and SDPA's `is_causal` (top-left aligned when q_len < kv_len) masks
them wrongly. `install_vprime` makes the mask explicit for that prefill.
"""

import hashlib
import inspect
import math
import types
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F
from transformers import DynamicCache
from transformers.models.llama.modeling_llama import LlamaSdpaAttention, apply_rotary_pos_emb, repeat_kv

from .vla_cache_utils import ATTN_LAYER_ID, layer_attention_entropy, text_vision_relation

VLA_MODES = ("B0", "B1", "V", "Vp")
# Modes that pin the LLM to SDPA explicitly when loading (B1 / V keep the public loading call)
SDPA_MODES = ("B0", "Vp")
# Modes that call `generate(output_attentions=True)`
ATTENTION_OUTPUT_MODES = ("B1", "V")
_MODE_ALIASES = {"V'": "Vp", "VP": "Vp", "VPRIME": "Vp"}

# How V' computes the attention output: "sdpa", or "eager_debug" (eager math, for the equivalence check only)
ATTN_COMPUTE_CHOICES = ("sdpa", "eager_debug")

# sha256 prefix of `LlamaSdpaAttention.forward` in the forked transformers commit that
# `VPrimeLlamaSdpaAttention.forward` was copied from (siyuhsu/transformers@2302fce)
_FORK_SDPA_FORWARD_SHA = "7660ca1ac5b9dccf"


def get_vla_mode(cfg) -> str:
    """Inference mode of `cfg`: `vla_mode` if set, else the public `use_vla_cache` switch (True -> V, False -> B1)."""
    mode = getattr(cfg, "vla_mode", None)
    if mode is None:
        return "V" if cfg.use_vla_cache else "B1"
    mode = _MODE_ALIASES.get(str(mode).upper(), mode)
    if mode not in VLA_MODES:
        raise ValueError(f"Unknown vla_mode `{mode}`; expected one of {VLA_MODES}")
    return mode


def resolve_vla_mode(cfg) -> str:
    """Normalizes `cfg.vla_mode` and keeps `cfg.use_vla_cache` consistent with it (`vla_mode` wins)."""
    mode = get_vla_mode(cfg)
    cfg.vla_mode = mode
    cfg.use_vla_cache = mode in ("V", "Vp")
    return mode


@dataclass
class AttnStats:
    """What V' keeps from one prefill instead of the attention maps."""

    # Per-layer mean token entropy (0-dim, attention dtype), as `get_layer_mask_schedule` computes it
    entropies: List[torch.Tensor]
    # Text->vision relation of `ATTN_LAYER_ID` (fp32, on device), as `token_attention_merge` computes it
    text_vision_score: torch.Tensor


def attn_stats_from_attentions(multihead_attention) -> AttnStats:
    """Reduces V's attention tuple (per-layer maps + final `cache_position`) to the statistics V' collects."""
    return AttnStats(
        entropies=[layer_attention_entropy(attn) for attn in multihead_attention[:-1]],
        text_vision_score=text_vision_relation(multihead_attention[ATTN_LAYER_ID], multihead_attention[-1]),
    )


def causal_mask_from_positions(cache_position, target_length, dtype):
    """Causal mask with the values `LlamaModel._update_causal_mask` builds for a dynamic cache without padding."""
    min_dtype = torch.finfo(dtype).min
    device = cache_position.device
    mask = torch.full((cache_position.shape[0], target_length), fill_value=min_dtype, dtype=dtype, device=device)
    mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
    return mask[None, None, :, :]


def eager_attention_weights(query_states, key_states, causal_mask, cache_position):
    """Attention probabilities exactly as `LlamaAttention` (eager) computes them, in the query dtype."""
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(query_states.shape[-1])
    if causal_mask is None:
        causal_mask = causal_mask_from_positions(cache_position, key_states.shape[-2], attn_weights.dtype)
    attn_weights = attn_weights + causal_mask[:, :, :, : key_states.shape[-2]]
    return F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)


class AttnStatsCollector:
    """Receives the per-layer statistics from `VPrimeLlamaSdpaAttention` during a prefill."""

    def __init__(self, num_layers: int, attn_compute: str = "sdpa") -> None:
        if attn_compute not in ATTN_COMPUTE_CHOICES:
            raise ValueError(f"Unknown attn_compute `{attn_compute}`; expected one of {ATTN_COMPUTE_CHOICES}")
        self.num_layers = num_layers
        self.attn_compute = attn_compute
        self.enabled = False
        self.timer = None  # Optional `StageTimer`; times the statistics as "attn_info"
        self.reset()

    def reset(self) -> None:
        self._entropies: List[Optional[torch.Tensor]] = [None] * self.num_layers
        self._text_vision_score: Optional[torch.Tensor] = None

    @torch.no_grad()
    def record(self, layer_idx, query_states, key_states, causal_mask, cache_position) -> torch.Tensor:
        """Computes this layer's statistics; returns the attention probabilities (only `eager_debug` uses them)."""
        if self.timer is not None:
            self.timer.start("attn_info")
        attn_weights = eager_attention_weights(query_states, key_states, causal_mask, cache_position)
        self._entropies[layer_idx] = layer_attention_entropy(attn_weights)
        if layer_idx == ATTN_LAYER_ID:
            self._text_vision_score = text_vision_relation(attn_weights, cache_position)
        if self.timer is not None:
            self.timer.stop("attn_info")
        return attn_weights

    def pop(self) -> AttnStats:
        missing = [i for i, e in enumerate(self._entropies) if e is None]
        if missing or self._text_vision_score is None:
            raise RuntimeError(f"V' prefill statistics are incomplete (missing layers: {missing})")
        stats = AttnStats(entropies=self._entropies, text_vision_score=self._text_vision_score)
        self.reset()
        return stats


class VPrimeLlamaSdpaAttention(LlamaSdpaAttention):
    """
    `LlamaSdpaAttention` of the forked transformers (copied below), plus the V' prefill statistics.
    With the collector absent or disabled, the outputs are those of the original class.
    """

    _vprime_collector: Optional[AttnStatsCollector] = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        if output_attentions:
            # Same as the original class: warn and fall back to the eager implementation (B1 / V)
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # In case static cache is used, it is an instance attribute.
        past_key_value = getattr(self, "past_key_value", past_key_value)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        # [V'] Prefill statistics, from the same query/key tensors (and layouts) the eager attention would use
        collector = self._vprime_collector
        attn_weights = None
        if collector is not None and collector.enabled and q_len > 1:
            attn_weights = collector.record(self.layer_idx, query_states, key_states, causal_mask, cache_position)

        if collector is not None and collector.attn_compute == "eager_debug":
            if attn_weights is None:
                attn_weights = eager_attention_weights(query_states, key_states, causal_mask, cache_position)
            attn_output = torch.matmul(attn_weights, value_states)
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
            attn_output = self.o_proj(attn_output)
            return attn_output, None, past_key_value
        del attn_weights

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # In case we are not compiling, we may set `causal_mask` to None, which is required to dispatch to SDPA's Flash Attention 2 backend, rather
        # relying on the `is_causal` argument.
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=causal_mask is None and q_len > 1,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


def _reuse_aware_update_causal_mask(
    self, attention_mask, input_tensor, cache_position, past_seen_tokens, output_attentions
):
    # The fork's SDPA shortcut returns None (`is_causal`) for the reuse prefill; build the mask V uses instead so
    # that `LlamaModel.forward` can slice out the rows of the dropped tokens.
    reuse_prefill = getattr(self.config, "proportion_attn_var", None) is not None and input_tensor.shape[1] > 1
    return type(self)._update_causal_mask(
        self, attention_mask, input_tensor, cache_position, past_seen_tokens, output_attentions or reuse_prefill
    )


def fork_sdpa_forward_sha() -> str:
    return hashlib.sha256(inspect.getsource(LlamaSdpaAttention.forward).encode()).hexdigest()[:16]


def install_vprime(vla, attn_compute: str = "sdpa") -> AttnStatsCollector:
    """
    Turns a loaded OpenVLA (LLM on SDPA) into V': swaps the attention classes, makes the reuse-prefill causal mask
    explicit, and attaches the statistics collector as `vla.vprime_collector`.
    """
    language_model = vla.language_model
    llama = language_model.model

    if language_model.config._attn_implementation != "sdpa":
        raise ValueError(
            f"V' needs the LLM on SDPA, got `{language_model.config._attn_implementation}` "
            '(load with attn_implementation="sdpa")'
        )
    if not hasattr(llama, "pruning_loc") or not hasattr(DynamicCache(), "cache_position"):
        raise RuntimeError("V' needs the VLA-Cache fork of transformers (see pyproject.toml)")
    if max(llama.pruning_loc) >= ATTN_LAYER_ID:
        # V reads the layer-15 query positions from the *final* `cache_position`; they only agree if no pruning
        # happens at or after that layer
        raise ValueError(f"pruning_loc {llama.pruning_loc} must end before layer {ATTN_LAYER_ID}")
    if getattr(language_model.config, "pretraining_tp", 1) != 1:
        raise ValueError("V' statistics mirror the eager attention for pretraining_tp == 1 only")
    if fork_sdpa_forward_sha() != _FORK_SDPA_FORWARD_SHA:
        print(
            "WARNING: transformers' LlamaSdpaAttention.forward differs from the version V' was copied from; "
            "re-sync VPrimeLlamaSdpaAttention.forward (experiments/robot/vla_cache_sdpa.py)"
        )

    collector = AttnStatsCollector(num_layers=len(llama.layers), attn_compute=attn_compute)
    for layer in llama.layers:
        if type(layer.self_attn) is not LlamaSdpaAttention:
            raise TypeError(f"Expected LlamaSdpaAttention, got {type(layer.self_attn).__name__}")
        layer.self_attn.__class__ = VPrimeLlamaSdpaAttention
        layer.self_attn._vprime_collector = collector

    llama._update_causal_mask = types.MethodType(_reuse_aware_update_causal_mask, llama)
    vla.vprime_collector = collector
    return collector
