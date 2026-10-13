"""The shared helpers must leave V's public selection functions bit-identical to the original code."""

import numpy as np
import pytest
import torch
from helpers import load_public_vla_cache_utils, random_attention_tuple, random_image

from experiments.robot import vla_cache_utils as utils

public = load_public_vla_cache_utils()
PRUNED = torch.arange(1, 257, 3)  # vision positions whose rows layers 2+ drop


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("pruned", [None, PRUNED])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_selection_matches_public_code(seed, pruned, dtype):
    attentions = random_attention_tuple(seed, pruned=pruned, dtype=dtype)
    image, prev_image = random_image(seed), random_image(seed + 100)
    significant = public.find_static_patches(image, image, top_k=130)  # identical frames -> all patches static

    assert torch.equal(utils.get_layer_mask_schedule(attentions), public.get_layer_mask_schedule(attentions))
    assert torch.equal(utils.token_attention_merge(attentions), public.token_attention_merge(attentions))
    assert utils.find_static_patches(image, prev_image, top_k=130) == public.find_static_patches(
        image, prev_image, top_k=130
    )

    overlay, remaining = utils.task_relevant_selection(attentions, image, significant)
    overlay_ref, remaining_ref = public.task_relevant_selection(attentions, image, significant)
    assert remaining == remaining_ref
    assert np.array_equal(overlay, overlay_ref)


@pytest.mark.parametrize("pruned", [None, PRUNED])
def test_helpers_reproduce_public_functions(pruned):
    """V' feeds the helpers per-layer values instead of the attention tuple; same results."""
    attentions = random_attention_tuple(3, pruned=pruned)
    image = random_image(3)
    significant = list(range(0, 256, 2))

    entropies = [utils.layer_attention_entropy(a) for a in attentions[:-1]]
    assert torch.equal(
        utils.layer_mask_schedule_from_entropies(entropies), public.get_layer_mask_schedule(attentions)
    )

    score = utils.text_vision_relation(attentions[utils.ATTN_LAYER_ID], attentions[-1])
    overlay, remaining = utils.task_relevant_selection_from_scores(score, image, significant)
    overlay_ref, remaining_ref = public.task_relevant_selection(attentions, image, significant)
    assert remaining == remaining_ref
    assert np.array_equal(overlay, overlay_ref)
