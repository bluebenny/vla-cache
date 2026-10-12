"""
tiny_openvla.py

CPU stand-ins for the unit tests and `verify_vprime_equivalence.py --tiny_random`: a small random OpenVLA built from
the real model classes (32-layer LLaMA, so layer 15 and `pruning_loc` exist), a processor that skips the tokenizer,
synthetic frame pairs, and stubs for the heavy imports that the CPU environment does not have.
"""

import hashlib
import importlib.machinery
import importlib.util
import os
import sys
import types

import numpy as np
import torch
from PIL import Image

OPENVLA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

TINY_UNNORM_KEY = "tiny"
TINY_ACTION_DIM = 7


class _NoCudaEvent:
    def __init__(self, enable_timing=False, **kwargs):
        pass

    def record(self, stream=None):
        pass

    def elapsed_time(self, end_event):
        return 0.0


def install_light_import_stubs() -> None:
    """
    Lets `experiments.robot.openvla_utils` import on a CPU box with only torch / transformers / timm installed:
      - `prismatic` as a bare package, skipping `prismatic/__init__` (training, datasets, TensorFlow, draccus, ...)
      - `tensorflow` (only used by `process_image`, i.e. `center_crop=True`) when it is not installed
      - `torch.cuda.Event` / `torch.cuda.synchronize` without CUDA (the forked `LlamaModel.forward` calls them)
    """
    if "prismatic" not in sys.modules:
        package = types.ModuleType("prismatic")
        package.__path__ = [os.path.join(OPENVLA_ROOT, "prismatic")]
        sys.modules["prismatic"] = package

    if "tensorflow" not in sys.modules and importlib.util.find_spec("tensorflow") is None:
        import transformers  # noqa: F401  (records "no TensorFlow" before the stub exists)

        stub = types.ModuleType("tensorflow")
        stub.__spec__ = importlib.machinery.ModuleSpec("tensorflow", loader=None)
        sys.modules["tensorflow"] = stub

    if not torch.cuda.is_available():
        torch.cuda.Event = _NoCudaEvent
        torch.cuda.synchronize = lambda device=None: None


def build_tiny_openvla(
    num_layers: int = 32,
    hidden_size: int = 64,
    num_heads: int = 4,
    attn_implementation: str = "sdpa",
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 0,
    qk_scale: float = 8.0,
):
    """
    Random OpenVLA with the real classes: small DINOv2 ViT (224px, patch 14 -> 256 patches) and a small LLaMA.
    `qk_scale` sharpens the random attention so that the selections are not dominated by ties.
    """
    install_light_import_stubs()
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction

    torch.manual_seed(seed)
    config = OpenVLAConfig(
        vision_backbone_id="dinov2-vit-l",
        llm_backbone_id="llama2-7b-pure",
        text_config={
            "hidden_size": hidden_size,
            "intermediate_size": 2 * hidden_size,
            "num_hidden_layers": num_layers,
            "num_attention_heads": num_heads,
            "num_key_value_heads": num_heads,
            "vocab_size": 32064,
            "max_position_embeddings": 2048,
            "pad_token_id": 32000,
        },
        norm_stats={
            TINY_UNNORM_KEY: {
                "action": {
                    "q01": [-1.0] * TINY_ACTION_DIM,
                    "q99": [1.0] * TINY_ACTION_DIM,
                    "mask": [True] * (TINY_ACTION_DIM - 1) + [False],
                }
            }
        },
        attn_implementation=attn_implementation,
    )
    # Small stand-in for the 300M-parameter ViT-L (same patch size and number of patches)
    config.timm_model_ids = ["vit_small_patch14_dinov2"]

    vla = OpenVLAForActionPrediction(config)
    with torch.no_grad():
        for layer in vla.language_model.model.layers:
            layer.self_attn.q_proj.weight.mul_(qk_scale)
            layer.self_attn.k_proj.weight.mul_(qk_scale)
        # `prepare_inputs_for_generation` treats a trailing 29871 as "prompt not consumed yet"; never generate it
        vla.language_model.lm_head.weight[29871] = 0.0
    vla = vla.to(dtype).eval()
    # Random weights can emit any token; always decode the full action
    vla.generation_config.eos_token_id = None
    vla.generation_config.pad_token_id = config.pad_token_id
    return vla


class TinyProcessor:
    """Replaces `PrismaticProcessor` without a tokenizer: deterministic token ids per prompt, 224px pixel values."""

    def __init__(self, num_text_tokens: int = 34) -> None:
        self.num_text_tokens = num_text_tokens

    def __call__(self, prompt, image):
        from transformers import BatchFeature

        seed = int(hashlib.sha256(prompt.encode()).hexdigest()[:8], 16)
        text_ids = torch.randint(3, 31000, (self.num_text_tokens,), generator=torch.Generator().manual_seed(seed))
        input_ids = torch.cat([torch.tensor([1]), text_ids])[None]

        pixels = np.asarray(image.convert("RGB").resize((224, 224)), dtype=np.float32) / 255.0
        pixel_values = (torch.from_numpy(pixels).permute(2, 0, 1)[None] - 0.5) / 0.5
        return BatchFeature(
            {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids), "pixel_values": pixel_values}
        )


def synthetic_frame_pair(seed: int = 0, moved_fraction: float = 0.4):
    """224x224 RGB frames on the 14px patch grid; `moved_fraction` of the patches change between the two."""
    rng = np.random.default_rng(seed)
    prev = rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)
    cur = prev.copy()
    grid = 224 // 14
    moved = rng.choice(grid * grid, size=int(moved_fraction * grid * grid), replace=False)
    for pid in moved:
        i, j = divmod(int(pid), grid)
        cur[i * 14 : (i + 1) * 14, j * 14 : (j + 1) * 14] = rng.integers(0, 256, size=(14, 14, 3), dtype=np.uint8)
    return prev, cur


def frame_to_image(frame) -> Image.Image:
    return Image.fromarray(frame).convert("RGB")
