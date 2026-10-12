"""
verify_vprime_equivalence.py

Fixed-input check that V' (VLA-Cache on the SDPA path) makes the same reuse decisions as V (public VLA-Cache, eager
attention). For one frame pair (prev -> cur) and one instruction, each run calls `get_vla_action` on the previous
frame with an empty cache (this produces the "previous attention"), then on the current frame with the reuse that
follows from it. One model serves all runs: `install_vprime` leaves V's path (output_attentions=True) untouched.

    A  logic      V's attention maps -> V's selection, and the same maps reduced to V''s statistics -> V''s
                  selection: same reusable indices and per-layer proportions (exact).
    B  in-module  V' with its attention output computed eagerly (`eager_debug`, so its hidden states match V):
                  in-module statistics == reduction of V's maps, and the current step makes the same decisions,
                  computes the same positions per layer and outputs the same action as V (exact).
    C  SDPA       V' as deployed: how far the SDPA numerics move the statistics, decisions and actions (info).

Usage:
    # CPU: tiny random model + synthetic frames
    python vla_cache_scripts/verify_vprime_equivalence.py --tiny_random

    # GPU: real checkpoint and frames
    python vla_cache_scripts/verify_vprime_equivalence.py \
        --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
        --prev_image prev.png --cur_image cur.png --center_crop \
        --instruction "pick up the black bowl between the plate and the ramekin and place it on the plate" \
        --unnorm_key libero_spatial --output verify_report.json

Exits with status 1 if check A or B fails.
"""

import argparse
import json
import os
import sys
from functools import partial
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

OPENVLA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if OPENVLA_ROOT not in sys.path:
    sys.path.insert(0, OPENVLA_ROOT)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tiny_random", action="store_true", help="CPU smoke test: tiny random model, synthetic frames")
    parser.add_argument("--pretrained_checkpoint", default=None)
    parser.add_argument("--prev_image", default=None, help="Previous frame (image file, 224x224 like the eval)")
    parser.add_argument("--cur_image", default=None, help="Current frame (image file)")
    parser.add_argument("--instruction", default="pick up the black bowl and place it on the plate")
    parser.add_argument("--unnorm_key", default=None)
    parser.add_argument("--center_crop", action="store_true", help="Same as the LIBERO eval's center_crop=True")
    parser.add_argument("--seed", type=int, default=0, help="Tiny model / synthetic frame seed")
    parser.add_argument("--output", default=None, help="Write the JSON report here")
    args = parser.parse_args(argv)
    real_inputs = (args.pretrained_checkpoint, args.prev_image, args.cur_image)
    if not args.tiny_random and any(arg is None for arg in real_inputs):
        parser.error("--pretrained_checkpoint, --prev_image and --cur_image are required without --tiny_random")
    return args


class PrefillRecorder:
    """While armed, records what the prefill actually does: the reuse decisions and the query positions per layer."""

    def __init__(self, vla) -> None:
        llama = vla.language_model.model
        self.config = vla.language_model.config
        self.handles = [llama.register_forward_pre_hook(self._model_pre, with_kwargs=True)]
        for idx, layer in enumerate(llama.layers):
            self.handles.append(layer.register_forward_pre_hook(partial(self._layer_pre, idx), with_kwargs=True))
        self.armed = False
        self._in_prefill = False
        self.decisions, self.positions = None, {}

    def arm(self) -> None:
        self.armed = True
        self.decisions, self.positions = None, {}

    def disarm(self) -> dict:
        self.armed = False
        return {"decisions": self.decisions, "positions": self.positions}

    def _model_pre(self, module, args, kwargs) -> None:
        embeds = kwargs.get("inputs_embeds")
        self._in_prefill = self.armed and embeds is not None and embeds.shape[1] > 1
        if self._in_prefill:
            proportions = self.config.proportion_attn_var
            patches = self.config.reusable_patches if proportions is not None else None
            self.decisions = {
                "reusable_patches": patches.tolist() if patches is not None else None,
                "proportions": proportions.tolist() if proportions is not None else None,
            }

    def _layer_pre(self, idx, module, args, kwargs) -> None:
        if self._in_prefill:
            self.positions[idx] = kwargs["cache_position"].tolist()

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()


def load_model(args):
    """Returns (vla, processor, unnorm_key, base_vla_name, (prev_frame, cur_frame)) with V' installed."""
    if args.tiny_random:
        from experiments.robot.tiny_openvla import (
            TINY_UNNORM_KEY,
            TinyProcessor,
            build_tiny_openvla,
            install_light_import_stubs,
            synthetic_frame_pair,
        )
        from experiments.robot.vla_cache_sdpa import install_vprime

        install_light_import_stubs()
        vla = build_tiny_openvla(seed=args.seed)
        install_vprime(vla)
        frames = synthetic_frame_pair(seed=args.seed)
        return vla, TinyProcessor(), TINY_UNNORM_KEY, "tiny-random", frames

    from experiments.robot.openvla_utils import get_processor, get_vla

    # Load like V' (SDPA explicit + install_vprime); V runs on the same model through its eager fallback
    cfg = SimpleNamespace(
        pretrained_checkpoint=args.pretrained_checkpoint,
        load_in_8bit=False,
        load_in_4bit=False,
        vla_mode="Vp",
        use_vla_cache=True,
    )
    vla = get_vla(cfg)
    processor = get_processor(cfg)
    unnorm_key = args.unnorm_key
    if unnorm_key not in vla.norm_stats and f"{unnorm_key}_no_noops" in vla.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"
    frames = tuple(np.array(Image.open(path).convert("RGB")) for path in (args.prev_image, args.cur_image))
    return vla, processor, unnorm_key, str(args.pretrained_checkpoint), frames


def run_pair(vla, processor, recorder, args, unnorm_key, base_vla_name, frames, mode, attn_compute="sdpa"):
    """prev frame with an empty cache, then the current frame with reuse; returns what the current step did."""
    from experiments.robot.openvla_utils import get_vla_action

    if mode == "Vp":
        vla.vprime_collector.attn_compute = attn_compute
    cfg = SimpleNamespace(vla_mode=mode, use_vla_cache=mode in ("V", "Vp"))
    prev_frame, cur_frame = frames
    query = partial(get_vla_action, cfg, vla, processor, base_vla_name, task_label=args.instruction,
                    unnorm_key=unnorm_key, center_crop=args.center_crop)

    _, prev_caches, _ = query(obs={"full_image": prev_frame, "prev_image": prev_frame}, last_caches=None)
    recorder.arm()
    try:
        action, _, _ = query(obs={"full_image": cur_frame, "prev_image": prev_frame}, last_caches=prev_caches)
    finally:
        recorded = recorder.disarm()
    return {"prev_caches": prev_caches, "action": action, **recorded}


def selection_images(args, frames):
    """The (current, previous) images exactly as `get_vla_action` hands them to the selection."""
    prev_image = Image.fromarray(frames[0])
    image = Image.fromarray(frames[1]).convert("RGB")
    if args.center_crop:
        from experiments.robot.openvla_utils import process_image

        image, prev_image = process_image(image), process_image(prev_image)
    return image, prev_image


def _stats_equal(a, b) -> bool:
    return all(torch.equal(x, y) for x, y in zip(a.entropies, b.entropies)) and torch.equal(
        a.text_vision_score, b.text_vision_score
    )


def _jaccard(a, b) -> float:
    a, b = set(a or []), set(b or [])
    return 1.0 if not a and not b else len(a & b) / len(a | b)


def verify(args) -> dict:
    vla, processor, unnorm_key, base_vla_name, frames = load_model(args)

    from experiments.robot.vla_cache_sdpa import attn_stats_from_attentions
    from experiments.robot.vla_cache_utils import (
        find_static_patches,
        get_layer_mask_schedule,
        layer_mask_schedule_from_entropies,
        task_relevant_selection,
        task_relevant_selection_from_scores,
    )

    recorder = PrefillRecorder(vla)
    run = partial(run_pair, vla, processor, recorder, args, unnorm_key, base_vla_name, frames)
    v_run = run("V")
    debug_run = run("Vp", attn_compute="eager_debug")
    sdpa_run = run("Vp", attn_compute="sdpa")
    recorder.remove()

    # A: same attention maps through both selections
    prev_attn = v_run["prev_caches"]["attentions"]
    image, prev_image = selection_images(args, frames)
    stable_patches = find_static_patches(image, prev_image, top_k=130)
    _, remaining_v = task_relevant_selection(prev_attn, image, stable_patches)
    schedule_v = get_layer_mask_schedule(prev_attn)
    reduced = attn_stats_from_attentions(prev_attn)
    _, remaining_p = task_relevant_selection_from_scores(reduced.text_vision_score, image, stable_patches)
    schedule_p = layer_mask_schedule_from_entropies(reduced.entropies)
    check_a = {
        "reusable_equal": remaining_v == remaining_p,
        "proportions_equal": torch.equal(schedule_v, schedule_p),
        "matches_v_run": v_run["decisions"]["reusable_patches"] == remaining_v,
        "num_stable": len(stable_patches),
        "num_reusable": len(remaining_v),
    }
    check_a["pass"] = check_a["reusable_equal"] and check_a["proportions_equal"] and check_a["matches_v_run"]

    # B: in-module statistics with eager numerics
    v_tokens = [len(v_run["positions"][i]) for i in sorted(v_run["positions"])]
    check_b = {
        "stats_equal": _stats_equal(debug_run["prev_caches"]["attn_stats"], reduced),
        "decisions_equal": debug_run["decisions"] == v_run["decisions"],
        "positions_equal": debug_run["positions"] == v_run["positions"],
        "action_equal": bool(np.array_equal(debug_run["action"], v_run["action"])),
    }
    check_b["pass"] = all(check_b.values())
    check_b["tokens_per_layer"] = v_tokens

    # C: deployed V' (SDPA numerics), informational
    sdpa_stats = sdpa_run["prev_caches"]["attn_stats"]
    v_dec, p_dec = v_run["decisions"], sdpa_run["decisions"]
    p_tokens = [len(sdpa_run["positions"][i]) for i in sorted(sdpa_run["positions"])]
    check_c = {
        "entropy_max_abs_diff": max(
            (a.float() - b.float()).abs().item() for a, b in zip(sdpa_stats.entropies, reduced.entropies)
        ),
        "text_vision_score_max_abs_diff": (sdpa_stats.text_vision_score - reduced.text_vision_score).abs().max().item(),
        "reusable_equal": p_dec["reusable_patches"] == v_dec["reusable_patches"],
        "reusable_jaccard": _jaccard(p_dec["reusable_patches"], v_dec["reusable_patches"]),
        "proportions_max_abs_diff": float(
            np.max(np.abs(np.array(p_dec["proportions"]) - np.array(v_dec["proportions"])))
        ),
        "tokens_per_layer_v": v_tokens,
        "tokens_per_layer_vprime": p_tokens,
        "action_equal": bool(np.array_equal(sdpa_run["action"], v_run["action"])),
        "action_max_abs_diff": float(np.max(np.abs(sdpa_run["action"] - v_run["action"]))),
    }

    return {
        "A_logic": check_a,
        "B_in_module": check_b,
        "C_sdpa_info": check_c,
        "pass": check_a["pass"] and check_b["pass"],
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    report = verify(args)
    text = json.dumps(report, indent=2)
    print(text)
    if args.output is not None:
        with open(args.output, "w") as f:
            f.write(text + "\n")
    print("PASS" if report["pass"] else "FAIL")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
