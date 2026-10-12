"""B0 / B1 / V / V' through `get_vla_action` + `predict_action` on a tiny random OpenVLA."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.robot.openvla_utils import get_vla_action
from experiments.robot.tiny_openvla import TINY_UNNORM_KEY, TinyProcessor, build_tiny_openvla, synthetic_frame_pair
from experiments.robot.vla_cache_sdpa import get_vla_mode, install_vprime, resolve_vla_mode
from experiments.robot.vla_timing import attach_stage_timer

INSTRUCTION = "pick up the black bowl and place it on the plate"


def _frames(num_steps=3):
    frames = [synthetic_frame_pair(seed=0)[0]]
    for step in range(1, num_steps):
        frames.append(synthetic_frame_pair(seed=step, moved_fraction=0.3)[1])
    # Make every frame share most patches with the previous one
    for step in range(1, num_steps):
        keep = np.random.default_rng(step).random((16, 16)) < 0.7
        mask = np.kron(keep, np.ones((14, 14), dtype=bool))[..., None]
        frames[step] = np.where(mask, frames[step - 1], frames[step])
    return frames


def _rollout(vla, mode, frames, record=None):
    cfg = SimpleNamespace(vla_mode=mode, use_vla_cache=mode in ("V", "Vp"))
    processor = TinyProcessor()
    last_caches, prev, outputs = None, None, []
    for frame in frames:
        prev = frame if prev is None else prev
        action, last_caches, overlay = get_vla_action(
            cfg, vla, processor, "tiny", {"full_image": frame, "prev_image": prev}, INSTRUCTION, TINY_UNNORM_KEY,
            center_crop=False, last_caches=last_caches,
        )
        config = vla.language_model.config
        outputs.append({
            "action": action,
            "last_caches": last_caches,
            "overlay": overlay,
            "reuse": config.proportion_attn_var is not None,
            "reusable": config.reusable_patches.tolist() if config.proportion_attn_var is not None else None,
            "proportions": config.proportion_attn_var.tolist() if config.proportion_attn_var is not None else None,
        })
        prev = frame
    return outputs


@pytest.fixture(scope="module")
def vla():
    model = build_tiny_openvla()
    install_vprime(model)
    return model


def test_b0_runs_without_attentions():
    model = build_tiny_openvla()  # B0 never installs V'
    outputs = _rollout(model, "B0", _frames(2))
    for out in outputs:
        assert out["action"].shape == (7,)
        assert out["last_caches"]["attentions"] is None
        assert not out["reuse"]


def test_b1_keeps_attention_outputs(vla):
    outputs = _rollout(vla, "B1", _frames(2))
    attentions = outputs[-1]["last_caches"]["attentions"]
    assert len(attentions) == len(vla.language_model.model.layers) + 1
    assert not any(out["reuse"] for out in outputs)


def test_vprime_eager_debug_reproduces_v(vla):
    frames = _frames(3)
    v_out = _rollout(vla, "V", frames)
    vla.vprime_collector.attn_compute = "eager_debug"
    try:
        p_out = _rollout(vla, "Vp", frames)
    finally:
        vla.vprime_collector.attn_compute = "sdpa"

    assert [o["reuse"] for o in v_out] == [False, True, True]
    for v, p in zip(v_out, p_out):
        assert v["reusable"] == p["reusable"]
        assert v["proportions"] == p["proportions"]
        assert np.array_equal(v["action"], p["action"])
        assert np.array_equal(v["overlay"], p["overlay"])
        assert p["last_caches"]["attentions"] is None
        assert p["last_caches"]["attn_stats"] is not None


def test_vprime_sdpa_reuses(vla):
    p_out = _rollout(vla, "Vp", _frames(3))
    assert [o["reuse"] for o in p_out] == [False, True, True]
    assert all(len(o["reusable"]) > 0 for o in p_out[1:])
    # The cache is cropped back to the prompt after each query, like V
    cache = p_out[-1]["last_caches"]["past_key_values"]
    assert cache.get_seq_length() == 256 + 36


def test_stage_timing_rows(tmp_path):
    model = build_tiny_openvla()
    install_vprime(model)
    log_path = tmp_path / "timing.jsonl"
    attach_stage_timer(model, str(log_path))
    _rollout(model, "Vp", _frames(2))

    rows = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert [row["step"] for row in rows] == [0, 1]
    for row in rows:
        for stage in ("vision", "prefill", "decode", "attn_info", "preprocess", "predict_total", "step_total"):
            assert f"{stage}_ms" in row and f"{stage}_wall_ms" in row
        assert row["decode_n"] == 6
        assert row["attn_info_n"] == len(model.language_model.model.layers)
        assert row["mode"] == "Vp"
    assert not rows[0]["reuse"] and rows[1]["reuse"]
    assert "select_task_ms" in rows[1] and "select_task_ms" not in rows[0]


def test_mode_resolution():
    assert get_vla_mode(SimpleNamespace(vla_mode=None, use_vla_cache=True)) == "V"
    assert get_vla_mode(SimpleNamespace(vla_mode=None, use_vla_cache=False)) == "B1"
    cfg = SimpleNamespace(vla_mode="V'", use_vla_cache=False)
    assert resolve_vla_mode(cfg) == "Vp" and cfg.use_vla_cache
    cfg = SimpleNamespace(vla_mode="B0", use_vla_cache=True)
    assert resolve_vla_mode(cfg) == "B0" and not cfg.use_vla_cache
    with pytest.raises(ValueError):
        get_vla_mode(SimpleNamespace(vla_mode="X", use_vla_cache=True))


def test_install_vprime_leaves_v_path_unchanged(vla):
    """V on a model without V' installed (as in real V runs) == V on the patched model used for verification."""
    frames = _frames(3)
    plain = build_tiny_openvla()
    for a, b in zip(_rollout(plain, "V", frames), _rollout(vla, "V", frames)):
        assert a["reusable"] == b["reusable"] and a["proportions"] == b["proportions"]
        assert np.array_equal(a["action"], b["action"])
