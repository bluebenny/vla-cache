"""
vla_timing.py

Per-stage timing of one VLA query (`--timing True`): CUDA events on GPU (perf_counter on CPU), plus wall-clock for
every stage. One JSONL row per `get_vla_action` call.

Stages:
    vision         vision backbone + projector
    prefill        LLM forward with q_len > 1
    decode         LLM forwards with q_len == 1 (summed; `decode_n` = number of steps)
    select*        reuse selection (select_static / select_task / select_schedule), CPU-heavy -> check `*_wall_ms`
    attn_info      V' in-module attention statistics (summed over layers; nested inside `prefill`)
    preprocess, predict_total, step_total
"""

import json
import os
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from typing import Optional

import torch


class StageTimer:
    def __init__(self, log_path: Optional[str] = None, use_cuda: Optional[bool] = None) -> None:
        self.use_cuda = torch.cuda.is_available() if use_cuda is None else use_cuda
        self.log_path = log_path
        self.context = {}  # Extra fields for every row (e.g. task_id / episode / t), set by the eval loop
        self.step = 0
        self.begin_step()

    def begin_step(self) -> None:
        self._open = {}
        self._spans = defaultdict(list)

    def _mark(self):
        event = None
        if self.use_cuda:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
        return event, time.perf_counter()

    def start(self, name: str) -> None:
        self._open[name] = self._mark()

    def stop(self, name: str) -> None:
        self._spans[name].append((self._open.pop(name), self._mark()))

    @contextmanager
    def stage(self, name: str):
        self.start(name)
        try:
            yield
        finally:
            self.stop(name)

    def end_step(self, **meta) -> dict:
        """Synchronizes once, writes one JSONL row and starts a new step."""
        if self.use_cuda:
            torch.cuda.synchronize()
        row = {"step": self.step, **self.context, **meta}
        for name, spans in self._spans.items():
            wall = sum(t1 - t0 for (_, t0), (_, t1) in spans) * 1e3
            row[f"{name}_ms"] = sum(e0.elapsed_time(e1) for (e0, _), (e1, _) in spans) if self.use_cuda else wall
            row[f"{name}_wall_ms"] = wall
            row[f"{name}_n"] = len(spans)
        if self.log_path is not None:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(row) + "\n")
        self.step += 1
        self.begin_step()
        return row


class _NullTimer:
    """Stand-in when timing is off: every call is a no-op."""

    def __init__(self) -> None:
        self.context = {}

    def begin_step(self) -> None:
        pass

    def start(self, name: str) -> None:
        pass

    def stop(self, name: str) -> None:
        pass

    def stage(self, name: str):
        return nullcontext()

    def end_step(self, **meta) -> None:
        return None


NULL_TIMER = _NullTimer()


def get_stage_timer(vla):
    return getattr(vla, "stage_timer", None) or NULL_TIMER


def _lm_stage(kwargs) -> str:
    tokens = kwargs.get("inputs_embeds")
    if tokens is None:
        tokens = kwargs.get("input_ids")
    return "prefill" if tokens.shape[1] > 1 else "decode"


def attach_stage_timer(vla, log_path: Optional[str] = None) -> StageTimer:
    """Creates a `StageTimer`, hooks the vision / LLM modules and exposes it as `vla.stage_timer`."""
    if log_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    timer = StageTimer(log_path)

    vla.vision_backbone.register_forward_pre_hook(lambda module, args: timer.start("vision"))
    vla.projector.register_forward_hook(lambda module, args, output: timer.stop("vision"))
    vla.language_model.register_forward_pre_hook(
        lambda module, args, kwargs: timer.start(_lm_stage(kwargs)), with_kwargs=True
    )
    vla.language_model.register_forward_hook(
        lambda module, args, kwargs, output: timer.stop(_lm_stage(kwargs)), with_kwargs=True
    )

    collector = getattr(vla, "vprime_collector", None)
    if collector is not None:
        collector.timer = timer
    vla.stage_timer = timer
    return timer
