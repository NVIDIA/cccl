# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
LLM demo -- single LoRA on one adapted Linear layer.

Smallest unit of composition that is recognisably the production LoRA
pattern:

    y_base  = x @ W                           (base projection)
    y_delta = (alpha / r) * (x @ A) @ B       (low-rank adapter delta)
    y       = y_base + y_delta                (combine)

No transformer block, no attention, no layernorm, no decode loop. The three
operations are wired as three STF tasks. The base and LoRA tasks share the
read-only input ``x`` and write to independent scratches, so STF schedules
them in parallel on separate streams; the combine is the fanin.

Production parallel
-------------------
This is the one adapted linear layer that every production multi-LoRA
serving stack composes: vLLM multi-LoRA, TensorRT-LLM multi-LoRA, S-LoRA,
LoRAX, HuggingFace PEFT. Scaling from single-LoRA to multi-LoRA is a
literal ``for`` loop over K adapters.

Composition story
-----------------
Each of ``stf_base_linear`` and ``stf_lora_linear`` simultaneously plays
three independent roles at three orthogonal axes of the stack:

  * ``PyTorch function``  -- the task body is ordinary PyTorch matmul math.
  * ``torch.compile(mode="default")``  -- Inductor fuses the body
    (``(x @ A) @ B`` is an ideal fusion target).
  * ``ctx.graph_scope()``  -- the task is captured as a local CUDA graph,
    replayed as a single child-graph node in STF's DAG.

STF itself owns the DAG: base and LoRA run concurrently; weights (``W``,
``A``, ``B``) are ``set_read_only()`` so they can be shared across
replays.

Toggles
-------
Both optimisations are on by default. Env-overridable toggles let the test
suite flip either axis off::

    LLM_LORA_COMPILE=0      disable torch.compile wrapping
    LLM_LORA_GRAPH_SCOPE=0  disable ctx.graph_scope() wrapping

Other knobs::

    LLM_LORA_HIDDEN=512
    LLM_LORA_SEQ=64
    LLM_LORA_RANK=8
    LLM_LORA_ALPHA=16.0

API note
--------
``ctx.graph_scope()`` is exposed on ``stf.stackable_context()``, not on the
plain ``stf.context()``. We therefore use ``stackable_context`` here, but
we do NOT push any ``while_loop`` / ``repeat`` scope -- the demo is a flat
sequence of tasks with optional per-task ``graph_scope`` wrappers, which
matches the safest configuration already exercised in
``test_stackable_graph_scope.py``.
"""

from __future__ import annotations

import os
import time
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pytorch_task import pytorch_task  # noqa: E402

import cuda.stf._experimental as stf  # noqa: E402


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoRAConfig:
    """Shape of the adapted linear layer + LoRA hyperparameters."""

    hidden: int = 512   # W is (hidden, hidden); input is (1, seq, hidden)
    seq: int = 64
    rank: int = 8
    alpha: float = 16.0
    dtype: str = "float32"

    @property
    def np_dtype(self):
        return np.float32 if self.dtype == "float32" else np.float16

    @property
    def torch_dtype(self):
        return torch.float32 if self.dtype == "float32" else torch.float16

    @property
    def alpha_over_r(self) -> float:
        return float(self.alpha) / float(self.rank)


# ---------------------------------------------------------------------------
# Module-level toggles and defaults (env-overridable)
# ---------------------------------------------------------------------------


USE_COMPILE = os.environ.get("LLM_LORA_COMPILE", "1") != "0"
USE_GRAPH_SCOPE = os.environ.get("LLM_LORA_GRAPH_SCOPE", "1") != "0"


def _default_cfg() -> LoRAConfig:
    return LoRAConfig(
        hidden=int(os.environ.get("LLM_LORA_HIDDEN", "512")),
        seq=int(os.environ.get("LLM_LORA_SEQ", "64")),
        rank=int(os.environ.get("LLM_LORA_RANK", "8")),
        alpha=float(os.environ.get("LLM_LORA_ALPHA", "16.0")),
    )


SEED = 0xC0DE


# ---------------------------------------------------------------------------
# On-device init helpers
#
# Copied-in-spirit from ``llm_helpers.py`` so this file is self-contained.
# Runs entirely on device -- no host-to-device staging for weights, and no
# reliance on torch's RNG (whose state is not graph-replay-safe).
# ---------------------------------------------------------------------------


def _init_random(ctx, ld, *, seed_idx: int, fan_in: int):
    """Fill a ``logical_data_empty`` with a deterministic pseudo-random pattern."""
    std = 1.0 / float(np.sqrt(max(1, fan_in)))
    with pytorch_task(ctx, ld.write()) as (tw,):
        idx = torch.arange(tw.numel(), device=tw.device, dtype=torch.float32)
        k = 0.0131 + 7.1e-5 * float(seed_idx)
        phi = 0.71 * float(seed_idx)
        vals = torch.sin(idx * k + phi) + 0.3 * torch.cos(idx * (3.7 * k) + 2 * phi)
        tw.copy_((vals * std).view(tw.shape).to(tw.dtype))


def _init_zeros(ctx, ld):
    with pytorch_task(ctx, ld.write()) as (tw,):
        tw.zero_()


# ---------------------------------------------------------------------------
# Weight allocation
# ---------------------------------------------------------------------------


def build_weights(ctx, cfg: LoRAConfig, *, seed: int = 0, zero_init_B: bool = False):
    """Allocate and initialise ``W``, ``A``, ``B`` as STF logical data.

    Shapes:

        W : (hidden, hidden)        base linear weight
        A : (hidden, rank)          LoRA down-projection
        B : (rank, hidden)          LoRA up-projection

    With ``zero_init_B=True``, ``B`` is filled with zeros -- the standard
    LoRA init convention: the delta ``alpha/r * (x @ A) @ B`` is then
    identically zero, so the adapted output equals the base output. Used
    by the ``zero_init_matches_base`` correctness test as a cheap shape
    guard.
    """
    dtype = cfg.np_dtype
    H, r = cfg.hidden, cfg.rank

    l_W = ctx.logical_data_empty((H, H), dtype, name="W")
    _init_random(ctx, l_W, seed_idx=seed + 1, fan_in=H)

    l_A = ctx.logical_data_empty((H, r), dtype, name="A")
    _init_random(ctx, l_A, seed_idx=seed + 2, fan_in=H)

    l_B = ctx.logical_data_empty((r, H), dtype, name="B")
    if zero_init_B:
        _init_zeros(ctx, l_B)
    else:
        _init_random(ctx, l_B, seed_idx=seed + 3, fan_in=r)

    if hasattr(l_W, "set_read_only"):
        l_W.set_read_only()
        l_A.set_read_only()
        l_B.set_read_only()

    return l_W, l_A, l_B


# ---------------------------------------------------------------------------
# Compiled bodies (module-level so torch.compile artifacts are cached once)
# ---------------------------------------------------------------------------


def _base_body(x, W):
    return torch.matmul(x, W)


def _lora_body(x, A, B, alpha_over_r: float):
    return alpha_over_r * torch.matmul(torch.matmul(x, A), B)


# ``mode="default"`` enables Inductor fusion but NOT the reduce-overhead
# CUDA-graph capture that would collide with STF's own graph_scope capture.
_base_compiled = torch.compile(_base_body, mode="default", fullgraph=True)
_lora_compiled = torch.compile(_lora_body, mode="default", fullgraph=True)


# Per-shape warmup cache. ``torch.compile`` keys on input shapes, so one
# warmup call per (H, S, r) is enough; subsequent calls hit the artifact.
_warmed_shapes: set[tuple[int, int, int, str]] = set()


def _warmup_compiled_bodies(cfg: LoRAConfig):
    """Trigger Inductor codegen OUTSIDE any STF / CUDA-graph capture.

    Dynamo probes ``torch.cuda.get_rng_state()`` on first compile. That
    call raises "Cannot call CUDAGeneratorImpl::current_seed during CUDA
    graph capture" when first-compile happens inside ``ctx.graph_scope()``
    (where capture is active). One eager call on dummy tensors with the
    right shapes populates the compile cache so all STF replays see a
    ready-made artifact.
    """
    key = (cfg.hidden, cfg.seq, cfg.rank, cfg.dtype)
    if key in _warmed_shapes:
        return

    device = torch.device("cuda")
    dtype = cfg.torch_dtype
    H, S, r = cfg.hidden, cfg.seq, cfg.rank

    x = torch.zeros((1, S, H), dtype=dtype, device=device)
    W = torch.zeros((H, H), dtype=dtype, device=device)
    A = torch.zeros((H, r), dtype=dtype, device=device)
    B = torch.zeros((r, H), dtype=dtype, device=device)

    _ = _base_compiled(x, W)
    _ = _lora_compiled(x, A, B, cfg.alpha_over_r)
    torch.cuda.synchronize()

    _warmed_shapes.add(key)


# ---------------------------------------------------------------------------
# STF-wrapped primitives
# ---------------------------------------------------------------------------


def _maybe_graph_scope(ctx, use_graph_scope: bool):
    """``ctx.graph_scope()`` when enabled, a no-op context otherwise."""
    if use_graph_scope:
        return ctx.graph_scope()
    return nullcontext()


def stf_base_linear(
    ctx, l_x, l_W, l_y_base, *, use_compile: bool, use_graph_scope: bool
):
    """One STF task: ``y_base = x @ W``.

    Optionally wrapped in ``ctx.graph_scope()`` (per-task local CUDA graph)
    and optionally dispatched through a ``torch.compile``-cached callable
    (intra-kernel fusion).
    """
    with _maybe_graph_scope(ctx, use_graph_scope):
        with pytorch_task(
            ctx, l_x.read(), l_W.read(), l_y_base.write()
        ) as (tx, tw, to):
            if use_compile:
                to[:] = _base_compiled(tx, tw)
            else:
                to[:] = torch.matmul(tx, tw)


def stf_lora_linear(
    ctx,
    l_x,
    l_A,
    l_B,
    l_y_delta,
    *,
    alpha_over_r: float,
    use_compile: bool,
    use_graph_scope: bool,
):
    """One STF task: ``y_delta = (alpha / r) * (x @ A) @ B``.

    Written as two back-to-back matmuls so Inductor can fuse them when
    ``use_compile=True``. Shapes: ``x (1,S,H) @ A (H,r) -> (1,S,r)``,
    then ``@ B (r,H) -> (1,S,H)``.
    """
    with _maybe_graph_scope(ctx, use_graph_scope):
        with pytorch_task(
            ctx, l_x.read(), l_A.read(), l_B.read(), l_y_delta.write()
        ) as (tx, ta, tb, to):
            if use_compile:
                to[:] = _lora_compiled(tx, ta, tb, alpha_over_r)
            else:
                to[:] = alpha_over_r * torch.matmul(torch.matmul(tx, ta), tb)


def stf_combine_add(ctx, l_y_base, l_y_delta, l_y):
    """One STF task: ``y = y_base + y_delta``.

    Intentionally not graph-scoped or compiled -- a single elementwise add
    is not worth the overhead of either wrapper.
    """
    with pytorch_task(
        ctx, l_y_base.read(), l_y_delta.read(), l_y.write()
    ) as (tyb, tyd, to):
        to[:] = tyb + tyd


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_lora_forward(
    cfg: LoRAConfig | None = None,
    *,
    use_compile: bool = True,
    use_graph_scope: bool = True,
    zero_init_B: bool = False,
    include_lora: bool = True,
    seed: int = 0,
):
    """Run one forward pass and return ``(y_host, elapsed_seconds)``.

    When ``include_lora=False``, emits only ``stf_base_linear`` -- the LoRA
    path and the combine task are not wired into the DAG. Used by the
    ``zero_init_matches_base`` correctness test to produce the reference
    output against which the zero-init LoRA output must match exactly.
    """
    if cfg is None:
        cfg = _default_cfg()

    # Pre-compile OUTSIDE any STF graph capture -- see _warmup_compiled_bodies.
    if use_compile:
        _warmup_compiled_bodies(cfg)

    H, S = cfg.hidden, cfg.seq
    x_host = np.random.default_rng(seed + 1).standard_normal(
        (1, S, H)
    ).astype(cfg.np_dtype)
    y_host = np.zeros((1, S, H), dtype=cfg.np_dtype)

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    # Stackable context without any while_loop / repeat / nested scope:
    # graph_scope lives on stackable_context in the bindings, but we do not
    # push a conditional or loop scope -- equivalent in shape to a plain
    # stf.context() forward pass plus optional per-task local CUDA graphs.
    ctx = stf.stackable_context()

    l_x = ctx.logical_data(x_host, name="x")
    if hasattr(l_x, "set_read_only"):
        l_x.set_read_only()
    l_y = ctx.logical_data(y_host, name="y")

    l_W, l_A, l_B = build_weights(ctx, cfg, seed=seed, zero_init_B=zero_init_B)

    if include_lora:
        l_y_base = ctx.logical_data_empty((1, S, H), cfg.np_dtype, name="y_base")
        l_y_delta = ctx.logical_data_empty((1, S, H), cfg.np_dtype, name="y_delta")

        stf_base_linear(
            ctx, l_x, l_W, l_y_base,
            use_compile=use_compile, use_graph_scope=use_graph_scope,
        )
        stf_lora_linear(
            ctx, l_x, l_A, l_B, l_y_delta,
            alpha_over_r=cfg.alpha_over_r,
            use_compile=use_compile, use_graph_scope=use_graph_scope,
        )
        stf_combine_add(ctx, l_y_base, l_y_delta, l_y)
    else:
        # No LoRA wiring: base task writes directly into the output buffer.
        stf_base_linear(
            ctx, l_x, l_W, l_y,
            use_compile=use_compile, use_graph_scope=use_graph_scope,
        )

    ctx.finalize()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    return y_host, elapsed


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_lora_adapted_linear_headline():
    """Slide-ready run: compile + graph_scope both on.

    Exercises the composition story -- ``stf_base_linear`` and
    ``stf_lora_linear`` run as sibling STF tasks, each wrapped in its own
    ``ctx.graph_scope()`` and with its body dispatched through a
    ``torch.compile(mode="default")``-cached callable.
    """
    cfg = _default_cfg()

    y, elapsed = run_lora_forward(
        cfg, use_compile=True, use_graph_scope=True, seed=0,
    )

    assert y.shape == (1, cfg.seq, cfg.hidden), f"shape mismatch: {y.shape}"
    assert np.isfinite(y).all(), "output contains non-finite values"

    print("\n=== LLM demo - single LoRA (STF + torch.compile + graph_scope) ===")
    print(
        f"Config: hidden={cfg.hidden}, seq={cfg.seq}, rank={cfg.rank}, "
        f"alpha={cfg.alpha}, dtype={cfg.dtype}"
    )
    print(f"Forward wall time: {elapsed * 1e3:.2f} ms")
    print(
        f"y stats: mean={y.mean():.4f}, std={y.std():.4f}, "
        f"min={y.min():.4f}, max={y.max():.4f}"
    )
    dot = os.environ.get("CUDASTF_DOT_FILE", "(not set; set env to dump DAG)")
    print(f"CUDASTF_DOT_FILE: {dot}")


def test_lora_zero_init_matches_base():
    """With ``B`` zero-initialised, the LoRA delta is provably zero.

    This means the adapted output must equal the base-only output to
    within floating-point tolerance. Catches shape / transpose bugs in
    ``stf_lora_linear`` cheaply: a mis-wired LoRA path would emit non-zero
    garbage instead of zero even with ``B = 0``.

    Uses the same ``seed`` in both runs so ``W`` is bitwise identical and
    the comparison is meaningful.
    """
    cfg = _default_cfg()

    # LoRA wired in, but B = 0 => delta identically 0 => y = y_base.
    y_lora, _ = run_lora_forward(
        cfg, use_compile=True, use_graph_scope=True,
        zero_init_B=True, include_lora=True, seed=42,
    )

    # Base only (no LoRA wiring at all).
    y_base, _ = run_lora_forward(
        cfg, use_compile=True, use_graph_scope=True,
        zero_init_B=True, include_lora=False, seed=42,
    )

    np.testing.assert_allclose(
        y_lora, y_base, atol=1e-5, rtol=1e-5,
        err_msg="LoRA zero-init output does not match base-only output",
    )


if __name__ == "__main__":
    test_lora_adapted_linear_headline()
    test_lora_zero_init_matches_base()
    print("\nAll LoRA demo tests passed.")
