# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Multi-LoRA baseline sweep: ``py/seq/eager`` vs ``py/seq/compile`` vs
``py/stream/compile`` vs ``stf/compile+graph_scope``.

Purpose
-------
Give a slide-grade answer to "what does STF actually buy us over plain
PyTorch, and at what coding cost?" for a realistic multi-LoRA serving
pattern::

    y_base    = x @ W                                 (shared, ONCE)
    y_delta_k = (alpha / r) * (x @ A_k) @ B_k         (K siblings, parallel)
    y_k       = y_base + y_delta_k                    (K combines, parallel)

Four modes are measured per K:

  ``py/seq/eager``      plain PyTorch, single default stream, no compile.
                        What a user writes if they just transcribe the math.
  ``py/seq/compile``    same, each body wrapped with
                        ``torch.compile(mode="default")``. Inductor fuses
                        ``(x @ A_k) @ B_k`` but the K adapters still
                        serialise on the default stream.
  ``py/stream/compile`` PyTorch with K manually-managed ``torch.cuda.Stream``s
                        and events, with compile. The closest plain-PyTorch
                        equivalent to STF's K-way concurrency. Requires
                        ~14 lines of stream / event boilerplate per forward
                        (see ``_build_py_multistream_forward``).
  ``stf/compile+gs``    STF with ``torch.compile`` + ``ctx.graph_scope`` per
                        adapter. Concurrency is expressed by ``.read()`` /
                        ``.write()`` deps plus one per-scope explicit
                        ``l_y_base.push(AccessMode.READ)``; zero LOC of
                        stream / event code.

Config
------
The default matches an LLM-serving-scale adapted Linear::

    hidden=4096, seq=512, rank=16, dtype=float32

At this size, one matmul is ~O(GFLOP), so launch overhead does not
dominate and concurrency has something to win. For a quick local sanity
check, ``LLM_LORA_BENCH_SIZE=tiny`` reverts to ``hidden=512, seq=64``.

Timing methodology
------------------
All four modes build their weights / contexts ONCE outside the timing
loop. Each timed iteration runs only the forward pass. STF uses a
persistent ``stackable_context`` + ``ctx.fence()`` per iteration, which
matches the real-world "decode loop on a long-lived context" pattern.

Env knobs
---------
``LLM_LORA_SWEEP_K=1,2,4,8,16``     comma-separated K values.
``LLM_LORA_SWEEP_ITERS=20``         timed iterations per cell.
``LLM_LORA_SWEEP_WARMUP=5``         warmup iterations per cell.
``LLM_LORA_BENCH_SIZE=realistic``   ``realistic`` (default) or ``tiny``.
"""

from __future__ import annotations

import os
import time
from contextlib import nullcontext
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pytorch_task import pytorch_task  # noqa: E402

import cuda.stf._experimental as stf  # noqa: E402

from test_llm_lora import (  # noqa: E402
    LoRAConfig,
    _base_compiled,
    _init_random,
    _lora_compiled,
    _maybe_graph_scope,
    _warmup_compiled_bodies,
)
from test_llm_multi_lora import build_multi_lora_weights  # noqa: E402


# ---------------------------------------------------------------------------
# Config selection
# ---------------------------------------------------------------------------


def _bench_cfg() -> LoRAConfig:
    """Return the benchmark config (realistic by default, ``tiny`` override)."""
    size = os.environ.get("LLM_LORA_BENCH_SIZE", "realistic")
    if size == "tiny":
        return LoRAConfig(hidden=512, seq=64, rank=8, alpha=16.0)
    return LoRAConfig(hidden=4096, seq=512, rank=16, alpha=32.0)


# ---------------------------------------------------------------------------
# Input / weight generation (shared by all PyTorch-side baselines).
# ---------------------------------------------------------------------------


def _gen_tensors(cfg: LoRAConfig, K: int, *, seed: int = 0, device: str = "cuda"):
    """Create ``x``, ``W``, ``[A_k]``, ``[B_k]`` as CUDA torch tensors."""
    g = torch.Generator(device=device).manual_seed(seed)
    H, S, r = cfg.hidden, cfg.seq, cfg.rank
    dt = cfg.torch_dtype

    scale_H = 1.0 / (H ** 0.5)
    scale_r = 1.0 / (r ** 0.5)

    x = torch.randn((1, S, H), generator=g, device=device, dtype=dt)
    W = torch.randn((H, H), generator=g, device=device, dtype=dt) * scale_H
    As = [
        torch.randn((H, r), generator=g, device=device, dtype=dt) * scale_H
        for _ in range(K)
    ]
    Bs = [
        torch.randn((r, H), generator=g, device=device, dtype=dt) * scale_r
        for _ in range(K)
    ]
    return x, W, As, Bs


# ---------------------------------------------------------------------------
# Baseline 1: plain PyTorch, single default stream, (+/- torch.compile).
#
# The "default code a user writes". No streams, no events.
# ---------------------------------------------------------------------------


def _build_py_sequential_forward(cfg: LoRAConfig, K: int, *, use_compile: bool):
    """Return a callable ``forward(x, W, As, Bs) -> list[Tensor]``."""
    alpha = cfg.alpha_over_r

    def _base_body(x, W):
        return x @ W

    def _lora_body(x, A, B):
        return alpha * ((x @ A) @ B)

    if use_compile:
        _base_body = torch.compile(_base_body, mode="default", fullgraph=True)
        _lora_body = torch.compile(_lora_body, mode="default", fullgraph=True)

    def forward(x, W, As, Bs):
        y_base = _base_body(x, W)
        y_list: list[Any] = [None] * K
        for k in range(K):
            y_list[k] = y_base + _lora_body(x, As[k], Bs[k])
        return y_list

    return forward


# ---------------------------------------------------------------------------
# Baseline 2: PyTorch + K streams + events + torch.compile.
#
# The closest hand-written equivalent to STF's K-way concurrency. Each
# adapter path runs on its own CUDA stream; the cross-stream dependency
# on ``y_base`` is expressed with an explicit event.
#
# Everything between the "concurrency boilerplate" banners counts as the
# effort needed to obtain the same concurrency STF gets from its DAG. In
# STF the same concurrency is implicit in the ``.read()`` / ``.write()``
# deps plus one per-scope ``l_y_base.push(AccessMode.READ)``.
# ---------------------------------------------------------------------------


def _build_py_multistream_forward(cfg: LoRAConfig, K: int, *, use_compile: bool):
    """Return ``(forward, streams)``. Streams are created once, reused per forward."""
    alpha = cfg.alpha_over_r

    def _base_body(x, W):
        return x @ W

    def _lora_body(x, A, B):
        return alpha * ((x @ A) @ B)

    def _combine(b, d):
        return b + d

    if use_compile:
        _base_body = torch.compile(_base_body, mode="default", fullgraph=True)
        _lora_body = torch.compile(_lora_body, mode="default", fullgraph=True)
        _combine = torch.compile(_combine, mode="default", fullgraph=True)

    # ===== concurrency boilerplate starts =====
    streams = [torch.cuda.Stream() for _ in range(K)]

    def forward(x, W, As, Bs):
        default = torch.cuda.current_stream()
        y_base = _base_body(x, W)
        base_event = torch.cuda.Event()
        base_event.record(default)

        y_list: list[Any] = [None] * K
        done_events: list[torch.cuda.Event] = []
        for k in range(K):
            with torch.cuda.stream(streams[k]):
                streams[k].wait_event(base_event)
                y_delta_k = _lora_body(x, As[k], Bs[k])
                y_list[k] = _combine(y_base, y_delta_k)
                ev = torch.cuda.Event()
                ev.record(streams[k])
                done_events.append(ev)

        for ev in done_events:
            default.wait_event(ev)
        return y_list
    # ===== concurrency boilerplate ends =====

    return forward, streams


# ---------------------------------------------------------------------------
# STF persistent-context builder. Builds ctx + logical data ONCE, returns a
# forward closure that submits K+1 graph scopes per call.
# ---------------------------------------------------------------------------


def _build_stf_persistent_forward(
    cfg: LoRAConfig, K: int, *, seed: int = 0, use_graph_scope: bool = True,
):
    """Return ``(forward, ctx)``. ``forward()`` submits one multi-LoRA pass.

    The STF context and all K+2 weight tensors (x, W, K*(A_k, B_k)) are
    allocated and initialised once by this builder -- matches the real
    deployment pattern where weights live on device for the duration of
    a serving session. The ``forward`` closure does only task submission
    (no allocation, no init), and is therefore an apples-to-apples match
    for the pre-built PyTorch baselines.

    With ``use_graph_scope=False`` each task is submitted as a plain
    ``ctx.task(...)`` instead of being wrapped in ``ctx.graph_scope()``
    + capture; useful for isolating the host-side graph_scope cost from
    the pure STF scheduling overhead.
    """
    _warmup_compiled_bodies(cfg)

    ctx = stf.stackable_context()

    H, S = cfg.hidden, cfg.seq
    rng = np.random.default_rng(seed + 1)
    x_host = rng.standard_normal((1, S, H)).astype(cfg.np_dtype)

    l_x = ctx.logical_data(x_host, name="x")
    if hasattr(l_x, "set_read_only"):
        l_x.set_read_only()

    l_W, adapters = build_multi_lora_weights(ctx, cfg, K, seed=seed)

    # Device-only intermediates / outputs. ``logical_data_empty`` avoids
    # binding a host numpy array that STF would have to stage back on
    # finalize; nothing in the benchmark path reads these values from
    # host, so a pure on-device buffer is the right thing.
    l_y_base = ctx.logical_data_empty((1, S, H), cfg.np_dtype, name="y_base")
    l_y_deltas = [
        ctx.logical_data_empty((1, S, H), cfg.np_dtype, name=f"y_delta_{k}")
        for k in range(K)
    ]
    l_y_list = [
        ctx.logical_data_empty((1, S, H), cfg.np_dtype, name=f"y_{k}")
        for k in range(K)
    ]

    alpha = cfg.alpha_over_r

    def _scope():
        return ctx.graph_scope() if use_graph_scope else nullcontext()

    def forward():
        """One multi-LoRA forward. Submits tasks; does not sync."""
        with _scope():
            with pytorch_task(ctx, l_x.read(), l_W.read(), l_y_base.write()) as (
                tx, tw, tob,
            ):
                tob[:] = _base_compiled(tx, tw)

        for k, (l_A_k, l_B_k) in enumerate(adapters):
            l_y_delta_k = l_y_deltas[k]
            l_y_k = l_y_list[k]
            with _scope():
                if use_graph_scope:
                    l_y_base.push(stf.AccessMode.READ)
                with pytorch_task(
                    ctx, l_x.read(), l_A_k.read(), l_B_k.read(), l_y_delta_k.write(),
                ) as (tx, ta, tb, tod):
                    tod[:] = _lora_compiled(tx, ta, tb, alpha)
                with pytorch_task(
                    ctx, l_y_base.read(), l_y_delta_k.read(), l_y_k.write(),
                ) as (tyb, tyd, to):
                    to[:] = tyb + tyd

    return forward, ctx


# ---------------------------------------------------------------------------
# Timing harness
# ---------------------------------------------------------------------------


def _time_pytorch(forward, inputs, *, iters: int, warmup: int) -> float:
    x, W, As, Bs = inputs

    for _ in range(warmup):
        _ = forward(x, W, As, Bs)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = forward(x, W, As, Bs)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def _time_stf(forward_callable, *, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        forward_callable()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        forward_callable()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


_MODES = (
    "py/seq/eager",
    "py/seq/compile",
    "py/stream/compile",
    "stf/compile",
    "stf/compile+gs",
)


def run_sweep(cfg: LoRAConfig, Ks: tuple[int, ...], *, iters: int, warmup: int):
    """Return list of dicts {K, mode: seconds}."""
    rows = []
    for K in Ks:
        row: dict[str, Any] = {"K": K}

        inputs = _gen_tensors(cfg, K, seed=0)

        fwd_eager = _build_py_sequential_forward(cfg, K, use_compile=False)
        row["py/seq/eager"] = _time_pytorch(
            fwd_eager, inputs, iters=iters, warmup=warmup
        )

        fwd_seq_c = _build_py_sequential_forward(cfg, K, use_compile=True)
        row["py/seq/compile"] = _time_pytorch(
            fwd_seq_c, inputs, iters=iters, warmup=warmup
        )

        fwd_ms_c, _streams = _build_py_multistream_forward(cfg, K, use_compile=True)
        row["py/stream/compile"] = _time_pytorch(
            fwd_ms_c, inputs, iters=iters, warmup=warmup
        )

        # STF persistent, plain tasks (no graph_scope). Isolates pure STF
        # scheduling overhead from the per-scope graph capture cost.
        stf_forward, stf_ctx = _build_stf_persistent_forward(
            cfg, K, seed=0, use_graph_scope=False,
        )
        try:
            row["stf/compile"] = _time_stf(
                stf_forward, iters=iters, warmup=warmup,
            )
        finally:
            stf_ctx.finalize()

        # STF persistent, with graph_scope per base + K adapters.
        stf_forward, stf_ctx = _build_stf_persistent_forward(
            cfg, K, seed=0, use_graph_scope=True,
        )
        try:
            row["stf/compile+gs"] = _time_stf(
                stf_forward, iters=iters, warmup=warmup,
            )
        finally:
            stf_ctx.finalize()

        rows.append(row)
    return rows


def _fmt_table(rows, cfg: LoRAConfig) -> str:
    lines = []
    lines.append(
        f"Config: hidden={cfg.hidden}, seq={cfg.seq}, rank={cfg.rank}, "
        f"alpha={cfg.alpha}, dtype={cfg.dtype}"
    )
    header = f"{'K':>4} " + " ".join(f"{m:>20}" for m in _MODES)
    lines.append(header)
    lines.append("-" * len(header))
    for r in rows:
        cells = [f"{r[m] * 1e3:>17.2f} ms" for m in _MODES]
        lines.append(f"{r['K']:>4d} " + " ".join(cells))
    return "\n".join(lines)


def _fmt_speedup_table(rows) -> str:
    lines = []
    header = f"{'K':>4} " + " ".join(f"{m:>20}" for m in _MODES)
    lines.append(header)
    lines.append("-" * len(header))
    for r in rows:
        base = r["py/seq/eager"]
        cells = [f"{base / r[m]:>17.2f}x " for m in _MODES]
        lines.append(f"{r['K']:>4d} " + " ".join(cells))
    return "\n".join(lines)


def _effort_summary() -> str:
    return (
        "Effort to obtain the K-way concurrency (LOC of scheduling code):\n"
        "  py/seq/eager        0  (no concurrency; K adapters serialise)\n"
        "  py/seq/compile      0  (no concurrency; K adapters serialise)\n"
        "  py/stream/compile  ~14 per forward: K streams + K events +\n"
        "                      wait_event / record / cross-stream sync\n"
        "                      (see _build_py_multistream_forward body).\n"
        "  stf/compile+gs      1  l_y_base.push(AccessMode.READ) per scope,\n"
        "                      inside the existing for-k loop. Scheduling,\n"
        "                      streams, and events are implicit in the DAG."
    )


# ---------------------------------------------------------------------------
# Pytest entry: runs the sweep, prints two tables + the effort summary.
# ---------------------------------------------------------------------------


def test_multi_lora_baseline_sweep():
    """Headline sweep for the slide. Prints two tables and an effort summary."""
    cfg = _bench_cfg()
    Ks = tuple(
        int(k) for k in os.environ.get("LLM_LORA_SWEEP_K", "1,2,4,8,16").split(",")
    )
    iters = int(os.environ.get("LLM_LORA_SWEEP_ITERS", "20"))
    warmup = int(os.environ.get("LLM_LORA_SWEEP_WARMUP", "5"))

    rows = run_sweep(cfg, Ks, iters=iters, warmup=warmup)

    print("\n=== Multi-LoRA baseline sweep -- wall-clock ms / forward ===")
    print(_fmt_table(rows, cfg))

    print("\n=== Same data, normalised (speedup vs. py/seq/eager) ===")
    print(_fmt_speedup_table(rows))

    print("\n" + _effort_summary())

    for r in rows:
        for m in _MODES:
            assert r[m] > 0.0, f"K={r['K']} mode={m}: non-positive time"


if __name__ == "__main__":
    test_multi_lora_baseline_sweep()
