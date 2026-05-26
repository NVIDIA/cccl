# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
LLM demo -- multi-LoRA on one shared adapted Linear layer.

Extension of ``test_llm_lora.py`` from one adapter to ``K`` adapters
sharing a single base projection. Matches the production multi-LoRA
serving pattern (vLLM multi-LoRA, S-LoRA, LoRAX, TensorRT-LLM): one
frozen base model, many lightweight adapters served concurrently::

    y_base   = x @ W                                    (shared, ONCE)
    y_delta_k = (alpha / r) * (x @ A_k) @ B_k           (K siblings, in parallel)
    y_k      = y_base + y_delta_k                       (K combines)

Why this demo is stronger than single-LoRA
------------------------------------------
- ``W`` is computed ONCE and read by K combine tasks. ``set_read_only()``
  on ``W``, ``A_k``, ``B_k`` lets STF schedule the K adapter paths with
  no spurious serialisation.
- K LoRA tasks are siblings in the DAG -- each already wrapped in its
  own ``ctx.graph_scope()`` from the shared primitives -- so STF has a
  real K-way scheduling decision instead of a 2-way one.
- The same ``torch.compile`` artifact is re-used across all K adapters
  (identical shapes), so Inductor codegen amortises over K call sites.
- DOT output: ONE base cluster fanning out to K sibling LoRA clusters,
  fanning in to K combines. Wide, slide-ready DAG.

Composition axes reused from ``test_llm_lora.py``
-------------------------------------------------
The three task primitives (``stf_base_linear``, ``stf_lora_linear``,
``stf_combine_add``) are imported as-is. Each LoRA task is still
simultaneously a PyTorch function, a ``torch.compile`` artifact, a
``ctx.graph_scope()`` local CUDA graph, and an STF DAG node -- just now
K of them instead of one.

Env knobs
---------
``LLM_MULTI_LORA_K=8``    number of adapters
plus all ``LLM_LORA_*`` knobs from ``test_llm_lora.py``.

Scope
-----
No loop / decode wrapper (that belongs on top of a multi-layer
transformer + KV cache, not on a single linear). No multi-device
placement yet -- that's the natural next step via
``exec_place.device(k)`` on each ``stf_lora_linear`` call.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from test_llm_lora import (  # noqa: E402
    LoRAConfig,
    _default_cfg,
    _init_random,
    _init_zeros,
    _maybe_graph_scope,
    _warmup_compiled_bodies,
    run_lora_forward,
    stf_base_linear,
    stf_combine_add,
    stf_lora_linear,
)

import cuda.stf._experimental as stf  # noqa: E402

# ---------------------------------------------------------------------------
# Weight allocation -- shared W plus K (A_k, B_k) adapter pairs.
# ---------------------------------------------------------------------------


def build_multi_lora_weights(
    ctx,
    cfg: LoRAConfig,
    K: int,
    *,
    seed: int = 0,
    zero_init_B: bool = False,
):
    """Allocate ``W`` and ``K`` adapter pairs ``(A_k, B_k)``.

    All K adapters share the same shapes (``rank``, ``hidden``) so the
    downstream ``torch.compile`` artifact is a single cache entry.

    With ``zero_init_B=True``, all ``B_k`` are filled with zeros -- the
    LoRA delta for every adapter is then identically zero and each
    ``y_k`` must equal the shared base output.
    """
    dtype = cfg.np_dtype
    H, r = cfg.hidden, cfg.rank

    l_W = ctx.logical_data_empty((H, H), dtype, name="W")
    _init_random(ctx, l_W, seed_idx=seed + 1, fan_in=H)
    if hasattr(l_W, "set_read_only"):
        l_W.set_read_only()

    adapters = []
    for k in range(K):
        l_A_k = ctx.logical_data_empty((H, r), dtype, name=f"A_{k}")
        _init_random(ctx, l_A_k, seed_idx=seed + 100 + 2 * k, fan_in=H)

        l_B_k = ctx.logical_data_empty((r, H), dtype, name=f"B_{k}")
        if zero_init_B:
            _init_zeros(ctx, l_B_k)
        else:
            _init_random(ctx, l_B_k, seed_idx=seed + 101 + 2 * k, fan_in=r)

        if hasattr(l_A_k, "set_read_only"):
            l_A_k.set_read_only()
            l_B_k.set_read_only()

        adapters.append((l_A_k, l_B_k))

    return l_W, adapters


# ---------------------------------------------------------------------------
# Driver -- shared x, K outputs.
# ---------------------------------------------------------------------------


def run_multi_lora_forward(
    cfg: LoRAConfig | None = None,
    K: int = 8,
    *,
    use_compile: bool = True,
    use_graph_scope: bool = True,
    zero_init_B: bool = False,
    seed: int = 0,
):
    """Run one forward pass with ``K`` LoRA adapters sharing base ``W``.

    DAG shape (K + 1 graph_scope regions in total)::

        graph_scope(base):
            stf_base_linear                 # reads x, W; writes y_base

        graph_scope(adapter_0):
            y_base.push(AccessMode.READ)    # explicit per-scope read-only import
            stf_lora_linear                 # reads x, A_0, B_0; writes y_delta_0
            stf_combine_add                 # reads y_base (R/O), y_delta_0; writes y_0
        ...
        graph_scope(adapter_{K-1}):
            y_base.push(AccessMode.READ)
            stf_lora_linear                 # reads x, A_{K-1}, B_{K-1}; writes y_delta_{K-1}
            stf_combine_add                 # reads y_base (R/O), y_delta_{K-1}; writes y_{K-1}

    Each adapter scope is one atomic "apply-adapter-k" replay region (LoRA
    delta + combine fused into a single captured CUDA graph). The K scopes
    have no output conflicts (each writes its own y_delta_k / y_k) and share
    only read-only inputs (x, A_k, B_k, y_base), so STF places them on K
    concurrent streams. A single shared ``torch.compile`` artifact serves
    every ``stf_lora_linear`` call.

    Returns ``(y_list, elapsed_seconds)`` where ``y_list`` is a list of K
    numpy arrays of shape ``(1, seq, hidden)``.
    """
    if cfg is None:
        cfg = _default_cfg()

    if use_compile:
        _warmup_compiled_bodies(cfg)

    H, S = cfg.hidden, cfg.seq
    x_host = (
        np.random.default_rng(seed + 1).standard_normal((1, S, H)).astype(cfg.np_dtype)
    )
    y_hosts = [np.zeros((1, S, H), dtype=cfg.np_dtype) for _ in range(K)]

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    ctx = stf.stackable_context()

    l_x = ctx.logical_data(x_host, name="x")
    if hasattr(l_x, "set_read_only"):
        l_x.set_read_only()

    l_y_list = [ctx.logical_data(y_hosts[k], name=f"y_{k}") for k in range(K)]

    l_W, adapters = build_multi_lora_weights(
        ctx,
        cfg,
        K,
        seed=seed,
        zero_init_B=zero_init_B,
    )

    # Shared base: computed ONCE, read by K combine tasks. This is
    # exactly the vLLM multi-LoRA optimisation: the base projection is
    # batch-shared; only the rank-r LoRA deltas are per-request.
    l_y_base = ctx.logical_data_empty((1, S, H), cfg.np_dtype, name="y_base")
    stf_base_linear(
        ctx,
        l_x,
        l_W,
        l_y_base,
        use_compile=use_compile,
        use_graph_scope=use_graph_scope,
    )

    # One graph_scope per adapter, grouping BOTH the LoRA task and its
    # combine into a single replay region. From STF's perspective each
    # scope is then an atomic "apply-adapter-k" unit; the K scopes have
    # no data conflicts on outputs (each writes its own y_delta_k / y_k)
    # and share only read-only inputs (x, A_k, B_k, y_base), so they run
    # in parallel on K streams.
    #
    # Critical for the K-way fanout: ``y_base`` is written by the base
    # scope above, then only READ by the K combine tasks below. Without
    # intervention STF imports ``y_base`` conservatively (as RW) the
    # first time a child scope touches it, which would serialise the K
    # adapter scopes even though they never actually conflict. We tell
    # STF exactly what mode we need inside each scope by calling
    # ``l_y_base.push(AccessMode.READ)`` at the top of the scope. This
    # is the per-scope equivalent of ``set_read_only()`` and, unlike
    # the sticky flag, leaves ``y_base`` freely writable at the parent
    # level (so e.g. a future decode loop could refresh it each step).
    for k, (l_A_k, l_B_k) in enumerate(adapters):
        l_y_delta_k = ctx.logical_data_empty(
            (1, S, H),
            cfg.np_dtype,
            name=f"y_delta_{k}",
        )
        with _maybe_graph_scope(ctx, use_graph_scope):
            if use_graph_scope and hasattr(l_y_base, "push"):
                l_y_base.push(stf.AccessMode.READ)
            # Inner primitives must NOT open their own graph_scope here
            # (that would nest two scopes and collapse the intended shape).
            stf_lora_linear(
                ctx,
                l_x,
                l_A_k,
                l_B_k,
                l_y_delta_k,
                alpha_over_r=cfg.alpha_over_r,
                use_compile=use_compile,
                use_graph_scope=False,
            )
            stf_combine_add(ctx, l_y_base, l_y_delta_k, l_y_list[k])

    ctx.finalize()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    return y_hosts, elapsed


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _env_K(default: int = 8) -> int:
    return int(os.environ.get("LLM_MULTI_LORA_K", str(default)))


def test_multi_lora_headline():
    """K-adapter multi-LoRA forward with compile + graph_scope on.

    Prints wall time for the full K-way fanout. For a slide-grade
    number, run alongside ``test_llm_lora.py::test_lora_adapted_linear_headline``
    (K=1) and compare: ratio ``t_K / t_1`` well below ``K`` means STF is
    actually overlapping the K LoRA paths rather than serialising them.
    """
    cfg = _default_cfg()
    K = _env_K(8)

    y_list, elapsed = run_multi_lora_forward(
        cfg,
        K=K,
        use_compile=True,
        use_graph_scope=True,
        seed=0,
    )

    assert len(y_list) == K
    for k, y in enumerate(y_list):
        assert y.shape == (1, cfg.seq, cfg.hidden), (
            f"adapter {k}: shape mismatch {y.shape}"
        )
        assert np.isfinite(y).all(), f"adapter {k}: non-finite output"

    # Multi-LoRA outputs should NOT all be identical (each adapter has a
    # different (A_k, B_k), so the deltas differ). Cheap sanity check.
    if K >= 2:
        diff = np.abs(y_list[0] - y_list[1]).max()
        assert diff > 1e-6, (
            "adapters 0 and 1 produced identical output; "
            "likely a wiring bug (same A/B or wrong indexing)"
        )

    print("\n=== LLM demo - multi-LoRA (K shared-base adapters) ===")
    print(
        f"Config: hidden={cfg.hidden}, seq={cfg.seq}, rank={cfg.rank}, "
        f"alpha={cfg.alpha}, dtype={cfg.dtype}, K={K}"
    )
    print(f"Forward wall time: {elapsed * 1e3:.2f} ms total")
    print(f"Per-adapter amortised: {elapsed * 1e3 / K:.2f} ms")
    y0 = y_list[0]
    print(
        f"y_0 stats: mean={y0.mean():.4f}, std={y0.std():.4f}, "
        f"min={y0.min():.4f}, max={y0.max():.4f}"
    )
    dot = os.environ.get("CUDASTF_DOT_FILE", "(not set; set env to dump DAG)")
    print(f"CUDASTF_DOT_FILE: {dot}")


def test_multi_lora_zero_init_all_match_base():
    """With every ``B_k = 0``, every ``y_k`` must equal the shared base.

    Doubles as a correctness guard for the K-fanout wiring: a swapped
    adapter index (e.g. always reading ``A_0``) or a mis-transposed delta
    would produce non-zero garbage that this test would catch without
    relying on output numerics.

    Reference is computed with ``run_lora_forward(include_lora=False)``
    from ``test_llm_lora.py``, seeded identically so ``W`` is bitwise
    identical across the two runs.
    """
    cfg = _default_cfg()
    K = 4  # small to keep the test fast

    y_list, _ = run_multi_lora_forward(
        cfg,
        K=K,
        use_compile=True,
        use_graph_scope=True,
        zero_init_B=True,
        seed=42,
    )

    # Reference base-only forward from the single-LoRA file.
    y_base, _ = run_lora_forward(
        cfg,
        use_compile=True,
        use_graph_scope=True,
        zero_init_B=True,
        include_lora=False,
        seed=42,
    )

    for k, y in enumerate(y_list):
        np.testing.assert_allclose(
            y,
            y_base,
            atol=1e-5,
            rtol=1e-5,
            err_msg=(
                f"adapter {k} zero-init output does not match base-only "
                f"(likely a wiring bug in the K-fanout)"
            ),
        )


if __name__ == "__main__":
    test_multi_lora_headline()
    test_multi_lora_zero_init_all_match_base()
    print("\nAll multi-LoRA demo tests passed.")
