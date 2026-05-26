# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
LLM demo — Layers B + C merged: same decode-step code, three deployment modes.

All three tests below call the *same* ``decode_step`` function (defined once
in this file). They differ only in the context they run on top of:

  1. ``test_decode_eager``           — ``stf.context()``               (host loop)
  2. ``test_decode_graph``           — ``stf.context(use_graph=True)`` (whole-
                                       program graph, host-unrolled)
  3. ``test_decode_stackable_while`` — ``stf.stackable_context()`` +
                                       ``ctx.while_loop()`` (conditional graph)

Presentation role: the slide "same code, three deployment modes" and the
slide "generation as a conditional graph" (Layer B).

Perf numbers are **not** the story here. Submit/exec timings are printed as
a sanity check; the headline claim is that all three modes produce the same
greedy token with identical weights & seed — i.e. the same code runs under
three different STF execution strategies without any per-mode rewrites.

Requires CUDA 12.4+ for ``test_decode_stackable_while`` (conditional graph
nodes). The other two tests work on older drivers.

Known issue
-----------
``stf.context(use_graph=True)`` exposes host pointers to PyTorch via
``__cuda_array_interface__`` at graph-capture time, which PyTorch's strict
CAI validation rejects ("pointer resides on host memory"). Until that is
resolved in the binding, ``test_decode_graph`` is skipped. The
``stackable_context + while_loop`` test below exercises the full graph
capture path.

Env knobs
---------
``LLM_MAX_TOKENS=16``   Number of decode steps per mode (default 8 here for
                        pytest speed; set larger when running the slide demo).
"""

import os
import time

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from llm_helpers import (  # noqa: E402
    TINY,
    build_random_weights,
    make_cond_scratch,
    stf_advance_counter_flag,
    stf_lm_head,
    stf_sample_argmax_last,
    stf_append_token_hidden,
    stf_transformer_stack,
    validate_forward,
)
from pytorch_task import pytorch_task  # noqa: E402

import cuda.stf._experimental as stf  # noqa: E402


MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "8"))
SEED = 0xC0DE


# ---------------------------------------------------------------------------
# The shared per-step body — identical across all three modes.
# ---------------------------------------------------------------------------


def decode_step(ctx, cfg, weights, l_hidden, l_logits, l_next):
    """One autoregressive step: transformer stack → lm_head → argmax → feed back.

    Writes the stack output into a fresh per-step ``h_next`` then copies
    back into ``l_hidden``. Every step allocates its own intermediates;
    this keeps STF's dep tracking simple at the cost of a graph whose
    node count grows with the number of unrolled steps (fine for the
    eager host-loop; for the stackable + while_loop mode the same body
    is captured once).
    """
    B, S, H = 1, cfg.seq, cfg.hidden
    l_hn = ctx.logical_data_empty((B, S, H), cfg.np_dtype, name="h_next")
    stf_transformer_stack(ctx, l_hidden, weights, cfg, l_hn)
    with pytorch_task(ctx, l_hn.read(), l_hidden.write()) as (thn, th):
        th[:] = thn

    stf_lm_head(ctx, l_hidden, weights["lm_head"], l_logits)
    stf_sample_argmax_last(ctx, l_logits, l_next)
    # Feed the new token back by overwriting the last-position hidden slot
    # with its embedding. Same trick as a 1-token KV append but for our
    # fixed-seq "rolling window" simplification.
    stf_append_token_hidden(ctx, l_hidden, weights["emb"], l_next)


# ---------------------------------------------------------------------------
# Mode 1 — eager ``stf.context``, host-driven loop
# ---------------------------------------------------------------------------


def _run_eager(cfg, max_new_tokens):
    B = 1
    rng = np.random.default_rng(SEED)
    hidden_host = rng.standard_normal((B, cfg.seq, cfg.hidden)).astype(cfg.np_dtype)
    logits_host = np.zeros((B, cfg.seq, cfg.vocab), dtype=cfg.np_dtype)
    next_host = np.zeros((B,), dtype=np.int64)

    t0 = time.perf_counter()
    ctx = stf.context()
    l_hidden = ctx.logical_data(hidden_host, name="hidden")
    l_logits = ctx.logical_data(logits_host, name="logits")
    l_next = ctx.logical_data(next_host, name="next_tok")
    weights = build_random_weights(ctx, cfg, seed=1, read_only=False)

    for _ in range(max_new_tokens):
        decode_step(ctx, cfg, weights, l_hidden, l_logits, l_next)

    ctx.finalize()
    elapsed = time.perf_counter() - t0

    validate_forward(hidden_host, cfg)
    return int(next_host[0]), elapsed


# ---------------------------------------------------------------------------
# Mode 2 — whole-program ``use_graph=True``, host-unrolled into one graph
# ---------------------------------------------------------------------------


def _run_graph(cfg, max_new_tokens):
    B = 1
    rng = np.random.default_rng(SEED)
    hidden_host = rng.standard_normal((B, cfg.seq, cfg.hidden)).astype(cfg.np_dtype)
    logits_host = np.zeros((B, cfg.seq, cfg.vocab), dtype=cfg.np_dtype)
    next_host = np.zeros((B,), dtype=np.int64)

    t0 = time.perf_counter()
    ctx = stf.context(use_graph=True)
    l_hidden = ctx.logical_data(hidden_host, name="hidden")
    l_logits = ctx.logical_data(logits_host, name="logits")
    l_next = ctx.logical_data(next_host, name="next_tok")
    weights = build_random_weights(ctx, cfg, seed=1, read_only=True)

    for _ in range(max_new_tokens):
        decode_step(ctx, cfg, weights, l_hidden, l_logits, l_next)

    ctx.finalize()
    elapsed = time.perf_counter() - t0

    validate_forward(hidden_host, cfg)
    return int(next_host[0]), elapsed


# ---------------------------------------------------------------------------
# Mode 3 — ``stackable_context`` + ``while_loop`` (conditional CUDA graph)
# ---------------------------------------------------------------------------


def _run_stackable_while(cfg, max_new_tokens):
    B = 1
    rng = np.random.default_rng(SEED)
    hidden_host = rng.standard_normal((B, cfg.seq, cfg.hidden)).astype(cfg.np_dtype)
    logits_host = np.zeros((B, cfg.seq, cfg.vocab), dtype=cfg.np_dtype)
    next_host = np.zeros((B,), dtype=np.int64)
    step_host = np.zeros((1,), dtype=np.float64)
    done_host = np.ones((1,), dtype=np.float64)

    t0 = time.perf_counter()
    ctx = stf.stackable_context()
    l_hidden = ctx.logical_data(hidden_host, name="hidden")
    l_logits = ctx.logical_data(logits_host, name="logits")
    l_next = ctx.logical_data(next_host, name="next_tok")
    l_step = ctx.logical_data(step_host, name="step")
    l_done = ctx.logical_data(done_host, name="done")
    # Pre-allocated scratch for the termination flag task — see
    # stf_advance_counter_flag. Must be declared before while_loop().
    l_cond_scratch = make_cond_scratch(ctx)
    weights = build_random_weights(ctx, cfg, seed=1, read_only=True)

    with ctx.while_loop() as loop:
        decode_step(ctx, cfg, weights, l_hidden, l_logits, l_next)

        # Advance step and compute done = (step < max_new_tokens) ? 1 : 0.
        # Uses the scratch-buffered helper to sidestep the PyTorch
        # caching-allocator / while-graph-capture mod-4 miscount described
        # in stf_advance_counter_flag and
        # tests/stf/probe_k_sweep_torch_variants.py.
        stf_advance_counter_flag(
            ctx, l_step, l_done, max_new_tokens, scratch=l_cond_scratch
        )

        loop.continue_while(l_done, ">", 0.5)

    ctx.finalize()
    elapsed = time.perf_counter() - t0

    validate_forward(hidden_host, cfg)
    return int(next_host[0]), elapsed


# ---------------------------------------------------------------------------
# Tests — each mode is its own pytest test so they show up separately.
# ---------------------------------------------------------------------------


def test_decode_eager():
    cfg = TINY
    tok, elapsed = _run_eager(cfg, MAX_TOKENS)
    print("\n=== LLM decode — eager ===")
    print(f"steps={MAX_TOKENS}  last_token={tok}  elapsed={elapsed * 1e3:.1f} ms")


@pytest.mark.skip(
    reason=(
        "stf.context(use_graph=True) + pytorch_task currently exposes host "
        "pointers at graph-capture time, which torch.as_tensor rejects. "
        "stackable_context + while_loop below covers the graph-capture story."
    )
)
def test_decode_graph():
    cfg = TINY
    tok, elapsed = _run_graph(cfg, MAX_TOKENS)
    print("\n=== LLM decode — use_graph=True ===")
    print(f"steps={MAX_TOKENS}  last_token={tok}  elapsed={elapsed * 1e3:.1f} ms")


def test_decode_stackable_while():
    cfg = TINY
    tok, elapsed = _run_stackable_while(cfg, MAX_TOKENS)
    print("\n=== LLM decode — stackable + while_loop ===")
    print(f"steps={MAX_TOKENS}  last_token={tok}  elapsed={elapsed * 1e3:.1f} ms")


def test_decode_modes_agree():
    """Eager and stackable+while_loop must emit the same greedy token.

    (Graph mode is skipped pending the host-pointer binding fix; see module
    docstring.)
    """
    cfg = TINY
    tok_eager, t_eager = _run_eager(cfg, MAX_TOKENS)
    tok_sw, t_sw = _run_stackable_while(cfg, MAX_TOKENS)

    print("\n=== LLM decode — same body, two deployment modes ===")
    print(f"  eager          : last_token={tok_eager}  ({t_eager * 1e3:.1f} ms)")
    print(f"  stackable+while: last_token={tok_sw}     ({t_sw * 1e3:.1f} ms)")

    assert tok_eager == tok_sw, (
        f"eager vs stackable+while diverged: {tok_eager} vs {tok_sw}"
    )
    print("Agreement: OK")


if __name__ == "__main__":
    test_decode_eager()
    test_decode_stackable_while()
    test_decode_modes_agree()
