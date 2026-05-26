# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
LLM demo — Layer D: speculative decoding (baseline, uncompiled).

Draft (2-layer) and target (6-layer) transformers share token embedding +
lm_head and run inside a single ``stackable_context`` whose body is a
fixed-count ``ctx.repeat(ROUNDS)`` CUDA-graph-replayed scope. Each outer
iteration:

  - draft proposes K tokens (sequentially, feeding each back)
  - target does ONE forward pass and verifies at the last K+1 positions
  - accept/reject via longest common prefix; emit the bonus token
  - slide both hidden states forward by one token

Presentation role: the slide "speculative decoding — one DAG, two models".
Perf IS the story here: compare wall time of target-only vs. spec-decode,
and report the per-round acceptance rate so the reader can sanity-check
the speedup ratio against theory.

Random weights give honest-but-small acceptance rates (the ``accept_rate``
printed should be understood as a structural number, not a quality
metric). With real draft/target pairs the same STF scaffold yields
real-world speedups; the demo focuses on the orchestration.

Scope choice: we use ``ctx.repeat(ROUNDS)`` rather than ``ctx.while_loop()``
because the number of rounds is fixed for the benchmark. Both are
``stackable_context`` CUDA-graph-replayed scopes; ``repeat`` simply skips
the conditional-termination predicate. The ``while_loop`` variant is
exercised by the ``test_llm_decode_loop.py`` demo (Layer B+C).

Requires CUDA 12.4+ (CUDA graphs + stackable context).

Env knobs
---------
``LLM_K=4``            speculative lookahead length
``LLM_ROUNDS=16``      number of spec-decode rounds = target forwards
``LLM_BASE_TOKENS``    number of target-only steps (defaults to ROUNDS*K)
"""

import os
import time

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from llm_helpers import (  # noqa: E402
    DRAFT,
    TINY,
    build_random_weights,
    spec_decode_loop,
    stf_append_token_hidden,
    stf_lm_head,
    stf_sample_argmax_last,
    stf_transformer_stack,
)
from pytorch_task import pytorch_task  # noqa: E402

import cuda.stf._experimental as stf  # noqa: E402

K = int(os.environ.get("LLM_K", "4"))
ROUNDS = int(os.environ.get("LLM_ROUNDS", "16"))
BASE_TOKENS = int(os.environ.get("LLM_BASE_TOKENS", str(ROUNDS * K)))
SEED = 0xC0DE


# ---------------------------------------------------------------------------
# Plain forwards used by both variants (no torch.compile in this file).
# ---------------------------------------------------------------------------


def make_forward(cfg, weights):
    """Build a ``(ctx, l_hidden, l_logits) -> None`` callable.

    Each call allocates fresh per-block scratches (the safer path for the
    K-serial-draft pattern inside ``spec_decode_loop``'s while_loop body —
    reusing a shared pool across K calls triggers STF dep-tracking hangs
    in practice).
    """

    def _forward(ctx, l_hidden, l_logits):
        B, S, H = 1, cfg.seq, cfg.hidden
        l_hn = ctx.logical_data_empty((B, S, H), cfg.np_dtype, name="h_next")
        stf_transformer_stack(ctx, l_hidden, weights, cfg, l_hn)
        with pytorch_task(ctx, l_hn.read(), l_hidden.write()) as (thn, th):
            th[:] = thn
        stf_lm_head(ctx, l_hidden, weights["lm_head"], l_logits)

    return _forward


# ---------------------------------------------------------------------------
# Variant A — target-only autoregressive decode (baseline).
# ---------------------------------------------------------------------------


def run_target_only(max_tokens: int):
    """Autoregressive baseline: target model decodes ``max_tokens`` in a
    CUDA-graph-replayed ``ctx.repeat(max_tokens)`` scope. Same scope kind
    as the spec-decode variant so the wall-time comparison is apples to
    apples."""
    cfg = TINY

    rng = np.random.default_rng(SEED)
    hidden_host = rng.standard_normal((1, cfg.seq, cfg.hidden)).astype(cfg.np_dtype)
    logits_host = np.zeros((1, cfg.seq, cfg.vocab), dtype=cfg.np_dtype)
    next_host = np.zeros((1,), dtype=np.int64)

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    ctx = stf.stackable_context()
    l_hidden = ctx.logical_data(hidden_host, name="target_hidden")
    l_logits = ctx.logical_data(logits_host, name="logits")
    l_next = ctx.logical_data(next_host, name="next_tok")

    weights = build_random_weights(ctx, cfg, seed=1, read_only=True)
    forward = make_forward(cfg, weights)

    with ctx.repeat(max_tokens):
        forward(ctx, l_hidden, l_logits)
        stf_sample_argmax_last(ctx, l_logits, l_next)
        stf_append_token_hidden(ctx, l_hidden, weights["emb"], l_next)

    ctx.finalize()
    torch.cuda.synchronize()
    return time.perf_counter() - t0, int(next_host[0])


# ---------------------------------------------------------------------------
# Variant B — STF speculative decode (draft + target, one DAG).
# ---------------------------------------------------------------------------


def run_spec_decode(max_rounds: int, K_val: int):
    cfg_t = TINY
    cfg_d = DRAFT

    rng = np.random.default_rng(SEED)
    hidden_t_host = rng.standard_normal((1, cfg_t.seq, cfg_t.hidden)).astype(
        cfg_t.np_dtype
    )
    hidden_d_host = hidden_t_host.copy()
    draft_toks_host = np.zeros((1, K_val + 1), dtype=np.int64)
    target_toks_host = np.zeros((1, K_val + 1), dtype=np.int64)
    accepted_host = np.zeros((1,), dtype=np.float64)

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    ctx = stf.stackable_context()
    l_hidden_t = ctx.logical_data(hidden_t_host, name="target_hidden")
    l_hidden_d = ctx.logical_data(hidden_d_host, name="draft_hidden")
    l_draft_tok = ctx.logical_data(draft_toks_host, name="draft_tok_buf")
    l_target_tok = ctx.logical_data(target_toks_host, name="target_tok_buf")
    l_accepted = ctx.logical_data(accepted_host, name="accepted_sum")

    target_w = build_random_weights(ctx, cfg_t, seed=1, read_only=True)
    draft_w = build_random_weights(
        ctx,
        cfg_d,
        seed=2,
        read_only=True,
        share_emb_lm_head_from=target_w,
    )

    target_forward = make_forward(cfg_t, target_w)
    draft_forward = make_forward(cfg_d, draft_w)

    spec_decode_loop(
        ctx,
        cfg_target=cfg_t,
        cfg_draft=cfg_d,
        target_forward=target_forward,
        draft_forward=draft_forward,
        l_hidden_target=l_hidden_t,
        l_hidden_draft=l_hidden_d,
        l_emb=target_w["emb"],
        l_draft_tokens=l_draft_tok,
        l_target_tokens=l_target_tok,
        l_accepted=l_accepted,
        K=K_val,
        max_rounds=max_rounds,
    )

    ctx.finalize()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    total_accepted = float(accepted_host[0])
    accept_rate = total_accepted / max(1, max_rounds * K_val)
    # Each round always emits the bonus token; accepted tokens are "extra"
    # prefix matches that reduce the effective per-token cost in real
    # spec-decode (with real draft/target pairs).
    tokens_emitted = max_rounds * (1.0 + total_accepted / max_rounds)
    return elapsed, tokens_emitted, accept_rate


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_speculative_decoding_headline():
    base_time, _ = run_target_only(BASE_TOKENS)
    spec_time, tokens_emitted, accept_rate = run_spec_decode(ROUNDS, K)

    base_tps = BASE_TOKENS / max(1e-6, base_time)
    spec_tps = tokens_emitted / max(1e-6, spec_time)
    speedup = spec_tps / max(1e-6, base_tps)

    print("\n=== LLM demo — Layer D: speculative decoding (STF, no compile) ===")
    print(f"Target model   : {TINY}")
    print(f"Draft model    : {DRAFT}")
    print(f"K (speculative): {K}   rounds: {ROUNDS}   base_tokens: {BASE_TOKENS}")
    print()
    print(f"{'variant':<32} {'tokens/s':>10} {'speedup':>8} {'accept_rate':>12}")
    print(f"{'target-only (STF repeat)':<32} {base_tps:>10.1f} {'1.00x':>8} {'—':>12}")
    print(
        f"{'spec decode (STF)':<32} "
        f"{spec_tps:>10.1f} {speedup:>7.2f}x {accept_rate:>12.2%}"
    )
    print()
    print("Note: with random weights, accept_rate is close to chance (~1/V).")
    print("This demo showcases the STF graph structure; on real draft/target")
    print("pairs the same scaffold is where spec-decode wins come from.")

    # Sanity asserts — cheap, since the test already ran.
    assert base_time > 0.0
    assert spec_time > 0.0
    assert 0.0 <= accept_rate <= 1.0


if __name__ == "__main__":
    test_speculative_decoding_headline()
