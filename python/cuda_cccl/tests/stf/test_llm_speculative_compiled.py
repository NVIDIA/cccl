# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
LLM demo — Layer D': speculative decoding with ``torch.compile``.

Companion to ``test_llm_speculative.py``. Same speculative-decoding
scaffold, but the per-block forward is wrapped in ``torch.compile(
mode="default")`` so Inductor fuses kernels within a block. STF still owns
graph capture (via ``stackable_context`` + ``ctx.repeat``); Inductor is
only used *inside* each ``pytorch_task`` for kernel-level fusion.

``mode="default"`` is deliberate: it enables Inductor fusion but NOT the
CUDA-graph capture that ``mode="reduce-overhead"`` performs. Nested graph
capture inside STF's own graph scope would collide — hence this explicit
choice.

Presentation role: the slide "STF + ``torch.compile`` compose". The
headline is again tokens/s, and the reader can compare against the
numbers printed by ``test_llm_speculative.py`` for the three-way story.

Requires CUDA 12.4+ (conditional graph nodes) + PyTorch >= 2.1 (Inductor).
Cold compile can take 10-30 s the first time — ``pytest-timeout`` should
allow for that in the test suite.

Env knobs
---------
``LLM_K=4``            speculative lookahead length
``LLM_ROUNDS=16``      spec-decode rounds
``LLM_BASE_TOKENS``    target-only steps (defaults to ROUNDS*K)
``LLM_WARMUP=5``       extra warmup iterations after first compile
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
)
from pytorch_task import pytorch_task  # noqa: E402

import cuda.stf._experimental as stf  # noqa: E402

K = int(os.environ.get("LLM_K", "4"))
ROUNDS = int(os.environ.get("LLM_ROUNDS", "16"))
BASE_TOKENS = int(os.environ.get("LLM_BASE_TOKENS", str(ROUNDS * K)))
WARMUP = int(os.environ.get("LLM_WARMUP", "5"))
SEED = 0xC0DE


# ---------------------------------------------------------------------------
# Compiled transformer block (one Inductor-fused kernel region).
#
# All weights flow in as positional tensor args so the compiler can specialize
# on shapes / strides. ``heads`` and ``head_dim`` are static scalars.
# ---------------------------------------------------------------------------


def _block_fn(
    x,
    ln1_g,
    ln1_b,
    Wq,
    Wk,
    Wv,
    Wo,
    ln2_g,
    ln2_b,
    Wup,
    bup,
    Wdn,
    bdn,
    heads: int,
    head_dim: int,
):
    F = torch.nn.functional
    hidden = heads * head_dim

    xn = F.layer_norm(x, ln1_g.shape, ln1_g, ln1_b, eps=1e-5)
    q = torch.matmul(xn, Wq)
    k = torch.matmul(xn, Wk)
    v = torch.matmul(xn, Wv)

    b, s, _ = q.shape
    q = q.view(b, s, heads, head_dim).transpose(1, 2).contiguous()
    k = k.view(b, s, heads, head_dim).transpose(1, 2).contiguous()
    v = v.view(b, s, heads, head_dim).transpose(1, 2).contiguous()
    att = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    att = att.transpose(1, 2).contiguous().view(b, s, hidden)

    x1 = x + torch.matmul(att, Wo)
    x1n = F.layer_norm(x1, ln2_g.shape, ln2_g, ln2_b, eps=1e-5)
    h = F.gelu(torch.matmul(x1n, Wup) + bup)
    return x1 + torch.matmul(h, Wdn) + bdn


# One compiled artifact shared by both target and draft. torch.compile keys
# on structure + input shapes, so target (6 layers) and draft (2 layers)
# using the same shapes will share the compile cache entry.
_compiled_block = torch.compile(_block_fn, mode="default", fullgraph=True)


def _stf_block_compiled(ctx, l_x, layer_w, l_out, cfg):
    """One STF task = one compiled transformer block."""
    with pytorch_task(
        ctx,
        l_x.read(),
        l_out.write(),
        layer_w["ln1_gamma"].read(),
        layer_w["ln1_beta"].read(),
        layer_w["Wq"].read(),
        layer_w["Wk"].read(),
        layer_w["Wv"].read(),
        layer_w["Wo"].read(),
        layer_w["ln2_gamma"].read(),
        layer_w["ln2_beta"].read(),
        layer_w["W_up"].read(),
        layer_w["b_up"].read(),
        layer_w["W_down"].read(),
        layer_w["b_down"].read(),
    ) as (
        tx,
        to,
        tg1,
        tb1,
        Wq,
        Wk,
        Wv,
        Wo,
        tg2,
        tb2,
        Wup,
        bup,
        Wdn,
        bdn,
    ):
        out = _compiled_block(
            tx,
            tg1,
            tb1,
            Wq,
            Wk,
            Wv,
            Wo,
            tg2,
            tb2,
            Wup,
            bup,
            Wdn,
            bdn,
            cfg.heads,
            cfg.head_dim,
        )
        to[:] = out


def _stf_transformer_stack_compiled(ctx, l_in, weights, cfg, l_out):
    """Same as llm_helpers.stf_transformer_stack but each block is one
    Inductor-compiled task."""
    B, S, H = 1, cfg.seq, cfg.hidden
    cur = l_in
    for i, layer_w in enumerate(weights["layers"]):
        nxt = (
            l_out
            if i == cfg.n_layers - 1
            else ctx.logical_data_empty((B, S, H), cfg.np_dtype, name=f"h{i + 1}")
        )
        _stf_block_compiled(ctx, cur, layer_w, nxt, cfg)
        cur = nxt


def _make_compiled_forward(cfg, weights):
    def _forward(ctx, l_hidden, l_logits):
        B, S, H = 1, cfg.seq, cfg.hidden
        l_hn = ctx.logical_data_empty((B, S, H), cfg.np_dtype, name="h_next")
        _stf_transformer_stack_compiled(ctx, l_hidden, weights, cfg, l_hn)
        with pytorch_task(ctx, l_hn.read(), l_hidden.write()) as (thn, th):
            th[:] = thn
        stf_lm_head(ctx, l_hidden, weights["lm_head"], l_logits)

    return _forward


# ---------------------------------------------------------------------------
# Compile-warmup: trigger Inductor codegen OUTSIDE any STF context.
#
# Without this, first-call compilation would happen mid-``while_loop`` body
# where CUDA graph capture is active — that path is fragile. One eager
# call on dummy tensors populates the torch.compile cache so all STF
# replays see a ready-made artifact.
# ---------------------------------------------------------------------------


def _warmup_compile(cfg):
    device = torch.device("cuda")
    B, S, H = 1, cfg.seq, cfg.hidden
    dtype = cfg.torch_dtype

    x = torch.randn(B, S, H, dtype=dtype, device=device)
    lng = torch.ones(H, dtype=dtype, device=device)
    lnb = torch.zeros(H, dtype=dtype, device=device)
    W = lambda a, b: torch.randn(a, b, dtype=dtype, device=device)  # noqa: E731
    b1d = lambda n: torch.zeros(n, dtype=dtype, device=device)  # noqa: E731

    args = (
        x,
        lng,
        lnb,
        W(H, H),
        W(H, H),
        W(H, H),
        W(H, H),
        lng,
        lnb,
        W(H, cfg.ffn_hidden),
        b1d(cfg.ffn_hidden),
        W(cfg.ffn_hidden, H),
        b1d(H),
        cfg.heads,
        cfg.head_dim,
    )
    for _ in range(1 + WARMUP):
        _ = _compiled_block(*args)
    torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Variant A — target-only decode, compiled block.
# ---------------------------------------------------------------------------


def run_target_only_compiled(max_tokens: int):
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
    forward = _make_compiled_forward(cfg, weights)

    with ctx.repeat(max_tokens):
        forward(ctx, l_hidden, l_logits)
        stf_sample_argmax_last(ctx, l_logits, l_next)
        stf_append_token_hidden(ctx, l_hidden, weights["emb"], l_next)

    ctx.finalize()
    torch.cuda.synchronize()
    return time.perf_counter() - t0, int(next_host[0])


# ---------------------------------------------------------------------------
# Variant B — STF speculative decode with compiled draft + target blocks.
# ---------------------------------------------------------------------------


def run_spec_decode_compiled(max_rounds: int, K_val: int):
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

    target_forward = _make_compiled_forward(cfg_t, target_w)
    draft_forward = _make_compiled_forward(cfg_d, draft_w)

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
    tokens_emitted = max_rounds * (1.0 + total_accepted / max_rounds)
    return elapsed, tokens_emitted, accept_rate


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


def test_speculative_decoding_compiled_headline():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")

    # Warm up the Inductor compile before any STF graph capture runs.
    _warmup_compile(TINY)
    _warmup_compile(DRAFT)

    base_time, _ = run_target_only_compiled(BASE_TOKENS)
    spec_time, tokens_emitted, accept_rate = run_spec_decode_compiled(ROUNDS, K)

    base_tps = BASE_TOKENS / max(1e-6, base_time)
    spec_tps = tokens_emitted / max(1e-6, spec_time)
    speedup = spec_tps / max(1e-6, base_tps)

    print("\n=== LLM demo — Layer D': speculative decoding (STF + torch.compile) ===")
    print(f"Target: {TINY}")
    print(f"Draft : {DRAFT}")
    print(f"K (speculative): {K}   rounds: {ROUNDS}   base_tokens: {BASE_TOKENS}")
    print(f"torch.compile mode=default, warmup={1 + WARMUP} per model")
    print()
    print(f"{'variant':<38} {'tokens/s':>10} {'speedup':>8} {'accept_rate':>12}")
    print(
        f"{'target-only compiled (STF repeat)':<38} "
        f"{base_tps:>10.1f} {'1.00x':>8} {'—':>12}"
    )
    print(
        f"{'spec decode (STF + compile)':<38} "
        f"{spec_tps:>10.1f} {speedup:>7.2f}x {accept_rate:>12.2%}"
    )
    print()
    print("Compare the tokens/s numbers against test_llm_speculative.py")
    print("for the three-way story (eager / spec / spec+compile).")

    assert base_time > 0.0
    assert spec_time > 0.0
    assert 0.0 <= accept_rate <= 1.0


if __name__ == "__main__":
    test_speculative_decoding_compiled_headline()
