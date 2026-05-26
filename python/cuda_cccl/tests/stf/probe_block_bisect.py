"""Bisect ``stf_transformer_block`` to find what triggers the while+asymmetric hang.

Each mode runs two disjoint chains of length 2 of a variant block inside a
``while_loop(rounds=1)`` body and prints PASS/HANG (via outer timeout).

Variants (progressively add stages of the real block):
  M0 linear only         : out = x @ W
  M1 layernorm + linear  : out = linear(ln(x))
  M2 M1 + residual add   : out = x + linear(ln(x))
  M3 M2 + attention      : "att" (sdpa) added
  M4 M3 + ffn            : full block, no residual 2
  M5 full block          : whole real block
"""

from __future__ import annotations

import numpy as np
from llm_helpers import (
    TINY,
    build_random_weights,
    make_cond_scratch,
    stf_advance_counter_flag,
    stf_attention_sdpa,
    stf_ffn_fused,
    stf_layernorm,
    stf_linear,
)
from pytorch_task import pytorch_task

import cuda.stf._experimental as stf


def block_M0(ctx, l_x, lw, l_out, cfg):
    """out = x @ Wo (plain linear)."""
    stf_linear(ctx, l_x, lw["Wo"], None, l_out)


def block_M1(ctx, l_x, lw, l_out, cfg):
    """out = linear(ln(x))."""
    s = ctx.logical_data_empty(l_x.shape, cfg.np_dtype, name="xn")
    stf_layernorm(ctx, l_x, lw["ln1_gamma"], lw["ln1_beta"], s)
    stf_linear(ctx, s, lw["Wo"], None, l_out)


def block_M2(ctx, l_x, lw, l_out, cfg):
    """out = x + linear(ln(x)) — residual add via persistent writeback."""
    s = ctx.logical_data_empty(l_x.shape, cfg.np_dtype, name="xn")
    stf_layernorm(ctx, l_x, lw["ln1_gamma"], lw["ln1_beta"], s)
    p = ctx.logical_data_empty(l_x.shape, cfg.np_dtype, name="proj")
    stf_linear(ctx, s, lw["Wo"], None, p)
    with pytorch_task(ctx, l_x.read(), p.read(), l_out.write()) as (tx, tp, to):
        to[:] = tx + tp


def block_M3(ctx, l_x, lw, l_out, cfg):
    """M2 + attention projection chain (no fused FFN)."""
    B, S, H = l_x.shape[0], l_x.shape[1], cfg.hidden
    s = ctx.logical_data_empty((B, S, H), cfg.np_dtype, name="xn")
    stf_layernorm(ctx, l_x, lw["ln1_gamma"], lw["ln1_beta"], s)
    Q = ctx.logical_data_empty((B, S, H), cfg.np_dtype, name="Q")
    K = ctx.logical_data_empty((B, S, H), cfg.np_dtype, name="K")
    V = ctx.logical_data_empty((B, S, H), cfg.np_dtype, name="V")
    stf_linear(ctx, s, lw["Wq"], None, Q)
    stf_linear(ctx, s, lw["Wk"], None, K)
    stf_linear(ctx, s, lw["Wv"], None, V)
    A = ctx.logical_data_empty((B, S, H), cfg.np_dtype, name="attn")
    stf_attention_sdpa(ctx, Q, K, V, A, cfg)
    p = ctx.logical_data_empty((B, S, H), cfg.np_dtype, name="proj")
    stf_linear(ctx, A, lw["Wo"], None, p)
    with pytorch_task(ctx, l_x.read(), p.read(), l_out.write()) as (tx, tp, to):
        to[:] = tx + tp


def block_M4(ctx, l_x, lw, l_out, cfg):
    """M3 + fused FFN on the residual output (no final residual)."""
    B, S, H = l_x.shape[0], l_x.shape[1], cfg.hidden
    s = ctx.logical_data_empty((B, S, H), cfg.np_dtype, name="xn")
    stf_layernorm(ctx, l_x, lw["ln1_gamma"], lw["ln1_beta"], s)
    Q = ctx.logical_data_empty((B, S, H), cfg.np_dtype, name="Q")
    K = ctx.logical_data_empty((B, S, H), cfg.np_dtype, name="K")
    V = ctx.logical_data_empty((B, S, H), cfg.np_dtype, name="V")
    stf_linear(ctx, s, lw["Wq"], None, Q)
    stf_linear(ctx, s, lw["Wk"], None, K)
    stf_linear(ctx, s, lw["Wv"], None, V)
    A = ctx.logical_data_empty((B, S, H), cfg.np_dtype, name="attn")
    stf_attention_sdpa(ctx, Q, K, V, A, cfg)
    p = ctx.logical_data_empty((B, S, H), cfg.np_dtype, name="proj")
    stf_linear(ctx, A, lw["Wo"], None, p)
    x1 = ctx.logical_data_empty((B, S, H), cfg.np_dtype, name="x1")
    with pytorch_task(ctx, l_x.read(), p.read(), x1.write()) as (tx, tp, to):
        to[:] = tx + tp
    xn2 = ctx.logical_data_empty((B, S, H), cfg.np_dtype, name="x1n")
    stf_layernorm(ctx, x1, lw["ln2_gamma"], lw["ln2_beta"], xn2)
    stf_ffn_fused(ctx, xn2, lw["W_up"], lw["b_up"], lw["W_down"], lw["b_down"], l_out)


def block_M5(ctx, l_x, lw, l_out, cfg):
    """Full real block."""
    from llm_helpers import stf_transformer_block

    stf_transformer_block(ctx, l_x, lw, l_out, cfg)


def block_M4b(ctx, l_x, lw, l_out, cfg):
    """M4 + extra no-op task writing l_out from ffn only (no re-read of x1)."""
    B, S, H = l_x.shape[0], l_x.shape[1], cfg.hidden
    s = ctx.logical_data_empty((B, S, H), cfg.np_dtype, name="xn")
    stf_layernorm(ctx, l_x, lw["ln1_gamma"], lw["ln1_beta"], s)
    Q = ctx.logical_data_empty((B, S, H), cfg.np_dtype, name="Q")
    K = ctx.logical_data_empty((B, S, H), cfg.np_dtype, name="K")
    V = ctx.logical_data_empty((B, S, H), cfg.np_dtype, name="V")
    stf_linear(ctx, s, lw["Wq"], None, Q)
    stf_linear(ctx, s, lw["Wk"], None, K)
    stf_linear(ctx, s, lw["Wv"], None, V)
    A = ctx.logical_data_empty((B, S, H), cfg.np_dtype, name="attn")
    stf_attention_sdpa(ctx, Q, K, V, A, cfg)
    p = ctx.logical_data_empty((B, S, H), cfg.np_dtype, name="proj")
    stf_linear(ctx, A, lw["Wo"], None, p)
    x1 = ctx.logical_data_empty((B, S, H), cfg.np_dtype, name="x1")
    with pytorch_task(ctx, l_x.read(), p.read(), x1.write()) as (tx, tp, to):
        to[:] = tx + tp
    xn2 = ctx.logical_data_empty((B, S, H), cfg.np_dtype, name="x1n")
    stf_layernorm(ctx, x1, lw["ln2_gamma"], lw["ln2_beta"], xn2)
    ffn = ctx.logical_data_empty((B, S, H), cfg.np_dtype, name="ffn_out")
    stf_ffn_fused(ctx, xn2, lw["W_up"], lw["b_up"], lw["W_down"], lw["b_down"], ffn)
    # Residual 2: only read ffn (NOT x1) and copy into l_out.
    with pytorch_task(ctx, ffn.read(), l_out.write()) as (tf, to):
        to[:] = tf


def block_M4c(ctx, l_x, lw, l_out, cfg):
    """M4b + final task ALSO reads x1 (exercises second read of Residual-1 out)."""
    B, S, H = l_x.shape[0], l_x.shape[1], cfg.hidden
    s = ctx.logical_data_empty((B, S, H), cfg.np_dtype, name="xn")
    stf_layernorm(ctx, l_x, lw["ln1_gamma"], lw["ln1_beta"], s)
    Q = ctx.logical_data_empty((B, S, H), cfg.np_dtype, name="Q")
    K = ctx.logical_data_empty((B, S, H), cfg.np_dtype, name="K")
    V = ctx.logical_data_empty((B, S, H), cfg.np_dtype, name="V")
    stf_linear(ctx, s, lw["Wq"], None, Q)
    stf_linear(ctx, s, lw["Wk"], None, K)
    stf_linear(ctx, s, lw["Wv"], None, V)
    A = ctx.logical_data_empty((B, S, H), cfg.np_dtype, name="attn")
    stf_attention_sdpa(ctx, Q, K, V, A, cfg)
    p = ctx.logical_data_empty((B, S, H), cfg.np_dtype, name="proj")
    stf_linear(ctx, A, lw["Wo"], None, p)
    x1 = ctx.logical_data_empty((B, S, H), cfg.np_dtype, name="x1")
    with pytorch_task(ctx, l_x.read(), p.read(), x1.write()) as (tx, tp, to):
        to[:] = tx + tp
    xn2 = ctx.logical_data_empty((B, S, H), cfg.np_dtype, name="x1n")
    stf_layernorm(ctx, x1, lw["ln2_gamma"], lw["ln2_beta"], xn2)
    ffn = ctx.logical_data_empty((B, S, H), cfg.np_dtype, name="ffn_out")
    stf_ffn_fused(ctx, xn2, lw["W_up"], lw["b_up"], lw["W_down"], lw["b_down"], ffn)
    # Residual 2: read BOTH x1 and ffn, write l_out.
    with pytorch_task(ctx, x1.read(), ffn.read(), l_out.write()) as (tx1, tf, to):
        to[:] = tx1 + tf


BLOCKS = {
    "M0": block_M0,
    "M1": block_M1,
    "M2": block_M2,
    "M3": block_M3,
    "M4": block_M4,
    "M4b": block_M4b,
    "M4c": block_M4c,
    "M5": block_M5,
}


def run(mode: str, NA: int = 2, NB: int = 2, rounds: int = 1):
    cfg = TINY
    rng = np.random.default_rng(0)
    ha = rng.standard_normal((1, cfg.seq, cfg.hidden)).astype(cfg.np_dtype)
    hb = rng.standard_normal((1, cfg.seq, cfg.hidden)).astype(cfg.np_dtype)
    r = np.zeros((1,), dtype=np.float64)
    d = np.ones((1,), dtype=np.float64)

    ctx = stf.stackable_context()
    l_a = ctx.logical_data(ha, name="ha")
    l_b = ctx.logical_data(hb, name="hb")
    l_r = ctx.logical_data(r, name="r")
    l_d = ctx.logical_data(d, name="d")
    l_cs = make_cond_scratch(ctx)
    w = build_random_weights(ctx, cfg, seed=1, read_only=True)
    layer = w["layers"][0]
    block = BLOCKS[mode]

    def chain(buf, n):
        cur = buf
        for _ in range(n):
            nxt = ctx.logical_data_empty(
                (1, cfg.seq, cfg.hidden), cfg.np_dtype, name="nxt"
            )
            block(ctx, cur, layer, nxt, cfg)
            cur = nxt
        return cur

    with ctx.while_loop() as loop:
        chain(l_a, NA)
        chain(l_b, NB)
        stf_advance_counter_flag(ctx, l_r, l_d, rounds, scratch=l_cs)
        loop.continue_while(l_d, ">", 0.5)

    ctx.finalize()


if __name__ == "__main__":
    # Is it chain length, block identity, or asymmetry?
    tests = [
        ("M4", 3, 3),  # longer symmetric M4
        ("M4", 4, 4),  # even longer M4
        ("M4b", 1, 1),  # short M4b
        ("M4b", 2, 1),  # asymmetric M4b
        ("M4b", 1, 2),  # asymmetric M4b, other way
    ]
    for mode, na, nb in tests:
        print(f"[{mode}] NA={na} NB={nb} ...", flush=True)
        run(mode, NA=na, NB=nb)
        print(f"[{mode}]   OK", flush=True)
    print("all modes passed", flush=True)
