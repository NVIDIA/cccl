"""Does a thin transformer-like block survive while_loop + asymmetric chains?

Thin block = layernorm -> linear -> gelu -> linear + residual. About 3-4 tasks.
Runs two chains of depth NA and NB inside a stackable while_loop.
"""

from __future__ import annotations

import numpy as np
import torch

import cuda.stf._experimental as stf
from llm_helpers import (
    TINY, make_cond_scratch, stf_advance_counter_flag, stf_layernorm, stf_linear,
)
from pytorch_task import pytorch_task


def thin_block(ctx, l_x, lw, l_out, cfg):
    """Residual-MLP: out = x + W2(gelu(W1(ln(x))))."""
    B, S, H = l_x.shape[0], l_x.shape[1], cfg.hidden
    d = cfg.np_dtype
    xn = ctx.logical_data_empty((B, S, H), d, name="xn")
    stf_layernorm(ctx, l_x, lw["ln1_gamma"], lw["ln1_beta"], xn)
    h = ctx.logical_data_empty((B, S, cfg.ffn_hidden), d, name="h")
    stf_linear(ctx, xn, lw["W_up"], lw["b_up"], h)
    p = ctx.logical_data_empty((B, S, H), d, name="p")
    with pytorch_task(ctx, h.read(), lw["W_down"].read(), lw["b_down"].read(),
                      p.write()) as (th, tw, tb, tp):
        tp[:] = torch.nn.functional.gelu(th) @ tw + tb
    with pytorch_task(ctx, l_x.read(), p.read(), l_out.write()) as (tx, tpp, to):
        to[:] = tx + tpp


def run(NA: int, NB: int, rounds: int = 4):
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

    from llm_helpers import build_random_weights
    w = build_random_weights(ctx, cfg, seed=1, read_only=True)
    lw = w["layers"][0]

    def chain(buf, n):
        cur = buf
        for _ in range(n):
            nxt = ctx.logical_data_empty(
                (1, cfg.seq, cfg.hidden), cfg.np_dtype, name="nxt"
            )
            thin_block(ctx, cur, lw, nxt, cfg)
            cur = nxt
        return cur

    with ctx.while_loop() as loop:
        chain(l_a, NA)
        chain(l_b, NB)
        stf_advance_counter_flag(ctx, l_r, l_d, rounds, scratch=l_cs)
        loop.continue_while(l_d, ">", 0.5)

    ctx.finalize()


if __name__ == "__main__":
    # rounds=1 first to isolate body-size vs rounds
    for na, nb, rnd in [(1, 1, 1), (4, 4, 1), (8, 8, 1),
                        (1, 1, 4), (4, 4, 2), (4, 4, 4),
                        (1, 4, 1), (4, 1, 1),
                        (1, 4, 4), (4, 1, 4)]:
        print(f"[thin] NA={na} NB={nb} rounds={rnd} ...", flush=True)
        run(na, nb, rounds=rnd)
        print(f"[thin]   OK", flush=True)
    print("ALL thin configs passed", flush=True)
