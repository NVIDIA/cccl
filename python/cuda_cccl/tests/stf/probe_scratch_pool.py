"""Does pooling block scratches (one allocation, reused every iter/layer) unhang?

Run: `probe_scratch_pool.py`. Uses same body as M4 from probe_block_bisect,
but with per-chain shared scratches so the captured while-body has a bounded
number of logical_data_empty nodes.
"""

from __future__ import annotations

import numpy as np

import cuda.stf._experimental as stf
from llm_helpers import (
    TINY, build_random_weights, make_cond_scratch,
    stf_advance_counter_flag,
    stf_attention_sdpa, stf_ffn_fused, stf_layernorm, stf_linear,
)
from pytorch_task import pytorch_task


def make_pool(ctx, cfg):
    B, S, H = 1, cfg.seq, cfg.hidden
    d = cfg.np_dtype
    return {k: ctx.logical_data_empty((B, S, H), d, name=k)
            for k in ["xn", "Q", "K", "V", "attn", "proj", "x1", "x1n", "ffn_out"]}


def block_pooled(ctx, l_x, lw, l_out, cfg, pool):
    """M4 body, but every intermediate comes from ``pool`` (shared every call)."""
    stf_layernorm(ctx, l_x, lw["ln1_gamma"], lw["ln1_beta"], pool["xn"])
    stf_linear(ctx, pool["xn"], lw["Wq"], None, pool["Q"])
    stf_linear(ctx, pool["xn"], lw["Wk"], None, pool["K"])
    stf_linear(ctx, pool["xn"], lw["Wv"], None, pool["V"])
    stf_attention_sdpa(ctx, pool["Q"], pool["K"], pool["V"], pool["attn"], cfg)
    stf_linear(ctx, pool["attn"], lw["Wo"], None, pool["proj"])
    with pytorch_task(ctx, l_x.read(), pool["proj"].read(),
                      pool["x1"].write()) as (tx, tp, to):
        to[:] = tx + tp
    stf_layernorm(ctx, pool["x1"], lw["ln2_gamma"], lw["ln2_beta"], pool["x1n"])
    stf_ffn_fused(ctx, pool["x1n"], lw["W_up"], lw["b_up"],
                  lw["W_down"], lw["b_down"], l_out)


def run(NA: int, NB: int, rounds: int = 1):
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
    pool_a = make_pool(ctx, cfg)
    pool_b = make_pool(ctx, cfg)

    # Inter-layer buffers, also pooled per chain.
    B, S, H = 1, cfg.seq, cfg.hidden
    inter_a = [ctx.logical_data_empty((B, S, H), cfg.np_dtype, name=f"ia{i}")
               for i in range(max(NA, 1))]
    inter_b = [ctx.logical_data_empty((B, S, H), cfg.np_dtype, name=f"ib{i}")
               for i in range(max(NB, 1))]

    def chain(buf, n, inter, pool):
        cur = buf
        for i in range(n):
            block_pooled(ctx, cur, layer, inter[i], cfg, pool)
            cur = inter[i]
        return cur

    with ctx.while_loop() as loop:
        chain(l_a, NA, inter_a, pool_a)
        chain(l_b, NB, inter_b, pool_b)
        stf_advance_counter_flag(ctx, l_r, l_d, rounds, scratch=l_cs)
        loop.continue_while(l_d, ">", 0.5)

    ctx.finalize()


if __name__ == "__main__":
    for na, nb in [(2, 2), (4, 4), (1, 4), (4, 1), (2, 6), (6, 2)]:
        print(f"[pooled] NA={na} NB={nb} ...", flush=True)
        run(na, nb)
        print(f"[pooled]   OK", flush=True)
    print("all pooled configs passed", flush=True)
