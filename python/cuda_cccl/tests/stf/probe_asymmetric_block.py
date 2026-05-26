"""Asymmetric chains of stf_transformer_block (no stack, no stf_lm_head).

Adds more complexity than ``probe_asymmetric_matmul`` (each block has
layernorm + 3 linears + sdpa + residual add + FFN + residual add) so we
can tell whether the hang is triggered by a specific op inside the block.
"""

from __future__ import annotations

import numpy as np
from llm_helpers import (
    TINY,
    build_random_weights,
    make_cond_scratch,
    stf_advance_counter_flag,
    stf_transformer_block,
)

import cuda.stf._experimental as stf


def run(NA: int, NB: int, *, rounds: int = 1):
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

    def chain(buf, n):
        cur = buf
        for _ in range(n):
            nxt = ctx.logical_data_empty(
                (1, cfg.seq, cfg.hidden), cfg.np_dtype, name="nxt"
            )
            stf_transformer_block(ctx, cur, layer, nxt, cfg)
            cur = nxt
        return cur

    with ctx.while_loop() as loop:
        # Same hidden buffer for both chains so they must serialize.
        chain(l_a, NA)
        chain(l_a, NB)
        stf_advance_counter_flag(ctx, l_r, l_d, rounds, scratch=l_cs)
        loop.continue_while(l_d, ">", 0.5)
    print("  body built, finalizing", flush=True)
    ctx.finalize()


if __name__ == "__main__":
    for NA, NB in [(1, 1), (2, 2), (6, 6), (2, 1), (1, 2), (6, 2), (2, 6)]:
        print(f"NA={NA} NB={NB} ...", flush=True)
        run(NA, NB)
        print("  OK", flush=True)
    print("all combos passed", flush=True)
