"""Minimal asymmetric-chain probe using just matmul + temp writeback.

Goal: isolate the smallest workload that reproduces the stackable + while
hang when two chains of different length are composed.

Each "chain of depth N" is N back-to-back stf_linear-style tasks on the
same persistent buffer. Two such chains on disjoint buffers share the
same while body with our stf_advance_counter_flag counter.
"""

from __future__ import annotations

import sys

import numpy as np
import torch
from llm_helpers import make_cond_scratch, stf_advance_counter_flag
from pytorch_task import pytorch_task

import cuda.stf._experimental as stf


def run(NA: int, NB: int, *, H: int = 32, rounds: int = 2, writeback="setitem"):
    """Two asymmetric matmul chains in a single while body.

    writeback:
      - "setitem": ``to[:] = torch.matmul(tx, tw)`` (temp → persistent)
      - "out":     ``torch.matmul(tx, tw, out=to)`` (direct)
      - "copy":    ``to.copy_(torch.matmul(tx, tw))`` (equivalent to setitem)
    """
    rng = np.random.default_rng(0)
    a = rng.standard_normal((1, H, H)).astype(np.float32)
    b = rng.standard_normal((1, H, H)).astype(np.float32)
    w = rng.standard_normal((H, H)).astype(np.float32) * 0.01  # near identity
    r = np.zeros((1,), dtype=np.float64)
    d = np.ones((1,), dtype=np.float64)

    ctx = stf.stackable_context()
    l_a = ctx.logical_data(a, name="a")
    l_b = ctx.logical_data(b, name="b")
    l_w = ctx.logical_data(w, name="w")
    l_w.set_read_only()
    l_r = ctx.logical_data(r, name="r")
    l_d = ctx.logical_data(d, name="d")
    l_cs = make_cond_scratch(ctx)

    def chain(buf, n):
        for _ in range(n):
            tmp = ctx.logical_data_empty((1, H, H), np.float32, name="tmp")
            with pytorch_task(ctx, buf.read(), l_w.read(), tmp.write()) as (tx, tw, to):
                if writeback == "setitem":
                    to[:] = torch.matmul(tx, tw)
                elif writeback == "out":
                    torch.matmul(tx, tw, out=to)
                elif writeback == "copy":
                    to.copy_(torch.matmul(tx, tw))
            with pytorch_task(ctx, buf.write(), tmp.read()) as (tbuf, ttmp):
                tbuf.copy_(ttmp)  # STF -> STF

    with ctx.while_loop() as loop:
        chain(l_a, NA)
        chain(l_b, NB)
        stf_advance_counter_flag(ctx, l_r, l_d, rounds, scratch=l_cs)
        loop.continue_while(l_d, ">", 0.5)

    ctx.finalize()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "setitem"
    for NA, NB in [(2, 2), (6, 6), (6, 2), (2, 6)]:
        print(f"[{mode}] NA={NA} NB={NB} ...", flush=True)
        run(NA, NB, writeback=mode)
        print("  OK", flush=True)
    print(f"[{mode}] all combos passed", flush=True)
