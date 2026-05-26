"""Truly minimal probe for the asymmetric-chain hang.

Strip out all PyTorch / transformer machinery. Just two chains of
``pytorch_task`` in-place adds of different lengths, inside a
``stackable_context.while_loop``.

We vary: (NA, NB, order) over small integers and print PASS / HANG.
Timeout is enforced at the outer shell level (this file just prints
progress so a hanging run is obvious).
"""

from __future__ import annotations

import sys

import numpy as np
from llm_helpers import make_cond_scratch, stf_advance_counter_flag
from pytorch_task import pytorch_task

import cuda.stf._experimental as stf


def chain(ctx, l_buf, n):
    """n chained .rw() in-place adds on l_buf."""
    for _ in range(n):
        with pytorch_task(ctx, l_buf.rw()) as (t,):
            t.add_(1.0)


def run(NA: int, NB: int, *, order="a_then_b", rounds=2):
    a = np.zeros(4, dtype=np.float32)
    b = np.zeros(4, dtype=np.float32)
    r = np.zeros((1,), dtype=np.float64)
    d = np.ones((1,), dtype=np.float64)

    ctx = stf.stackable_context()
    l_a = ctx.logical_data(a, name="a")
    l_b = ctx.logical_data(b, name="b")
    l_r = ctx.logical_data(r, name="r")
    l_d = ctx.logical_data(d, name="d")
    l_cs = make_cond_scratch(ctx)

    with ctx.while_loop() as loop:
        if order == "a_then_b":
            chain(ctx, l_a, NA)
            chain(ctx, l_b, NB)
        else:
            chain(ctx, l_b, NB)
            chain(ctx, l_a, NA)
        stf_advance_counter_flag(ctx, l_r, l_d, rounds, scratch=l_cs)
        loop.continue_while(l_d, ">", 0.5)

    ctx.finalize()
    return float(a[0]), float(b[0])


if __name__ == "__main__":
    for NA, NB in [(2, 2), (6, 6), (6, 2), (2, 6), (3, 5), (5, 3)]:
        for order in ("a_then_b", "b_then_a"):
            print(f"NA={NA} NB={NB} order={order} ...", flush=True)
            va, vb = run(NA, NB, order=order, rounds=2)
            print(f"  a={va} b={vb}", flush=True)
    print("all combos passed", flush=True)
    sys.exit(0)
