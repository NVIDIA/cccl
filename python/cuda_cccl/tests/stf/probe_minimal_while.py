"""Minimal sanity: does a plain while_loop with N rounds work at all?

Body = a few pytorch_task in-place adds. If this hangs with rounds>1, the
bug is in the while_loop machinery itself, not our workload.
"""

from __future__ import annotations

import numpy as np
from llm_helpers import make_cond_scratch, stf_advance_counter_flag
from pytorch_task import pytorch_task

import cuda.stf._experimental as stf


def run(depth: int, rounds: int):
    ctx = stf.stackable_context()
    a = np.zeros((4,), dtype=np.float32)
    l = ctx.logical_data(a, name="a")
    r = np.zeros((1,), dtype=np.float64)
    d = np.ones((1,), dtype=np.float64)
    l_r = ctx.logical_data(r, name="r")
    l_d = ctx.logical_data(d, name="d")
    l_cs = make_cond_scratch(ctx)

    with ctx.while_loop() as loop:
        for _ in range(depth):
            with pytorch_task(ctx, l.rw()) as (t,):
                t.add_(1.0)
        stf_advance_counter_flag(ctx, l_r, l_d, rounds, scratch=l_cs)
        loop.continue_while(l_d, ">", 0.5)

    ctx.finalize()
    out = a
    print(f"  depth={depth} rounds={rounds} -> a={out}")


if __name__ == "__main__":
    for depth, rounds in [(1, 1), (1, 4), (4, 1), (4, 4), (8, 8)]:
        print(f"[min] depth={depth} rounds={rounds}", flush=True)
        run(depth, rounds)
        print("[min]   OK", flush=True)
    print("all minimal configs passed", flush=True)
