# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Probe: distinguish "body-body-body long dep chain on ONE logical_data"
from "body allocates/recycles many logical_data".

Variants, all inside one ``stackable_context + while_loop`` body:

- one_ld_many_rw (baseline, known broken):
    one persistent logical_data, K sequential .rw() in body.

- many_ld_persistent_one_rw_each:
    K persistent logical_data, each .rw() exactly once in body, all chained
    via a running accumulator. If the bug were about "chain length on a
    single logical_data", this would also break at K=4. If the bug is
    about allocator/buffer recycling WITHIN a logical_data's access
    history in the captured body, this should stay correct.

- fresh_inbody_many_ld_one_rw_each:
    K fresh ``logical_data_empty`` allocated inside body, each .rw() once,
    chained via accumulator. Tests pure "fresh allocations inside body,
    no single long chain" — if this breaks it implicates in-body
    allocator state.

Expected final value of l_acc after ``rounds`` body replays, starting 0:
    all variants: rounds * K
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pytorch_task import pytorch_task  # noqa: E402

import cuda.stf._experimental as stf  # noqa: E402


N = 128


def _bump_round(ctx, l_round, l_done, max_rounds, loop):
    with pytorch_task(ctx, l_round.rw(), l_done.write()) as (tr, td):
        tr[:] = tr + 1.0
        flag = (tr < float(max_rounds)).to(td.dtype)
        td.copy_(flag.view(td.shape))
    loop.continue_while(l_done, ">", 0.5)


def run_one_ld_many_rw(rounds, K):
    acc = np.zeros(N, dtype=np.float32)
    r = np.zeros((1,), dtype=np.float64)
    d = np.ones((1,), dtype=np.float64)
    ctx = stf.stackable_context()
    l_acc = ctx.logical_data(acc, name="acc")
    l_r = ctx.logical_data(r, name="round")
    l_d = ctx.logical_data(d, name="done")
    with ctx.while_loop() as loop:
        for _ in range(K):
            with pytorch_task(ctx, l_acc.rw()) as (ta,):
                ta[:] = ta + 1.0
        _bump_round(ctx, l_r, l_d, rounds, loop)
    ctx.finalize()
    return float(acc[0])


def run_many_ld_persistent_one_rw_each(rounds, K):
    """K persistent logical_data (allocated OUTSIDE body), each used with
    exactly one .rw() inside the body. Accumulator is also persistent."""
    acc = np.zeros(N, dtype=np.float32)
    r = np.zeros((1,), dtype=np.float64)
    d = np.ones((1,), dtype=np.float64)
    ctx = stf.stackable_context()
    l_acc = ctx.logical_data(acc, name="acc")
    l_r = ctx.logical_data(r, name="round")
    l_d = ctx.logical_data(d, name="done")
    # K persistent logical_data, each of them initialised to 1.0 on device.
    l_ones = []
    for k in range(K):
        ones_host = np.ones(N, dtype=np.float32)
        l_ones.append(ctx.logical_data(ones_host, name=f"ones{k}"))
    with ctx.while_loop() as loop:
        for k in range(K):
            # single .rw() on acc, single .read() on l_ones[k]
            with pytorch_task(ctx, l_acc.rw(), l_ones[k].read()) as (ta, to):
                ta[:] = ta + to
        _bump_round(ctx, l_r, l_d, rounds, loop)
    ctx.finalize()
    return float(acc[0])


def run_fresh_inbody_many_ld_one_rw_each(rounds, K):
    """K fresh logical_data allocated INSIDE the body, each .read() exactly
    once, chained via persistent acc."""
    acc = np.zeros(N, dtype=np.float32)
    r = np.zeros((1,), dtype=np.float64)
    d = np.ones((1,), dtype=np.float64)
    ctx = stf.stackable_context()
    l_acc = ctx.logical_data(acc, name="acc")
    l_r = ctx.logical_data(r, name="round")
    l_d = ctx.logical_data(d, name="done")
    with ctx.while_loop() as loop:
        for k in range(K):
            l_one = ctx.logical_data_empty((N,), np.float32, name=f"one{k}")
            with pytorch_task(ctx, l_one.write()) as (to,):
                to[:] = 1.0
            with pytorch_task(ctx, l_acc.rw(), l_one.read()) as (ta, to):
                ta[:] = ta + to
        _bump_round(ctx, l_r, l_d, rounds, loop)
    ctx.finalize()
    return float(acc[0])


def main():
    print(f"{'variant':<36} {'rounds':>6} {'K':>3}  {'exp':>6} {'got':>6}  ok")
    print("-" * 72)
    combos = [(1, 1), (1, 2), (1, 3), (1, 4), (1, 6), (2, 4), (4, 4), (8, 4)]
    for name, fn in [
        ("one_ld_many_rw",                       run_one_ld_many_rw),
        ("many_ld_persistent_one_rw_each",       run_many_ld_persistent_one_rw_each),
        ("fresh_inbody_many_ld_one_rw_each",     run_fresh_inbody_many_ld_one_rw_each),
    ]:
        for rounds, K in combos:
            try:
                got = fn(rounds, K)
                exp = rounds * K
                ok = "OK" if abs(got - exp) < 1e-3 else "!!"
            except Exception as e:
                got = float("nan")
                exp = rounds * K
                ok = f"ERR {type(e).__name__}"
            print(f"{name:<36} {rounds:>6d} {K:>3d}  {exp:>6d} {got:>6.1f}  {ok}")


if __name__ == "__main__":
    main()
