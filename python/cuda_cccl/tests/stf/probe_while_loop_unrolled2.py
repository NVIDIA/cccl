# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Probe 2: vary the INTRA-BODY pattern while keeping everything else fixed.

- Variant A (fresh_scratch): K times { fresh scratch, write scratch=h+1, h=scratch }
- Variant B (shared_scratch): reuse ONE scratch across K iterations
- Variant C (direct_rw): K times { h.rw(): h+=1 } — no scratch at all
- Variant D (two_step_nosc): K times { h.rw(): h+=0.5 } x2 per k — 2K .rw on h

Expected final value of h after `rounds` body replays, starting at h=0:
  A/B/C: rounds * K
  D:     rounds * K  (2K half-steps)
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pytorch_task import pytorch_task  # noqa: E402

import cuda.stf._experimental as stf  # noqa: E402

N = 128


def _body_fresh_scratch(ctx, l_hidden, l_round, l_done, K, max_rounds):
    with ctx.while_loop() as loop:
        for k in range(K):
            l_s = ctx.logical_data_empty((N,), np.float32, name=f"scr{k}")
            with pytorch_task(ctx, l_hidden.read(), l_s.write()) as (th, ts):
                ts[:] = th + 1.0
            with pytorch_task(ctx, l_s.read(), l_hidden.write()) as (ts, th):
                th[:] = ts
        _bump_round(ctx, l_round, l_done, max_rounds, loop)


def _body_shared_scratch(ctx, l_hidden, l_round, l_done, K, max_rounds):
    l_s = ctx.logical_data_empty((N,), np.float32, name="scr")
    with ctx.while_loop() as loop:
        for k in range(K):
            with pytorch_task(ctx, l_hidden.read(), l_s.write()) as (th, ts):
                ts[:] = th + 1.0
            with pytorch_task(ctx, l_s.read(), l_hidden.write()) as (ts, th):
                th[:] = ts
        _bump_round(ctx, l_round, l_done, max_rounds, loop)


def _body_direct_rw(ctx, l_hidden, l_round, l_done, K, max_rounds):
    with ctx.while_loop() as loop:
        for k in range(K):
            with pytorch_task(ctx, l_hidden.rw()) as (th,):
                th[:] = th + 1.0
        _bump_round(ctx, l_round, l_done, max_rounds, loop)


def _body_two_step_nosc(ctx, l_hidden, l_round, l_done, K, max_rounds):
    with ctx.while_loop() as loop:
        for k in range(K):
            with pytorch_task(ctx, l_hidden.rw()) as (th,):
                th[:] = th + 0.5
            with pytorch_task(ctx, l_hidden.rw()) as (th,):
                th[:] = th + 0.5
        _bump_round(ctx, l_round, l_done, max_rounds, loop)


def _bump_round(ctx, l_round, l_done, max_rounds, loop):
    with pytorch_task(ctx, l_round.rw(), l_done.write()) as (tr, td):
        tr[:] = tr + 1.0
        flag = (tr < float(max_rounds)).to(td.dtype)
        td.copy_(flag.view(td.shape))
    loop.continue_while(l_done, ">", 0.5)


def run(variant, rounds, K):
    h = np.zeros(N, dtype=np.float32)
    r = np.zeros((1,), dtype=np.float64)
    d = np.ones((1,), dtype=np.float64)

    ctx = stf.stackable_context()
    lh = ctx.logical_data(h, name="h")
    lr = ctx.logical_data(r, name="round")
    ld = ctx.logical_data(d, name="done")

    body = {
        "fresh_scratch": _body_fresh_scratch,
        "shared_scratch": _body_shared_scratch,
        "direct_rw": _body_direct_rw,
        "two_step_nosc": _body_two_step_nosc,
    }[variant]
    body(ctx, lh, lr, ld, K, rounds)
    ctx.finalize()
    return float(h[0])


def main():
    print(f"{'variant':<16} {'rounds':>6} {'K':>3}  {'exp':>6} {'got':>6}  {'ok':>3}")
    print("-" * 55)

    combos = [(1, 1), (1, 2), (1, 4), (2, 2), (2, 4), (4, 2), (4, 4), (8, 2)]
    variants = ["fresh_scratch", "shared_scratch", "direct_rw", "two_step_nosc"]

    for v in variants:
        for rounds, K in combos:
            try:
                got = run(v, rounds, K)
                exp = rounds * K
                ok = abs(got - exp) < 1e-3
                mark = "OK" if ok else "!!"
            except Exception as e:
                got = float("nan")
                exp = rounds * K
                mark = f"ERR {type(e).__name__}"
            print(f"{v:<16} {rounds:>6d} {K:>3d}  {exp:>6d} {got:>6.1f}  {mark}")


if __name__ == "__main__":
    main()
