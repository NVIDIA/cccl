# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Sweep K from 1 to 12 on the minimal "one persistent LD, K sequential .rw()
in body" pattern. Is K=4 a singleton failure or is there a periodic / range
pattern?
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pytorch_task import pytorch_task  # noqa: E402

import cuda.stf._experimental as stf  # noqa: E402

N = 128


def run(K, rounds=1):
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
        with pytorch_task(ctx, l_r.rw(), l_d.write()) as (tr, td):
            tr[:] = tr + 1.0
            flag = (tr < float(rounds)).to(td.dtype)
            td.copy_(flag.view(td.shape))
        loop.continue_while(l_d, ">", 0.5)
    ctx.finalize()
    return float(acc[0])


def main():
    print(f"{'K':>3} {'exp':>5} {'got':>7}  status")
    for K in range(1, 17):
        got = run(K, rounds=1)
        exp = K
        ok = "OK" if abs(got - exp) < 1e-3 else f"!! off by {got - exp:+.1f}"
        print(f"{K:>3} {exp:>5} {got:>7.1f}  {ok}")


if __name__ == "__main__":
    main()
