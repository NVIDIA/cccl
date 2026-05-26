# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Same shape as probe_k_sweep.py, but replace pytorch_task with a numba-compiled
kernel launched via ctx.task(...). If this probe still shows the mod-4 drop,
the bug is in the Cython stackable_task glue; if it passes, the bug is
specific to PyTorch integration (ExternalStream + caching allocator inside
while_graph_scope capture).
"""

from __future__ import annotations

import numpy as np
import pytest

numba_cuda = pytest.importorskip("numba.cuda")

from numba_decorator import jit as stf_jit  # noqa: E402

import cuda.stf._experimental as stf  # noqa: E402

N = 128


@stf_jit
def add_one(data):
    i = numba_cuda.grid(1)
    if i < data.shape[0]:
        data[i] = data[i] + 1.0


@stf_jit
def zero_done(done):
    i = numba_cuda.grid(1)
    if i < done.shape[0]:
        done[i] = 0.0


def run(K, rounds=1):
    acc = np.zeros(N, dtype=np.float32)
    d = np.ones((1,), dtype=np.float64)
    ctx = stf.stackable_context()
    l_acc = ctx.logical_data(acc, name="acc")
    l_d = ctx.logical_data(d, name="done")
    with ctx.while_loop() as loop:
        for _ in range(K):
            add_one[(N + 63) // 64, 64](l_acc.rw())
        zero_done[1, 1](l_d.write())
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
