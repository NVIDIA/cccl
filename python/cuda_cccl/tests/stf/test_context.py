# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np

from cuda.cccl.experimental.stf._stf_bindings import context, read, rw


def test_ctx():
    ctx = context()
    del ctx


def test_graph_ctx():
    ctx = context(use_graph=True)
    ctx.finalize()


def test_ctx2():
    X = np.ones(16, dtype=np.float32)
    Y = np.ones(16, dtype=np.float32)
    Z = np.ones(16, dtype=np.float32)

    ctx = context()
    lX = ctx.logical_data(X)
    lY = ctx.logical_data(Y)
    lZ = ctx.logical_data(Z)

    t = ctx.task(rw(lX))
    t.start()
    t.end()

    t2 = ctx.task(read(lX), rw(lY))
    t2.start()
    t2.end()

    t3 = ctx.task(read(lX), rw(lZ))
    t3.start()
    t3.end()

    t4 = ctx.task(read(lY), rw(lZ))
    t4.start()
    t4.end()

    del ctx


def test_ctx3():
    X = np.ones(16, dtype=np.float32)
    Y = np.ones(16, dtype=np.float32)
    Z = np.ones(16, dtype=np.float32)

    ctx = context()
    lX = ctx.logical_data(X)
    lY = ctx.logical_data(Y)
    lZ = ctx.logical_data(Z)

    with ctx.task(rw(lX)):
        pass

    with ctx.task(read(lX), rw(lY)):
        pass

    with ctx.task(read(lX), rw(lZ)):
        pass

    with ctx.task(read(lY), rw(lZ)):
        pass

    del ctx


if __name__ == "__main__":
    print("Running CUDASTF examples...")
    test_ctx3()
