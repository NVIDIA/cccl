# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np

import cuda.stf as stf


def test_ctx():
    ctx = stf.context()
    del ctx


def test_graph_ctx():
    ctx = stf.context(use_graph=True)
    ctx.finalize()


def test_ctx2():
    X = np.ones(16, dtype=np.float32)
    Y = np.ones(16, dtype=np.float32)
    Z = np.ones(16, dtype=np.float32)

    ctx = stf.context()
    lX = ctx.logical_data(X)
    lY = ctx.logical_data(Y)
    lZ = ctx.logical_data(Z)

    t = ctx.task(lX.rw())
    t.start()
    t.end()

    t2 = ctx.task(lX.read(), lY.rw())
    t2.start()
    t2.end()

    t3 = ctx.task(lX.read(), lZ.rw())
    t3.start()
    t3.end()

    t4 = ctx.task(lY.read(), lZ.rw())
    t4.start()
    t4.end()

    del ctx


def test_ctx3():
    X = np.ones(16, dtype=np.float32)
    Y = np.ones(16, dtype=np.float32)
    Z = np.ones(16, dtype=np.float32)

    ctx = stf.context()
    lX = ctx.logical_data(X)
    lY = ctx.logical_data(Y)
    lZ = ctx.logical_data(Z)

    with ctx.task(lX.rw()):
        pass

    with ctx.task(lX.read(), lY.rw()):
        pass

    with ctx.task(lX.read(), lZ.rw()):
        pass

    with ctx.task(lY.read(), lZ.rw()):
        pass

    del ctx


if __name__ == "__main__":
    test_ctx3()
