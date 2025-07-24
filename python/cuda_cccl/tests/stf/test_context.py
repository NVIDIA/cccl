# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from cuda.cccl.experimental.stf._stf_bindings_impl import logical_data, context, AccessMode, read, rw, write
import ctypes
import numpy as np

def test_ctx():
    ctx = _stf_bindings_impl.context()
    del ctx

def test_ctx2():
    X = np.ones(16, dtype=np.float32)
    Y = np.ones(16, dtype=np.float32)

    ctx = context()
    lX = ctx.logical_data(X)
    lY = ctx.logical_data(Y)

    t = ctx.task(read(lX), rw(lY))
    t.start()
    t.end()

    t2 = ctx.task()
    t2.add_dep(rw(lX))
    t2.start()
    t2.end()

    del ctx

if __name__ == "__main__":
    print("Running CUDASTF examples...")
    test_ctx2()
