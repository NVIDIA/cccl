# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from cuda.cccl.experimental.stf._stf_bindings_impl import logical_data, context
import ctypes
import numpy as np

def test_ctx():
    ctx = _stf_bindings_impl.context()
    del ctx

def test_ctx2():
    X = np.ones(16, dtype=np.float32)
    Y = np.ones(16, dtype=np.float32)

    ctx = _stf_bindings_impl.context()
    lX = ctx.logical_data(X)
    lY = ctx.logical_data(Y)
    del ctx
