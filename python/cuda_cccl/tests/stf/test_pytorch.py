# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


import numba
import numpy as np
import pytest

torch = pytest.importorskip("torch")

numba.config.CUDA_ENABLE_PYNVJITLINK = 1
numba.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0

from cuda.cccl.experimental.stf._stf_bindings import (
    context,
    rw,
)


def test_pytorch():
    n = 1024 * 1024
    X = np.ones(n, dtype=np.float32)
    Y = np.ones(n, dtype=np.float32)
    Z = np.ones(n, dtype=np.float32)

    ctx = context()
    lX = ctx.logical_data(X)
    lY = ctx.logical_data(Y)
    lZ = ctx.logical_data(Z)

    with ctx.task(rw(lX)) as t:
        torch_stream = torch.cuda.ExternalStream(t.stream_ptr())
        with torch.cuda.stream(torch_stream):
            tX = t.tensor_arguments()
            tX = tX * 2

    with ctx.task(lX.read(), lY.write()) as t:
        torch_stream = torch.cuda.ExternalStream(t.stream_ptr())
        with torch.cuda.stream(torch_stream):
            tX = t.get_arg_as_tensor(0)
            tY = t.get_arg_as_tensor(1)
            tY = tX * 2

    with (
        ctx.task(lX.read(), lZ.write()) as t,
        torch.cuda.stream(torch.cuda.ExternalStream(t.stream_ptr())),
    ):
        tX, tY = t.tensor_arguments()
        tZ = tX * 4 + 1

    with (
        ctx.task(lY.read(), lZ.rw()) as t,
        torch.cuda.stream(torch.cuda.ExternalStream(t.stream_ptr())),
    ):
        tX, tZ = t.tensor_arguments()
        tZ = tY * 2 - 3

    ctx.finalize()


if __name__ == "__main__":
    print("Running CUDASTF examples...")
    test_pytorch()
