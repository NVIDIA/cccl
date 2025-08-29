# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


import numba
import numpy as np
import pytest
import torch
from numba import cuda

numba.config.CUDA_ENABLE_PYNVJITLINK = 1
numba.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0

from cuda.cccl.experimental.stf._stf_bindings import (
    context, 
    data_place,
    exec_place,
    read,
    rw,
    write,
)

import torch

def torch_from_cai(obj):
    """
    Convert an object exposing the CUDA Array Interface (__cuda_array_interface__)
    into a torch.Tensor (on GPU). Zero-copy if possible.

    Strategy:
      1. If obj has .to_dlpack(), use it directly.
      2. Otherwise, try to wrap with CuPy (which understands CAI) and then use DLPack.
    """
    # Path 1: direct DLPack (Numba >=0.53, some other libs)
    if hasattr(obj, "to_dlpack"):
        return torch.utils.dlpack.from_dlpack(obj.to_dlpack())

    # Path 2: via CuPy bridge
    try:
        import cupy as cp
    except ImportError as e:
        raise RuntimeError(
            "Object does not support .to_dlpack and CuPy is not installed. "
            "Cannot convert __cuda_array_interface__ to torch.Tensor."
        ) from e

    # CuPy knows how to wrap CAI
    cupy_arr = cp.asarray(obj)
    return torch.utils.dlpack.from_dlpack(cupy_arr.toDlpack())


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
        sptr = t.stream_ptr()
        torch_stream = torch.cuda.ExternalStream(sptr, device=torch.device("cuda:0"))
        with torch.cuda.stream(torch_stream):
            dX = cuda.from_cuda_array_interface(t.get_arg_cai(0), owner=None, sync=False)
            tX = torch_from_cai(dX)
            # same as tX =t.get_arg_as_tensor(0) 
            tX = tX*2

    with ctx.task(lX.read(), lY.write()) as t:
        sptr = t.stream_ptr()
        torch_stream = torch.cuda.ExternalStream(sptr, device=torch.device("cuda:0"))
        with torch.cuda.stream(torch_stream):
            tX =t.get_arg_as_tensor(0) 
            tY =t.get_arg_as_tensor(1) 
            tY = tX*2

    with ctx.task(lX.read(), lZ.write()) as t:
        sptr = t.stream_ptr()
        torch_stream = torch.cuda.ExternalStream(sptr, device=torch.device("cuda:0"))
        with torch.cuda.stream(torch_stream):
            tX, tY = t.tensor_arguments()
            tZ = tX*4 + 1

    with ctx.task(lY.read(), lZ.rw()) as t:
        sptr = t.stream_ptr()
        torch_stream = torch.cuda.ExternalStream(sptr, device=torch.device("cuda:0"))
        with torch.cuda.stream(torch_stream):
            tX, tZ = t.tensor_arguments()
            tZ = tY*2 - 3

    ctx.finalize()

if __name__ == "__main__":
    print("Running CUDASTF examples...")
    test_pytorch()

