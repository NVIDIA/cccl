# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Small CUDA Driver allocations for CUTLASS numerical tests."""

from contextlib import contextmanager

import cutlass
import numpy as np
from cutlass import cute
from cutlass.cute.runtime import make_ptr

from cuda.bindings import driver

NUMPY_DTYPES = (
    np.int8,
    np.uint8,
    np.int16,
    np.uint16,
    np.int32,
    np.uint32,
    np.int64,
    np.uint64,
    np.float32,
    np.float64,
)
_CUTLASS_DTYPES = dict(
    zip(
        map(np.dtype, NUMPY_DTYPES),
        (
            cutlass.Int8,
            cutlass.Uint8,
            cutlass.Int16,
            cutlass.Uint16,
            cutlass.Int32,
            cutlass.Uint32,
            cutlass.Int64,
            cutlass.Uint64,
            cutlass.Float32,
            cutlass.Float64,
        ),
    )
)


def check_cuda(result):
    if int(result[0]):
        raise RuntimeError(f"CUDA Driver call failed: {result[0]}")
    return result[1] if len(result) == 2 else result[1:]


def cutlass_dtype(dtype):
    return _CUTLASS_DTYPES[np.dtype(dtype)]


@contextmanager
def device_array(values):
    """Yield a typed pointer; copy the result back before freeing the allocation."""

    values = np.asarray(values)
    if not values.flags.c_contiguous:
        raise ValueError("device_array requires contiguous input")
    cutlass.cuda.initialize_cuda_context()
    allocation = check_cuda(driver.cuMemAlloc(values.nbytes))
    try:
        check_cuda(driver.cuMemcpyHtoD(allocation, values.ctypes.data, values.nbytes))
        pointer = make_ptr(
            cutlass_dtype(values.dtype),
            int(allocation),
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        yield pointer
        check_cuda(driver.cuCtxSynchronize())
        check_cuda(driver.cuMemcpyDtoH(values.ctypes.data, allocation, values.nbytes))
    finally:
        check_cuda(driver.cuMemFree(allocation))


def values_for(dtype, size, *, shift=0):
    dtype = np.dtype(dtype)
    values = (np.arange(size, dtype=np.int64) * 3 + shift) % 97
    if dtype.kind in {"i", "f"}:
        values -= 48
    values = values.astype(dtype)
    if dtype.kind == "u":
        values += dtype.type(1 << (dtype.itemsize * 8 - 1))
    return values
