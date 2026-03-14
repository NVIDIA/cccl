# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Fill / init for STF logical data. Uses cuda.core.Buffer.fill for 1/2/4-byte;
single fallback (CuPy) for 8-byte.
"""

import numpy as np

from cuda.core import Buffer, Stream


def init_logical_data(ctx, ld, value, data_place=None, exec_place=None):
    """
    Initialize a logical data with a constant value.

    Uses cuda.core.Buffer.fill for 1/2/4-byte element types. For 8-byte types
    (e.g. float64, int64) uses CuPy if available; otherwise raises.

    Parameters
    ----------
    ctx : context
        STF context
    ld : logical_data
        Logical data to initialize
    value : scalar
        Value to fill the array with
    data_place : data_place, optional
        Data place for the initialization task
    exec_place : exec_place, optional
        Execution place for the fill operation

    Raises
    ------
    ImportError
        If dtype is 8-byte and CuPy is not installed.
    """
    dep_arg = ld.write(data_place) if data_place else ld.write()

    task_args = []
    if exec_place is not None:
        task_args.append(exec_place)
    task_args.append(dep_arg)

    with ctx.task(*task_args) as t:
        ld_index = 1 if exec_place is not None else 0
        cai = t.get_arg_cai(ld_index)
        ptr = cai["data"][0]
        shape = tuple(cai["shape"])
        dtype = np.dtype(cai["typestr"])
        size = int(np.prod(shape)) * dtype.itemsize

        core_stream = Stream.from_handle(t.stream_ptr())
        buf = Buffer.from_handle(ptr, size, owner=None)

        if dtype.itemsize in (1, 2, 4):
            if value == 0 or value == 0.0:
                fill_val = 0
            else:
                fill_val = np.array([value], dtype=dtype).tobytes()
            buf.fill(fill_val, stream=core_stream)
        else:
            # 8-byte: single fallback via CuPy
            _fill_8byte_cupy(shape, dtype, value, ptr, size, t.stream_ptr())


def _fill_8byte_cupy(shape, dtype, value, ptr, size, stream_ptr):
    """Fill 8-byte buffer using CuPy. Raises ImportError if CuPy not available."""
    try:
        import cupy as cp
    except ImportError:
        raise ImportError(
            "Fill for 8-byte dtypes (e.g. float64) requires CuPy. "
            "Install CuPy or use a 1/2/4-byte dtype (e.g. np.float32)."
        ) from None

    mem = cp.cuda.UnownedMemory(ptr, size, owner=None)
    memptr = cp.cuda.MemoryPointer(mem, 0)
    arr = cp.ndarray(shape, dtype=dtype, memptr=memptr)
    with cp.cuda.ExternalStream(stream_ptr):
        arr.fill(value)
