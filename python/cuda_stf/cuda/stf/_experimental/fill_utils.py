# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Fill / init for STF logical data.

Uses ``cuda.core.Buffer.fill`` for 1/2/4-byte element types and CUDA-driver
strided 32-bit memsets for 8-byte types, so no optional third-party package
(e.g. CuPy) is required for any supported element size.
"""

import numpy as np

from cuda.core import Buffer, Stream


def init_logical_data(ctx, ld, value, data_place=None, exec_place=None):
    """
    Initialize a logical data with a constant value.

    Uses ``cuda.core.Buffer.fill`` for 1/2/4-byte element types and a pair of
    CUDA-driver strided 32-bit memsets for 8-byte types (e.g. float64, int64).
    All fills are enqueued on the task's stream, so they are correctly ordered
    with the rest of the task's work and require no host synchronization.

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
    ValueError
        If the element type has an unsupported size (not 1, 2, 4, or 8 bytes)
        and ``value`` is nonzero.
    """
    dep_arg = ld.write(data_place) if data_place else ld.write()

    task_args = []
    if exec_place is not None:
        task_args.append(exec_place)
    task_args.append(dep_arg)

    with ctx.task(*task_args) as t:
        # exec_place configures the task itself; it does not consume a dependency slot.
        cai = t.get_arg_cai(0)
        ptr = cai["data"][0]
        shape = tuple(cai["shape"])
        dtype = np.dtype(cai["typestr"])
        # An empty shape () is a 0-d scalar, i.e. exactly one element (np.prod
        # of an empty product is 1); it must not be treated as zero elements.
        count = int(np.prod(shape))
        size = count * dtype.itemsize

        if count == 0 or size == 0:
            return

        stream_ptr = t.stream_ptr()
        core_stream = Stream.from_handle(stream_ptr)
        buf = Buffer.from_handle(ptr, size, owner=None)

        # A bytewise zero fill is valid for any numeric dtype, including 8-byte
        # types that cannot use cuda.core's nonzero fill patterns.
        if value == 0 or value == 0.0:
            buf.fill(0, stream=core_stream)
        elif dtype.itemsize in (1, 2, 4):
            fill_val = np.array([value], dtype=dtype).tobytes()
            buf.fill(fill_val, stream=core_stream)
        elif dtype.itemsize == 8:
            _fill_8byte_driver(dtype, value, ptr, count, stream_ptr)
        else:
            raise ValueError(
                f"cannot fill dtype {dtype!r} (itemsize {dtype.itemsize}) with a "
                "nonzero value; only 1/2/4/8-byte element types are supported"
            )


def _fill_8byte_driver(dtype, value, ptr, count, stream_ptr):
    """Fill ``count`` 8-byte elements at ``ptr`` with ``value`` on ``stream_ptr``.

    ``cuMemsetD*32`` only fills 32-bit patterns, so an arbitrary 8-byte value
    is written as two strided 32-bit memsets (low then high half), treating the
    buffer as ``count`` rows of one 32-bit word with an 8-byte row pitch.
    """
    from cuda.bindings import driver

    raw = np.array([value], dtype=dtype).tobytes()  # exactly 8 bytes
    low = int.from_bytes(raw[0:4], "little")
    high = int.from_bytes(raw[4:8], "little")

    # Row 0..count-1: word at row offset 0 gets the low half, offset 4 the high.
    _memset_d2d32(driver, int(ptr), 8, low, count, stream_ptr)
    _memset_d2d32(driver, int(ptr) + 4, 8, high, count, stream_ptr)


def _memset_d2d32(driver, dst, pitch, value, height, stream_ptr):
    """cuMemsetD2D32Async wrapper: fill ``height`` rows of one 32-bit word."""
    (err,) = driver.cuMemsetD2D32Async(dst, pitch, value, 1, height, stream_ptr)
    if int(err) != 0:
        raise RuntimeError(f"cuMemsetD2D32Async failed with error code {int(err)}")
