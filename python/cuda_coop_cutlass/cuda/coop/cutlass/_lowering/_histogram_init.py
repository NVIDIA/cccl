# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Validate shared histogram counters and emit cooperative initialization."""

from numbers import Integral

import cutlass
from cutlass import cute

from .._compiler._launch import current_kernel_launch_facts
from .._planning.group import resolve_thread_group
from .._temp_storage import TempStorage
from ._histogram import _COUNTERS, _resolve_type


def counter_extent(counters):
    """Return the static bin count for a directly addressable C++ array."""

    message = (
        "histogram counters must be a contiguous, naturally aligned, "
        "one-dimensional shared-memory tensor with a positive static size "
        "and 32- or 64-bit integer elements"
    )
    if not isinstance(counters, cute.Tensor) or not isinstance(
        counters.layout, cute.Layout
    ):
        raise TypeError(message)
    pointer = counters.iterator
    if (
        not isinstance(pointer, cute.Pointer)
        or pointer.memspace != cute.AddressSpace.smem
        or pointer.type.is_swizzled
        or counters.element_type not in _COUNTERS
        or pointer.alignment < counters.element_type.width // 8
    ):
        raise TypeError(message)
    shape, stride = counters.shape, counters.stride
    if isinstance(shape, tuple):
        if len(shape) != 1:
            raise TypeError(message)
        shape, stride = shape[0], stride[0]
    if not isinstance(shape, Integral) or not 0 < shape <= (1 << 31) - 1:
        raise ValueError(message)
    # CuTe canonicalizes a singleton stride to zero.
    if not isinstance(stride, Integral) or not (
        stride == 1 or (shape == 1 and stride == 0)
    ):
        raise TypeError(message)
    return int(shape)


@cute.jit
def _clear(
    counters: cute.Tensor, bins: cutlass.Constexpr, width: cutlass.Constexpr
):
    rank = cute.arch.thread_idx()[0]
    for i in cutlass.range(rank, bins, width):
        counters[i] = counters.element_type(0)
    cute.arch.sync_threads()


def initialize_counters(group, counters, bins, counter_dtype):
    """Clear caller-owned counters or reserve a typed private slice first."""

    launch = current_kernel_launch_facts()
    resolve_thread_group(group, launch).require_supported()
    width, y, z = launch.exact_block_dim
    if y != 1 or z != 1:
        raise ValueError("histogram_init supports only one-dimensional blocks")
    if counters is None:
        if bins > (1 << 31) - 1:
            raise ValueError("histogram_init bins must fit a positive int32")
        dtype = _resolve_type(
            cutlass.Int32 if counter_dtype is None else counter_dtype,
            allowed=_COUNTERS,
            feature="counter_dtype",
        )
        counters = TempStorage(sharing="exclusive", auto_sync=False).reserve(
            bins, dtype
        )
    else:
        bins = counter_extent(counters)
    _clear(counters, bins, width)
    return counters
