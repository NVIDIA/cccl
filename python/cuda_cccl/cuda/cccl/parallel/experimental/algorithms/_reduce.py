# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations  # TODO: required for Python 3.7 docs env

from typing import Callable, Union

import numba
import numpy as np

from .. import _bindings
from .. import _cccl_interop as cccl
from .._caching import CachableFunction, cache_with_key
from .._cccl_interop import call_build, set_cccl_iterator_state, to_cccl_value_state
from .._utils import protocols
from .._utils.protocols import get_data_pointer, validate_and_get_stream
from .._utils.temp_storage_buffer import TempStorageBuffer
from ..iterators._iterators import IteratorBase
from ..op import OpKind
from ..typing import DeviceArrayLike, GpuStruct


class _Reduce:
    __slots__ = [
        "d_in_cccl",
        "d_out_cccl",
        "h_init_cccl",
        "op_wrapper",
        "build_result",
    ]

    # TODO: constructor shouldn't require concrete `d_in`, `d_out`:
    def __init__(
        self,
        d_in: DeviceArrayLike | IteratorBase,
        d_out: DeviceArrayLike,
        op: Callable | OpKind,
        h_init: np.ndarray | GpuStruct,
    ):
        self.d_in_cccl = cccl.to_cccl_iter(d_in)
        self.d_out_cccl = cccl.to_cccl_iter(d_out)
        self.h_init_cccl = cccl.to_cccl_value(h_init)
        if isinstance(h_init, np.ndarray):
            value_type = numba.from_dtype(h_init.dtype)
        else:
            value_type = numba.typeof(h_init)

        # For well-known operations, we don't need a signature
        if isinstance(op, OpKind):
            self.op_wrapper = cccl.to_cccl_op(op, None)
        else:
            self.op_wrapper = cccl.to_cccl_op(op, value_type(value_type, value_type))
        self.build_result = call_build(
            _bindings.DeviceReduceBuildResult,
            self.d_in_cccl,
            self.d_out_cccl,
            self.op_wrapper,
            self.h_init_cccl,
        )

    def __call__(
        self,
        temp_storage,
        d_in,
        d_out,
        num_items: int,
        h_init: np.ndarray | GpuStruct,
        stream=None,
    ):
        set_cccl_iterator_state(self.d_in_cccl, d_in)
        set_cccl_iterator_state(self.d_out_cccl, d_out)

        self.h_init_cccl.state = to_cccl_value_state(h_init)

        stream_handle = validate_and_get_stream(stream)

        if temp_storage is None:
            temp_storage_bytes = 0
            d_temp_storage = 0
        else:
            temp_storage_bytes = temp_storage.nbytes
            d_temp_storage = get_data_pointer(temp_storage)

        temp_storage_bytes = self.build_result.compute(
            d_temp_storage,
            temp_storage_bytes,
            self.d_in_cccl,
            self.d_out_cccl,
            num_items,
            self.op_wrapper,
            self.h_init_cccl,
            stream_handle,
        )
        return temp_storage_bytes


def make_cache_key(
    d_in: DeviceArrayLike | IteratorBase,
    d_out: DeviceArrayLike,
    op: Callable | OpKind,
    h_init: np.ndarray,
):
    d_in_key = (
        d_in.kind if isinstance(d_in, IteratorBase) else protocols.get_dtype(d_in)
    )
    d_out_key = protocols.get_dtype(d_out)
    # Handle well-known operations differently
    op_key: Union[tuple[str, int], CachableFunction]
    if isinstance(op, OpKind):
        op_key = (op.name, op.value)
    else:
        op_key = CachableFunction(op)
    h_init_key = h_init.dtype
    return (d_in_key, d_out_key, op_key, h_init_key)


# TODO Figure out `sum` without operator and initial value
# TODO Accept stream
@cache_with_key(make_cache_key)
def make_reduce_into(
    d_in: DeviceArrayLike | IteratorBase,
    d_out: DeviceArrayLike,
    op: Callable | OpKind,
    h_init: np.ndarray,
):
    """Computes a device-wide reduction using the specified binary ``op`` and initial value ``init``.

    Example:
        Below, ``reduce_into`` is used to compute the minimum value of a sequence of integers.

        .. literalinclude:: ../../python/cuda_cccl/tests/parallel/test_reduce_api.py
            :language: python
            :dedent:
            :start-after: example-begin reduce-min
            :end-before: example-end reduce-min

    Args:
        d_in: Device array or iterator containing the input sequence of data items
        d_out: Device array (of size 1) that will store the result of the reduction
        op: Callable or OpKind representing the binary operator to apply
        init: Numpy array storing initial value of the reduction

    Returns:
        A callable object that can be used to perform the reduction
    """
    return _Reduce(d_in, d_out, op, h_init)


def reduce_into(
    d_in: DeviceArrayLike | IteratorBase,
    d_out: DeviceArrayLike,
    op: Callable | OpKind,
    num_items: int,
    h_init: np.ndarray | GpuStruct,
    stream=None,
):
    """
    Performs device-wide reduction.

    This function automatically handles temporary storage allocation and execution.

    Args:
        d_in: Device array or iterator containing the input sequence of data items
        d_out: Device array to store the result of the reduction
        op: Binary reduction operator
        num_items: Number of items to reduce
        h_init: Initial value for the reduction
        stream: CUDA stream for the operation (optional)
    """
    reducer = make_reduce_into(d_in, d_out, op, h_init)
    tmp_storage_bytes = reducer(None, d_in, d_out, num_items, h_init, stream)
    tmp_storage = TempStorageBuffer(tmp_storage_bytes, stream)
    reducer(tmp_storage, d_in, d_out, num_items, h_init, stream)
