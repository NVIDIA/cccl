# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Callable

import numpy as np

from .. import _bindings
from .. import _cccl_interop as cccl
from .._caching import cache_with_key
from .._cccl_interop import (
    call_build,
    get_value_type,
    set_cccl_iterator_state,
    to_cccl_value_state,
)
from .._utils import protocols
from .._utils.protocols import get_data_pointer, validate_and_get_stream
from .._utils.temp_storage_buffer import TempStorageBuffer
from ..determinism import Determinism
from ..iterators._iterators import IteratorBase
from ..op import OpAdapter, OpKind, make_op_adapter
from ..typing import DeviceArrayLike, GpuStruct


class _Reduce:
    __slots__ = [
        "d_in_cccl",
        "d_out_cccl",
        "h_init_cccl",
        "op",
        "op_cccl",
        "build_result",
        "device_reduce_fn",
    ]

    # TODO: constructor shouldn't require concrete `d_in`, `d_out`:
    def __init__(
        self,
        d_in: DeviceArrayLike | IteratorBase,
        d_out: DeviceArrayLike | IteratorBase,
        op: OpAdapter,
        h_init: np.ndarray | GpuStruct,
        determinism: Determinism,
    ):
        self.d_in_cccl = cccl.to_cccl_input_iter(d_in)
        self.d_out_cccl = cccl.to_cccl_output_iter(d_out)
        self.h_init_cccl = cccl.to_cccl_value(h_init)

        # Compile the op with value types
        value_type = get_value_type(h_init)
        self.op_cccl = op.compile((value_type, value_type), value_type)

        self.build_result = call_build(
            _bindings.DeviceReduceBuildResult,
            self.d_in_cccl,
            self.d_out_cccl,
            self.op_cccl,
            self.h_init_cccl,
            determinism,
        )

        match determinism:
            case Determinism.RUN_TO_RUN:
                self.device_reduce_fn = self.build_result.compute
            case Determinism.NOT_GUARANTEED:
                self.device_reduce_fn = self.build_result.compute_nondeterministic
            case _:
                raise ValueError(f"Invalid determinism: {determinism}")

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

        temp_storage_bytes = self.device_reduce_fn(
            d_temp_storage,
            temp_storage_bytes,
            self.d_in_cccl,
            self.d_out_cccl,
            num_items,
            self.op_cccl,
            self.h_init_cccl,
            stream_handle,
        )
        return temp_storage_bytes


def _make_cache_key(
    d_in: DeviceArrayLike | IteratorBase,
    d_out: DeviceArrayLike | IteratorBase,
    op: OpAdapter,
    h_init: np.ndarray | GpuStruct,
    **kwargs,
):
    d_in_key = (
        d_in.kind if isinstance(d_in, IteratorBase) else protocols.get_dtype(d_in)
    )
    d_out_key = (
        d_out.kind if isinstance(d_out, IteratorBase) else protocols.get_dtype(d_out)
    )
    h_init_key = h_init.dtype
    determinism = kwargs.get("determinism", Determinism.RUN_TO_RUN)
    return (d_in_key, d_out_key, op.get_cache_key(), h_init_key, determinism)


@cache_with_key(_make_cache_key)
def _make_reduce_into_cached(
    d_in: DeviceArrayLike | IteratorBase,
    d_out: DeviceArrayLike | IteratorBase,
    op: OpAdapter,
    h_init: np.ndarray | GpuStruct,
    **kwargs,
):
    """Internal cached factory for _Reduce."""
    return _Reduce(
        d_in, d_out, op, h_init, kwargs.get("determinism", Determinism.RUN_TO_RUN)
    )


# TODO Figure out `sum` without operator and initial value
# TODO Accept stream
def make_reduce_into(
    d_in: DeviceArrayLike | IteratorBase,
    d_out: DeviceArrayLike | IteratorBase,
    op: Callable | OpKind,
    h_init: np.ndarray | GpuStruct,
    **kwargs,
):
    """Computes a device-wide reduction using the specified binary ``op`` and initial value ``init``.

    Example:
        Below, ``make_reduce_into`` is used to create a reduction object that can be reused.

        .. literalinclude:: ../../python/cuda_cccl/tests/compute/examples/reduction/reduce_object.py
            :language: python
            :start-after: # example-begin


    Args:
        d_in: Device array or iterator containing the input sequence of data items
        d_out: Device array (of size 1) that will store the result of the reduction
        op: Callable or OpKind representing the binary operator to apply
        init: Numpy array storing initial value of the reduction

    Returns:
        A callable object that can be used to perform the reduction
    """
    op_adapter = make_op_adapter(op)
    return _make_reduce_into_cached(d_in, d_out, op_adapter, h_init, **kwargs)


def reduce_into(
    d_in: DeviceArrayLike | IteratorBase,
    d_out: DeviceArrayLike | IteratorBase,
    op: Callable | OpKind,
    num_items: int,
    h_init: np.ndarray | GpuStruct,
    stream=None,
    **kwargs,
):
    """
    Performs device-wide reduction.

    This function automatically handles temporary storage allocation and execution.

    Example:
        Below, ``reduce_into`` is used to compute the sum of a sequence of integers.

        .. literalinclude:: ../../python/cuda_cccl/tests/compute/examples/reduction/sum_reduction.py
            :language: python
            :start-after: # example-begin

    Args:
        d_in: Device array or iterator containing the input sequence of data items
        d_out: Device array to store the result of the reduction
        op: Binary reduction operator
        num_items: Number of items to reduce
        h_init: Initial value for the reduction
        stream: CUDA stream for the operation (optional)
    """
    reducer = make_reduce_into(d_in, d_out, op, h_init, **kwargs)
    tmp_storage_bytes = reducer(None, d_in, d_out, num_items, h_init, stream)
    tmp_storage = TempStorageBuffer(tmp_storage_bytes, stream)
    reducer(tmp_storage, d_in, d_out, num_items, h_init, stream)
