# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Callable

import numpy as np

from .. import _bindings
from .. import _cccl_interop as cccl
from .._caching import cache_with_registered_key_functions
from .._cccl_interop import (
    call_build,
    get_value_type,
    set_cccl_iterator_state,
    to_cccl_value_state,
)
from .._utils.protocols import (
    get_data_pointer,
    validate_and_get_stream,
)
from .._utils.temp_storage_buffer import TempStorageBuffer
from ..iterators._iterators import IteratorBase
from ..op import OpAdapter, OpKind, make_op_adapter
from ..typing import DeviceArrayLike, GpuStruct


class _SegmentedReduce:
    __slots__ = [
        "build_result",
        "d_in_cccl",
        "d_out_cccl",
        "start_offsets_in_cccl",
        "end_offsets_in_cccl",
        "h_init_cccl",
        "op_cccl",
    ]

    def __init__(
        self,
        d_in: DeviceArrayLike | IteratorBase,
        d_out: DeviceArrayLike | IteratorBase,
        start_offsets_in: DeviceArrayLike | IteratorBase,
        end_offsets_in: DeviceArrayLike | IteratorBase,
        op: OpAdapter,
        h_init: np.ndarray | GpuStruct,
    ):
        self.d_in_cccl = cccl.to_cccl_input_iter(d_in)
        self.d_out_cccl = cccl.to_cccl_output_iter(d_out)
        self.start_offsets_in_cccl = cccl.to_cccl_input_iter(start_offsets_in)
        self.end_offsets_in_cccl = cccl.to_cccl_input_iter(end_offsets_in)
        self.h_init_cccl = cccl.to_cccl_value(h_init)

        # Compile the op with value types
        value_type = get_value_type(h_init)

        self.op_cccl = op.compile((value_type, value_type), value_type)

        self.build_result = call_build(
            _bindings.DeviceSegmentedReduceBuildResult,
            self.d_in_cccl,
            self.d_out_cccl,
            self.start_offsets_in_cccl,
            self.end_offsets_in_cccl,
            self.op_cccl,
            self.h_init_cccl,
        )

    def __call__(
        self,
        temp_storage,
        d_in,
        d_out,
        op: Callable | OpAdapter,
        num_segments: int,
        start_offsets_in,
        end_offsets_in,
        h_init,
        stream=None,
    ):
        if num_segments > np.iinfo(np.int32).max:
            raise RuntimeError(
                "Segmented sort does not currently support more than 2^31-1 segments."
            )
        set_cccl_iterator_state(self.d_in_cccl, d_in)
        set_cccl_iterator_state(self.d_out_cccl, d_out)
        set_cccl_iterator_state(self.start_offsets_in_cccl, start_offsets_in)
        set_cccl_iterator_state(self.end_offsets_in_cccl, end_offsets_in)

        # Update op state for stateful ops
        op_adapter = make_op_adapter(op)
        op_adapter.update_op_state(self.op_cccl)

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
            num_segments,
            self.start_offsets_in_cccl,
            self.end_offsets_in_cccl,
            self.op_cccl,
            self.h_init_cccl,
            stream_handle,
        )
        return temp_storage_bytes


@cache_with_registered_key_functions
def make_segmented_reduce(
    d_in: DeviceArrayLike | IteratorBase,
    d_out: DeviceArrayLike | IteratorBase,
    start_offsets_in: DeviceArrayLike | IteratorBase,
    end_offsets_in: DeviceArrayLike | IteratorBase,
    op: Callable | OpKind,
    h_init: np.ndarray | GpuStruct,
):
    """Computes a device-wide segmented reduction using the specified binary ``op`` and initial value ``init``.

    Example:
        Below, ``make_segmented_reduce`` is used to create a segmented reduction object that can be reused.

        .. literalinclude:: ../../python/cuda_cccl/tests/compute/examples/segmented/segmented_reduce_object.py
            :language: python
            :start-after: # example-begin


    Args:
        d_in: Device array or iterator containing the input sequence of data items
        d_out: Device array that will store the result of the reduction
        start_offsets_in: Device array or iterator containing offsets to start of segments
        end_offsets_in: Device array or iterator containing offsets to end of segments
        op: Callable or OpKind representing the binary operator to apply
        init: Numpy array storing initial value of the reduction

    Returns:
        A callable object that can be used to perform the reduction
    """
    op_adapter = make_op_adapter(op)
    return _SegmentedReduce(
        d_in, d_out, start_offsets_in, end_offsets_in, op_adapter, h_init
    )


def segmented_reduce(
    d_in: DeviceArrayLike | IteratorBase,
    d_out: DeviceArrayLike | IteratorBase,
    start_offsets_in: DeviceArrayLike | IteratorBase,
    end_offsets_in: DeviceArrayLike | IteratorBase,
    op: Callable | OpKind,
    h_init: np.ndarray | GpuStruct,
    num_segments: int,
    stream=None,
):
    """
    Performs device-wide segmented reduction.

    This function automatically handles temporary storage allocation and execution.

    Example:
        Below, ``segmented_reduce`` is used to compute the minimum value of segments in a sequence of integers.

        .. literalinclude:: ../../python/cuda_cccl/tests/compute/examples/segmented/segmented_reduce_basic.py
            :language: python
            :start-after: # example-begin


    Args:
        d_in: Device array or iterator containing the input sequence of data items
        d_out: Device array to store the result of the reduction for each segment
        start_offsets_in: Device array or iterator containing the sequence of beginning offsets
        end_offsets_in: Device array or iterator containing the sequence of ending offsets
        op: Binary reduction operator
        h_init: Initial value for the reduction
        num_segments: Number of segments to reduce
        stream: CUDA stream for the operation (optional)
    """
    reducer = make_segmented_reduce(
        d_in, d_out, start_offsets_in, end_offsets_in, op, h_init
    )
    tmp_storage_bytes = reducer(
        None,
        d_in,
        d_out,
        op,
        num_segments,
        start_offsets_in,
        end_offsets_in,
        h_init,
        stream,
    )
    tmp_storage = TempStorageBuffer(tmp_storage_bytes, stream)
    reducer(
        tmp_storage,
        d_in,
        d_out,
        op,
        num_segments,
        start_offsets_in,
        end_offsets_in,
        h_init,
        stream,
    )
