# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Callable

from .. import _bindings
from .. import _cccl_interop as cccl
from .._caching import cache_with_registered_key_functions
from .._cccl_interop import set_cccl_iterator_state
from .._utils import protocols
from ..iterators._iterators import IteratorBase
from ..op import OpAdapter, OpKind, make_op_adapter
from ..typing import DeviceArrayLike


class _UnaryTransform:
    __slots__ = ["d_in_cccl", "d_out_cccl", "op_cccl", "build_result"]

    def __init__(
        self,
        d_in: DeviceArrayLike | IteratorBase,
        d_out: DeviceArrayLike | IteratorBase,
        op: OpAdapter,
    ):
        self.d_in_cccl = cccl.to_cccl_input_iter(d_in)
        self.d_out_cccl = cccl.to_cccl_output_iter(d_out)

        # Compile the op with input/output types
        in_type = cccl.get_value_type(d_in)
        out_type = cccl.get_value_type(d_out)
        self.op_cccl = op.compile((in_type,), out_type)

        self.build_result = cccl.call_build(
            _bindings.DeviceUnaryTransform,
            self.d_in_cccl,
            self.d_out_cccl,
            self.op_cccl,
        )

    def __call__(
        self,
        d_in,
        d_out,
        op: Callable | OpAdapter,
        num_items: int,
        stream=None,
    ):
        op_adapter = make_op_adapter(op)

        set_cccl_iterator_state(self.d_in_cccl, d_in)
        set_cccl_iterator_state(self.d_out_cccl, d_out)
        op_adapter.update_op_state(self.op_cccl)

        stream_handle = protocols.validate_and_get_stream(stream)
        self.build_result.compute(
            self.d_in_cccl,
            self.d_out_cccl,
            num_items,
            self.op_cccl,
            stream_handle,
        )
        return None


class _BinaryTransform:
    __slots__ = [
        "d_in1_cccl",
        "d_in2_cccl",
        "d_out_cccl",
        "op_cccl",
        "build_result",
    ]

    def __init__(
        self,
        d_in1: DeviceArrayLike | IteratorBase,
        d_in2: DeviceArrayLike | IteratorBase,
        d_out: DeviceArrayLike | IteratorBase,
        op: OpAdapter,
    ):
        self.d_in1_cccl = cccl.to_cccl_input_iter(d_in1)
        self.d_in2_cccl = cccl.to_cccl_input_iter(d_in2)
        self.d_out_cccl = cccl.to_cccl_output_iter(d_out)

        # Compile the op with input/output types
        in1_type = cccl.get_value_type(d_in1)
        in2_type = cccl.get_value_type(d_in2)
        out_type = cccl.get_value_type(d_out)
        self.op_cccl = op.compile((in1_type, in2_type), out_type)

        self.build_result = cccl.call_build(
            _bindings.DeviceBinaryTransform,
            self.d_in1_cccl,
            self.d_in2_cccl,
            self.d_out_cccl,
            self.op_cccl,
        )

    def __call__(
        self,
        d_in1,
        d_in2,
        d_out,
        op: Callable | OpAdapter,
        num_items: int,
        stream=None,
    ):
        set_cccl_iterator_state(self.d_in1_cccl, d_in1)
        set_cccl_iterator_state(self.d_in2_cccl, d_in2)
        set_cccl_iterator_state(self.d_out_cccl, d_out)

        op_adapter = make_op_adapter(op)
        op_adapter.update_op_state(self.op_cccl)

        stream_handle = protocols.validate_and_get_stream(stream)
        self.build_result.compute(
            self.d_in1_cccl,
            self.d_in2_cccl,
            self.d_out_cccl,
            num_items,
            self.op_cccl,
            stream_handle,
        )
        return None


@cache_with_registered_key_functions
def make_unary_transform(
    d_in: DeviceArrayLike | IteratorBase,
    d_out: DeviceArrayLike | IteratorBase,
    op: Callable | OpKind,
):
    """
    Create a unary transform object that can be called to apply a transformation
    to each element of the input according to the unary operation ``op``.

    This is the object-oriented API that allows explicit control over temporary
    storage allocation. For simpler usage, consider using :func:`unary_transform`.

    Example:
        .. literalinclude:: ../../python/cuda_cccl/tests/compute/examples/transform/unary_transform_object.py
           :language: python
           :start-after: # example-begin


    Args:
        d_in: Device array or iterator containing the input sequence of data items.
        d_out: Device array or iterator to store the result of the transformation.
        op: Callable or OpKind representing the unary operation to apply to each element of the input.

    Returns:
        A callable object that performs the transformation.
    """
    op_adapter = make_op_adapter(op)
    return _UnaryTransform(d_in, d_out, op_adapter)


@cache_with_registered_key_functions
def make_binary_transform(
    d_in1: DeviceArrayLike | IteratorBase,
    d_in2: DeviceArrayLike | IteratorBase,
    d_out: DeviceArrayLike | IteratorBase,
    op: Callable | OpKind,
):
    """
    Create a binary transform object that can be called to apply a transformation
    to the given pair of input sequences according to the binary operation ``op``.

    This is the object-oriented API that allows explicit control over temporary
    storage allocation. For simpler usage, consider using :func:`binary_transform`.

    Example:
        .. literalinclude:: ../../python/cuda_cccl/tests/compute/examples/transform/binary_transform_object.py
           :language: python
           :start-after: # example-begin


    Args:
        d_in1: Device array or iterator containing the first input sequence of data items.
        d_in2: Device array or iterator containing the second input sequence of data items.
        d_out: Device array or iterator to store the result of the transformation.
        op: Callable or OpKind representing the binary operation to apply to each pair of items from the input sequences.

    Returns:
        A callable object that performs the transformation.
    """
    op_adapter = make_op_adapter(op)
    return _BinaryTransform(d_in1, d_in2, d_out, op_adapter)


def unary_transform(
    d_in: DeviceArrayLike | IteratorBase,
    d_out: DeviceArrayLike | IteratorBase,
    op: Callable | OpKind,
    num_items: int,
    stream=None,
):
    """
    Performs device-wide unary transform.

    This function automatically handles temporary storage allocation and execution.

    The ``op`` function can reference device arrays as globals or closures - they will
    be automatically captured as state arrays, enabling stateful operations like counting.

    Example:
        Below, ``unary_transform`` is used to apply a transformation to each element of the input.

        .. literalinclude:: ../../python/cuda_cccl/tests/compute/examples/transform/unary_transform_basic.py
           :language: python
           :start-after: # example-begin

        When working with custom struct types, you need to provide type annotations
        to help with type inference. See the binary transform struct example for reference:

        .. literalinclude:: ../../python/cuda_cccl/tests/compute/examples/struct/struct_transform.py
           :language: python
           :start-after: # example-begin


    Args:
        d_in: Device array or iterator containing the input sequence of data items.
        d_out: Device array or iterator to store the result of the transformation.
        op: Callable or OpKind representing the unary operation to apply to each element of the input.
            Can reference device arrays as globals/closures - they will be automatically captured.
        num_items: Number of items to transform.
        stream: CUDA stream to use for the operation.
    """
    op_adapter = make_op_adapter(op)
    transformer = make_unary_transform(d_in, d_out, op_adapter)
    transformer(d_in, d_out, op_adapter, num_items, stream)


def binary_transform(
    d_in1: DeviceArrayLike | IteratorBase,
    d_in2: DeviceArrayLike | IteratorBase,
    d_out: DeviceArrayLike | IteratorBase,
    op: Callable | OpKind,
    num_items: int,
    stream=None,
):
    """
    Performs device-wide binary transform.

    This function automatically handles temporary storage allocation and execution.

    Example:
        Below, ``binary_transform`` is used to apply a transformation to pairs of elements from two input sequences.

        .. literalinclude:: ../../python/cuda_cccl/tests/compute/examples/transform/binary_transform_basic.py
           :language: python
           :start-after: # example-begin

        When working with custom struct types, you need to provide type annotations
        to help with type inference. See the following example:

        .. literalinclude:: ../../python/cuda_cccl/tests/compute/examples/struct/struct_transform.py
           :language: python
           :start-after: # example-begin


    Args:
        d_in1: Device array or iterator containing the first input sequence of data items.
        d_in2: Device array or iterator containing the second input sequence of data items.
        d_out: Device array or iterator to store the result of the transformation.
        op: Callable or OpKind representing the binary operation to apply to each pair of items from the input sequences.
            Can reference device arrays as globals/closures - they will be automatically captured.
        num_items: Number of items to transform.
        stream: CUDA stream to use for the operation.
    """
    op_adapter = make_op_adapter(op)
    transformer = make_binary_transform(d_in1, d_in2, d_out, op_adapter)
    transformer(d_in1, d_in2, d_out, op_adapter, num_items, stream)
