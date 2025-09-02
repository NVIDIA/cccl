# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Callable, Union

from .. import _bindings
from .. import _cccl_interop as cccl
from .._caching import CachableFunction, cache_with_key
from .._cccl_interop import set_cccl_iterator_state
from .._utils import protocols
from ..iterators._iterators import IteratorBase
from ..numba_utils import get_inferred_return_type
from ..op import OpKind
from ..typing import DeviceArrayLike


class _UnaryTransform:
    __slots__ = [
        "d_in_cccl",
        "d_out_cccl",
        "op_wrapper",
        "build_result",
    ]

    def __init__(
        self,
        d_in: DeviceArrayLike | IteratorBase,
        d_out: DeviceArrayLike | IteratorBase,
        op: Callable | OpKind,
    ):
        self.d_in_cccl = cccl.to_cccl_iter(d_in)
        self.d_out_cccl = cccl.to_cccl_iter(d_out)
        in_value_type = cccl.get_value_type(d_in)
        out_value_type = cccl.get_value_type(d_out)

        # For well-known operations, we don't need a signature
        if isinstance(op, OpKind):
            self.op_wrapper = cccl.to_cccl_op(op, None)
        else:
            if not out_value_type.is_internal:
                out_value_type = get_inferred_return_type(op, (in_value_type,))
            sig = out_value_type(in_value_type)
            self.op_wrapper = cccl.to_cccl_op(op, sig=sig)
        self.build_result = cccl.call_build(
            _bindings.DeviceUnaryTransform,
            self.d_in_cccl,
            self.d_out_cccl,
            self.op_wrapper,
        )

    def __call__(
        self,
        d_in,
        d_out,
        num_items: int,
        stream=None,
    ):
        set_cccl_iterator_state(self.d_in_cccl, d_in)
        set_cccl_iterator_state(self.d_out_cccl, d_out)
        stream_handle = protocols.validate_and_get_stream(stream)
        self.build_result.compute(
            self.d_in_cccl,
            self.d_out_cccl,
            num_items,
            self.op_wrapper,
            stream_handle,
        )
        return None


class _BinaryTransform:
    __slots__ = [
        "d_in1_cccl",
        "d_in2_cccl",
        "d_out_cccl",
        "op_wrapper",
        "build_result",
    ]

    def __init__(
        self,
        d_in1: DeviceArrayLike | IteratorBase,
        d_in2: DeviceArrayLike | IteratorBase,
        d_out: DeviceArrayLike | IteratorBase,
        op: Callable | OpKind,
    ):
        self.d_in1_cccl = cccl.to_cccl_iter(d_in1)
        self.d_in2_cccl = cccl.to_cccl_iter(d_in2)
        self.d_out_cccl = cccl.to_cccl_iter(d_out)
        in1_value_type = cccl.get_value_type(d_in1)
        in2_value_type = cccl.get_value_type(d_in2)
        out_value_type = cccl.get_value_type(d_out)

        # For well-known operations, we don't need a signature
        if isinstance(op, OpKind):
            self.op_wrapper = cccl.to_cccl_op(op, None)
        else:
            if not out_value_type.is_internal:
                out_value_type = get_inferred_return_type(
                    op, (in1_value_type, in2_value_type)
                )
            sig = out_value_type(in1_value_type, in2_value_type)
            self.op_wrapper = cccl.to_cccl_op(op, sig=sig)
        self.build_result = cccl.call_build(
            _bindings.DeviceBinaryTransform,
            self.d_in1_cccl,
            self.d_in2_cccl,
            self.d_out_cccl,
            self.op_wrapper,
        )

    def __call__(
        self,
        d_in1,
        d_in2,
        d_out,
        num_items: int,
        stream=None,
    ):
        set_cccl_iterator_state(self.d_in1_cccl, d_in1)
        set_cccl_iterator_state(self.d_in2_cccl, d_in2)
        set_cccl_iterator_state(self.d_out_cccl, d_out)
        stream_handle = protocols.validate_and_get_stream(stream)
        self.build_result.compute(
            self.d_in1_cccl,
            self.d_in2_cccl,
            self.d_out_cccl,
            num_items,
            self.op_wrapper,
            stream_handle,
        )
        return None


def make_unary_transform_cache_key(
    d_in: DeviceArrayLike | IteratorBase,
    d_out: DeviceArrayLike,
    op: Callable | OpKind,
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

    return (d_in_key, d_out_key, op_key)


def make_binary_transform_cache_key(
    d_in1: DeviceArrayLike | IteratorBase,
    d_in2: DeviceArrayLike | IteratorBase,
    d_out: DeviceArrayLike,
    op: Callable | OpKind,
):
    d_in1_key = (
        d_in1.kind if isinstance(d_in1, IteratorBase) else protocols.get_dtype(d_in1)
    )
    d_in2_key = (
        d_in2.kind if isinstance(d_in2, IteratorBase) else protocols.get_dtype(d_in2)
    )
    d_out_key = protocols.get_dtype(d_out)

    # Handle well-known operations differently
    op_key: Union[tuple[str, int], CachableFunction]
    if isinstance(op, OpKind):
        op_key = (op.name, op.value)
    else:
        op_key = CachableFunction(op)

    return (d_in1_key, d_in2_key, d_out_key, op_key)


@cache_with_key(make_unary_transform_cache_key)
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
        .. literalinclude:: ../../python/cuda_cccl/tests/parallel/test_transform.py
           :language: python
           :dedent:
           :start-after: example-begin transform-unary
           :end-before: example-end transform-unary

    Args:
        d_in: Device array or iterator containing the input sequence of data items.
        d_out: Device array or iterator to store the result of the transformation.
        op: Callable or OpKind representing the unary operation to apply to each element of the input.

    Returns:
        A callable object that performs the transformation.
    """
    return _UnaryTransform(d_in, d_out, op)


@cache_with_key(make_binary_transform_cache_key)
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
        .. literalinclude:: ../../python/cuda_cccl/tests/parallel/test_transform.py
           :language: python
           :dedent:
           :start-after: example-begin transform-binary
           :end-before: example-end transform-binary

    Args:
        d_in1: Device array or iterator containing the first input sequence of data items.
        d_in2: Device array or iterator containing the second input sequence of data items.
        d_out: Device array or iterator to store the result of the transformation.
        op: Callable or OpKind representing the binary operation to apply to each pair of items from the input sequences.

    Returns:
        A callable object that performs the transformation.
    """
    return _BinaryTransform(d_in1, d_in2, d_out, op)


def unary_transform(
    d_in: DeviceArrayLike | IteratorBase,
    d_out: DeviceArrayLike | IteratorBase,
    op: Callable | OpKind,
    num_items: int,
    stream=None,
):
    """
    Create a unary transform object that can be called to apply a transformation
    to each element of the input according to the unary operation ``op``.

    This is the two-phase API that returns a transform object for execution.

    Example:
        .. literalinclude:: ../../python/cuda_cccl/tests/parallel/test_transform.py
           :language: python
           :dedent:
           :start-after: example-begin transform-unary
           :end-before: example-end transform-unary

    Args:
        d_in: Device array or iterator containing the input sequence of data items.
        d_out: Device array or iterator to store the result of the transformation.
        op: Callable or OpKind representing the unary operation to apply to each element of the input.
        num_items: Number of items to transform.
        stream: CUDA stream to use for the operation.
    """
    transformer = make_unary_transform(d_in, d_out, op)
    transformer(d_in, d_out, num_items, stream)


def binary_transform(
    d_in1: DeviceArrayLike | IteratorBase,
    d_in2: DeviceArrayLike | IteratorBase,
    d_out: DeviceArrayLike | IteratorBase,
    op: Callable | OpKind,
    num_items: int,
    stream=None,
):
    """
    Create a binary transform object that can be called to apply a transformation
    to the given pair of input sequences according to the binary operation ``op``.

    This is the two-phase API that returns a transform object for execution.

    Example:
        .. literalinclude:: ../../python/cuda_cccl/tests/parallel/test_transform.py
           :language: python
           :dedent:
           :start-after: example-begin transform-binary
           :end-before: example-end transform-binary

    Args:
        d_in1: Device array or iterator containing the first input sequence of data items.
        d_in2: Device array or iterator containing the second input sequence of data items.
        d_out: Device array or iterator to store the result of the transformation.
        op: Callable or OpKind representing the binary operation to apply to each pair of items from the input sequences.
        num_items: Number of items to transform.
        stream: CUDA stream to use for the operation.
    """
    transformer = make_binary_transform(d_in1, d_in2, d_out, op)
    transformer(d_in1, d_in2, d_out, num_items, stream)
