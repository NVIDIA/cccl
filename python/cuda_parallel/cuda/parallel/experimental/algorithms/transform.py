# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import ctypes
from typing import Callable

from numba.cuda.cudadrv import enums

from .. import _cccl as cccl
from .._bindings import call_build, get_bindings
from .._caching import CachableFunction, cache_with_key
from .._utils import protocols
from ..iterators._iterators import IteratorBase, scrub_duplicate_ltoirs
from ..typing import DeviceArrayLike


class _UnaryTransform:
    def __init__(
        self,
        d_in: DeviceArrayLike | IteratorBase,
        d_out: DeviceArrayLike | IteratorBase,
        op: Callable,
    ):
        # Referenced from __del__:
        self.build_result = None
        self.d_in_cccl = cccl.to_cccl_iter(d_in)
        self.d_out_cccl = cccl.to_cccl_iter(d_out)
        in_value_type = cccl.get_value_type(d_in)
        out_value_type = cccl.get_value_type(d_out)

        if not out_value_type.is_internal:
            sig = (in_value_type,)
        else:
            sig = out_value_type(in_value_type)

        self.op_wrapper = cccl.to_cccl_op(op, sig=sig)
        self.build_result = cccl.DeviceTransformBuildResult()
        self.bindings = get_bindings()
        error = call_build(
            self.bindings.cccl_device_unary_transform_build,
            ctypes.byref(self.build_result),
            self.d_in_cccl,
            self.d_out_cccl,
            self.op_wrapper,
        )
        if error != enums.CUDA_SUCCESS:
            raise ValueError("Error building unary transform")

    def __call__(
        self,
        d_in,
        d_out,
        num_items: int,
        stream=None,
    ):
        set_state_fn = cccl.set_cccl_iterator_state
        set_state_fn(self.d_in_cccl, d_in)
        set_state_fn(self.d_out_cccl, d_out)

        stream_handle = protocols.validate_and_get_stream(stream)

        error = self.bindings.cccl_device_unary_transform(
            self.build_result,
            self.d_in_cccl,
            self.d_out_cccl,
            ctypes.c_ulonglong(num_items),
            self.op_wrapper,
            ctypes.c_void_p(stream_handle),
        )

        if error != enums.CUDA_SUCCESS:
            raise ValueError("Error performing unary transform")

        return None

    def __del__(self):
        if self.build_result is None:
            return
        self.bindings.cccl_device_transform_cleanup(ctypes.byref(self.build_result))


class _BinaryTransform:
    def __init__(
        self,
        d_in1: DeviceArrayLike | IteratorBase,
        d_in2: DeviceArrayLike | IteratorBase,
        d_out: DeviceArrayLike | IteratorBase,
        op: Callable,
    ):
        # Referenced from __del__:
        self.build_result = None

        d_in1, d_in2 = scrub_duplicate_ltoirs(d_in1, d_in2)

        self.d_in1_cccl = cccl.to_cccl_iter(d_in1)
        self.d_in2_cccl = cccl.to_cccl_iter(d_in2)
        self.d_out_cccl = cccl.to_cccl_iter(d_out)
        in1_value_type = cccl.get_value_type(d_in1)
        in2_value_type = cccl.get_value_type(d_in2)
        out_value_type = cccl.get_value_type(d_out)

        if not out_value_type.is_internal:
            sig = (in1_value_type, in2_value_type)
        else:
            sig = out_value_type(in1_value_type, in2_value_type)

        self.op_wrapper = cccl.to_cccl_op(op, sig=sig)
        self.build_result = cccl.DeviceTransformBuildResult()
        self.bindings = get_bindings()
        error = call_build(
            self.bindings.cccl_device_binary_transform_build,
            ctypes.byref(self.build_result),
            self.d_in1_cccl,
            self.d_in2_cccl,
            self.d_out_cccl,
            self.op_wrapper,
        )
        if error != enums.CUDA_SUCCESS:
            raise ValueError("Error building binary transform")

    def __call__(
        self,
        d_in1,
        d_in2,
        d_out,
        num_items: int,
        stream=None,
    ):
        set_state_fn = cccl.set_cccl_iterator_state
        set_state_fn(self.d_in1_cccl, d_in1)
        set_state_fn(self.d_in2_cccl, d_in2)
        set_state_fn(self.d_out_cccl, d_out)

        stream_handle = protocols.validate_and_get_stream(stream)

        error = self.bindings.cccl_device_binary_transform(
            self.build_result,
            self.d_in1_cccl,
            self.d_in2_cccl,
            self.d_out_cccl,
            ctypes.c_ulonglong(num_items),
            self.op_wrapper,
            ctypes.c_void_p(stream_handle),
        )

        if error != enums.CUDA_SUCCESS:
            raise ValueError("Error performing binary transform")

        return None

    def __del__(self):
        if self.build_result is None:
            return
        self.bindings.cccl_device_transform_cleanup(ctypes.byref(self.build_result))


def make_unary_transform_cache_key(
    d_in: DeviceArrayLike | IteratorBase,
    d_out: DeviceArrayLike,
    op: Callable,
):
    d_in_key = (
        d_in.kind if isinstance(d_in, IteratorBase) else protocols.get_dtype(d_in)
    )
    d_out_key = protocols.get_dtype(d_out)
    op_key = CachableFunction(op)
    return (d_in_key, d_out_key, op_key)


def make_binary_transform_cache_key(
    d_in1: DeviceArrayLike | IteratorBase,
    d_in2: DeviceArrayLike | IteratorBase,
    d_out: DeviceArrayLike,
    op: Callable,
):
    d_in1_key = (
        d_in1.kind if isinstance(d_in1, IteratorBase) else protocols.get_dtype(d_in1)
    )
    d_in2_key = (
        d_in2.kind if isinstance(d_in2, IteratorBase) else protocols.get_dtype(d_in2)
    )
    d_out_key = protocols.get_dtype(d_out)
    op_key = CachableFunction(op)
    return (d_in1_key, d_in2_key, d_out_key, op_key)


@cache_with_key(make_unary_transform_cache_key)
def unary_transform(
    d_in: DeviceArrayLike | IteratorBase,
    d_out: DeviceArrayLike | IteratorBase,
    op: Callable,
):
    """
    Apply a transformation to each element of the input according to the
    unary operation ``op``.

    Example:
        .. literalinclude:: ../../python/cuda_parallel/tests/test_transform.py
           :language: python
           :dedent:
           :start-after: example-begin transform-unary
           :end-before: example-end transform-unary

    Args:
        d_in: Device array or iterator containing the input sequence of data items.
        d_out: Device array or iterator to store the result of the transformation.
        op: Unary operation to apply to each element of the input.

    Returns:
        A callable that performs the transformation.
    """
    return _UnaryTransform(d_in, d_out, op)


@cache_with_key(make_binary_transform_cache_key)
def binary_transform(
    d_in1: DeviceArrayLike | IteratorBase,
    d_in2: DeviceArrayLike | IteratorBase,
    d_out: DeviceArrayLike | IteratorBase,
    op: Callable,
):
    """
    Apply a transformation to the given pair of input sequences according to the
    binary operation ``op``.

    Example:
        .. literalinclude:: ../../python/cuda_parallel/tests/test_transform.py
           :language: python
           :dedent:
           :start-after: example-begin transform-binary
           :end-before: example-end transform-binary

    Args:
        d_in1: Device array or iterator containing the first input sequence of data items.
        d_in2: Device array or iterator containing the second input sequence of data items.
        d_out: Device array or iterator to store the result of the transformation.
        op: Binary operation to apply to each pair of items from the input sequences.

    Returns:
        A callable that performs the transformation.
    """
    return _BinaryTransform(d_in1, d_in2, d_out, op)
