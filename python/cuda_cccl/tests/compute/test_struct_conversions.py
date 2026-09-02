# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Conversion semantics for ``gpu_struct`` fields.

Covers how values are converted as they are packed into struct fields: the
constructor, the tuple-to-struct cast, and the struct-to-struct cast.
"""

import numpy as np
import pytest
from _utils.device_array import DeviceArray

import cuda.compute
from cuda.compute import gpu_struct

# Signed inputs whose sign must survive being widened into a wider field.
NEGATIVE_INPUT = np.array([-1, -2147483648, -7, 0, 5], dtype=np.int32)


def test_constructor_widens_signed_field():
    """``Wide(x)`` with a narrower signed ``x`` sign-extends into the field."""
    Wide = gpu_struct({"a": np.int64, "b": np.int64})

    def widen(x):
        return Wide(x, x)

    d_in = DeviceArray.from_numpy(NEGATIVE_INPUT)
    d_out = DeviceArray.empty(NEGATIVE_INPUT.shape, Wide.dtype)

    cuda.compute.unary_transform(
        d_in=d_in, d_out=d_out, op=widen, num_items=NEGATIVE_INPUT.size
    )

    result = d_out.copy_to_host()
    expected = NEGATIVE_INPUT.astype(np.int64)
    np.testing.assert_array_equal(result["a"], expected)
    np.testing.assert_array_equal(result["b"], expected)


def test_tuple_return_widens_signed_field():
    """A tuple of narrower signed values packed into a struct sign-extends."""
    Wide = gpu_struct({"a": np.int64, "b": np.int64})

    def widen(x):
        return (x, x)

    d_in = DeviceArray.from_numpy(NEGATIVE_INPUT)
    d_out = DeviceArray.empty(NEGATIVE_INPUT.shape, Wide.dtype)

    cuda.compute.unary_transform(
        d_in=d_in, d_out=d_out, op=widen, num_items=NEGATIVE_INPUT.size
    )

    result = d_out.copy_to_host()
    expected = NEGATIVE_INPUT.astype(np.int64)
    np.testing.assert_array_equal(result["a"], expected)
    np.testing.assert_array_equal(result["b"], expected)


def test_nested_struct_widens_signed_field():
    """Sign is preserved through a nested struct field built from a tuple."""
    Inner = gpu_struct({"a": np.int64})
    Outer = gpu_struct({"x": np.int64, "inner": Inner})

    def widen(x):
        return Outer(x, (x,))

    d_in = DeviceArray.from_numpy(NEGATIVE_INPUT)
    d_out = DeviceArray.empty(NEGATIVE_INPUT.shape, Outer.dtype)

    cuda.compute.unary_transform(
        d_in=d_in, d_out=d_out, op=widen, num_items=NEGATIVE_INPUT.size
    )

    result = d_out.copy_to_host()
    expected = NEGATIVE_INPUT.astype(np.int64)
    np.testing.assert_array_equal(result["x"], expected)
    np.testing.assert_array_equal(result["inner"]["a"], expected)


def test_unsigned_widening_zero_extends():
    """Unsigned sources still zero-extend when widened."""
    Wide = gpu_struct({"a": np.uint64})
    h_in = np.array([0, 1, 4294967295], dtype=np.uint32)

    def widen(x):
        return Wide(x)

    d_in = DeviceArray.from_numpy(h_in)
    d_out = DeviceArray.empty(h_in.shape, Wide.dtype)

    cuda.compute.unary_transform(d_in=d_in, d_out=d_out, op=widen, num_items=h_in.size)

    np.testing.assert_array_equal(d_out.copy_to_host()["a"], h_in.astype(np.uint64))


def test_nested_tuple_too_short_is_rejected():
    """A tuple with fewer values than the nested struct's fields is an error."""
    Inner = gpu_struct({"a": np.int32, "b": np.int32})
    Outer = gpu_struct({"x": np.int32, "inner": Inner})

    def build(x):
        return Outer(x, (x,))

    h_in = np.arange(3, dtype=np.int32)
    d_in = DeviceArray.from_numpy(h_in)
    d_out = DeviceArray.empty(h_in.shape, Outer.dtype)

    with pytest.raises(Exception, match="tuple of size 1"):
        cuda.compute.unary_transform(
            d_in=d_in, d_out=d_out, op=build, num_items=h_in.size
        )


def test_nested_tuple_too_long_is_rejected():
    """A tuple with more values than the nested struct's fields is an error."""
    Inner = gpu_struct({"a": np.int32, "b": np.int32})
    Outer = gpu_struct({"x": np.int32, "inner": Inner})

    def build(x):
        return Outer(x, (x, x, x))

    h_in = np.arange(3, dtype=np.int32)
    d_in = DeviceArray.from_numpy(h_in)
    d_out = DeviceArray.empty(h_in.shape, Outer.dtype)

    with pytest.raises(Exception, match="tuple of size 3"):
        cuda.compute.unary_transform(
            d_in=d_in, d_out=d_out, op=build, num_items=h_in.size
        )


def test_tuple_for_scalar_field_is_rejected():
    """A tuple supplied for a scalar field is an error, not an AttributeError."""
    Inner = gpu_struct({"a": np.int32, "b": np.int32})
    Outer = gpu_struct({"x": np.int32, "inner": Inner})

    def build(x):
        return Outer((x, x), Inner(x, x))

    h_in = np.arange(3, dtype=np.int32)
    d_in = DeviceArray.from_numpy(h_in)
    d_out = DeviceArray.empty(h_in.shape, Outer.dtype)

    with pytest.raises(Exception, match="only a nested struct field"):
        cuda.compute.unary_transform(
            d_in=d_in, d_out=d_out, op=build, num_items=h_in.size
        )
