# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Field conversion and access semantics for ``gpu_struct``.

Covers how values are converted as they are packed into struct fields (the
constructor, the tuple-to-struct cast and the struct-to-struct cast) and how
fields are selected by index.
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

    with pytest.raises(Exception, match="cannot initialize field"):
        cuda.compute.unary_transform(
            d_in=d_in, d_out=d_out, op=build, num_items=h_in.size
        )


def test_constant_index_selects_the_field():
    """``struct[i]`` with a constant index reads field ``i``."""
    Pair = gpu_struct({"a": np.int32, "b": np.int32})

    def second(s):
        return s[1]

    h_in = np.zeros(4, dtype=Pair.dtype)
    h_in["a"] = np.arange(4)
    h_in["b"] = np.arange(100, 104)

    d_in = DeviceArray.from_numpy(h_in)
    d_out = DeviceArray.empty(h_in.shape, np.dtype(np.int32))

    cuda.compute.unary_transform(d_in=d_in, d_out=d_out, op=second, num_items=h_in.size)

    np.testing.assert_array_equal(d_out.copy_to_host(), h_in["b"])


def test_index_out_of_range_is_rejected():
    """An out-of-range constant index is reported against the struct."""
    Pair = gpu_struct({"a": np.int32, "b": np.int32})

    def out_of_range(s):
        return s[5]

    h_in = np.zeros(4, dtype=Pair.dtype)
    d_in = DeviceArray.from_numpy(h_in)
    d_out = DeviceArray.empty(h_in.shape, np.dtype(np.int32))

    with pytest.raises(Exception, match="out of range"):
        cuda.compute.unary_transform(
            d_in=d_in, d_out=d_out, op=out_of_range, num_items=h_in.size
        )


def test_runtime_index_is_rejected():
    """A non-constant index is reported as needing a compile-time constant."""
    Pair = gpu_struct({"a": np.int32, "b": np.int32})

    def runtime_index(s):
        total = 0
        for i in range(2):
            total += s[i]
        return total

    h_in = np.zeros(4, dtype=Pair.dtype)
    d_in = DeviceArray.from_numpy(h_in)
    d_out = DeviceArray.empty(h_in.shape, np.dtype(np.int64))

    with pytest.raises(Exception, match="compile-time constant index"):
        cuda.compute.unary_transform(
            d_in=d_in, d_out=d_out, op=runtime_index, num_items=h_in.size
        )


def test_constructor_rejects_incompatible_argument_type():
    """A field cannot be initialized from an unrelated type.

    The error names the field and both types, rather than failing later while
    the constructor is lowered.
    """
    Pair = gpu_struct({"a": np.int32, "b": np.int32})
    Other = gpu_struct({"c": np.int32})

    def build(s):
        # 's' is a struct; field 'a' is a scalar.
        return Pair(s, s.c)

    h_in = np.zeros(4, dtype=Other.dtype)
    d_in = DeviceArray.from_numpy(h_in)
    d_out = DeviceArray.empty(h_in.shape, Pair.dtype)

    with pytest.raises(Exception, match="cannot initialize field"):
        cuda.compute.unary_transform(
            d_in=d_in, d_out=d_out, op=build, num_items=h_in.size
        )


def test_complex_field_is_read_and_constructed():
    """A struct may hold a complex field.

    Complex values are MLIR complex scalars in SSA but are stored as a literal
    ``{real, imag}`` LLVM struct, which is the only form the LLVM dialect
    accepts as a struct member.
    """
    Sample = gpu_struct({"z": np.complex64, "n": np.int32})

    h_in = np.zeros(4, dtype=Sample.dtype)
    h_in["z"] = np.array([1 + 2j, 3 + 4j, 5 + 6j, 7 + 8j], dtype=np.complex64)
    h_in["n"] = np.arange(4)
    d_in = DeviceArray.from_numpy(h_in)

    def read_complex(s):
        return s.z

    d_z = DeviceArray.empty(h_in.shape, np.dtype(np.complex64))
    cuda.compute.unary_transform(
        d_in=d_in, d_out=d_z, op=read_complex, num_items=h_in.size
    )
    np.testing.assert_array_equal(d_z.copy_to_host(), h_in["z"])

    def scale(s):
        return Sample(s.z * 2, s.n + 1)

    d_out = DeviceArray.empty(h_in.shape, Sample.dtype)
    cuda.compute.unary_transform(d_in=d_in, d_out=d_out, op=scale, num_items=h_in.size)
    result = d_out.copy_to_host()
    np.testing.assert_array_equal(result["z"], h_in["z"] * 2)
    np.testing.assert_array_equal(result["n"], h_in["n"] + 1)
