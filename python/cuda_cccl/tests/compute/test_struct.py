# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for struct dtype handling and validation."""

import cupy as cp
import numpy as np
import pytest

import cuda.compute


def test_reduce_rejects_unaligned_h_init_scalar():
    """Test that reduce_into rejects unaligned np.void h_init values."""
    # Create unaligned dtype (missing align=True)
    unaligned_dtype = np.dtype([("x", np.int32), ("y", np.int64)])
    assert not unaligned_dtype.isalignedstruct

    d_in = cp.ones(10, dtype=np.int32)
    d_out = cp.empty(1, dtype=np.int32)

    # Create unaligned h_init
    h_init = np.void((0, 0), dtype=unaligned_dtype)

    def op(a, b):
        return a + b

    with pytest.raises(ValueError, match="align=True"):
        cuda.compute.reduce_into(d_in, d_out, op, len(d_in), h_init)


def test_reduce_rejects_unaligned_h_init_array():
    """Test that reduce_into rejects unaligned np.ndarray h_init values."""
    # Create unaligned dtype (missing align=True)
    unaligned_dtype = np.dtype([("x", np.int32), ("y", np.int64)])
    assert not unaligned_dtype.isalignedstruct

    d_in = cp.ones(10, dtype=np.int32)
    d_out = cp.empty(1, dtype=np.int32)

    # Create unaligned h_init as array
    h_init = np.zeros(1, dtype=unaligned_dtype)

    def op(a, b):
        return a + b

    with pytest.raises(ValueError, match="align=True"):
        cuda.compute.reduce_into(d_in, d_out, op, len(d_in), h_init)


def test_reduce_rejects_unaligned_input_array():
    """Test that reduce_into rejects unaligned input arrays."""
    # Create unaligned dtype (missing align=True)
    unaligned_dtype = np.dtype([("x", np.int32), ("y", np.int32)])
    assert not unaligned_dtype.isalignedstruct

    # Aligned dtype for h_init
    aligned_dtype = np.dtype([("x", np.int32), ("y", np.int32)], align=True)

    # Create unaligned input array - cupy struct arrays are created by
    # making an empty array with the dtype, then copying from numpy
    h_data = np.zeros(10, dtype=unaligned_dtype)
    d_in = cp.empty(10, dtype=unaligned_dtype)
    d_in.set(h_data)
    d_out = cp.empty(1, dtype=aligned_dtype)
    h_init = np.void((0, 0), dtype=aligned_dtype)

    def op(a, b):
        return (a.x + b.x, a.y + b.y)

    with pytest.raises(ValueError, match="align=True"):
        cuda.compute.reduce_into(d_in, d_out, op, len(d_in), h_init)


def test_reduce_accepts_aligned_struct_dtype():
    """Test that reduce_into accepts properly aligned struct dtypes."""
    # Create aligned dtype
    aligned_dtype = np.dtype([("x", np.int32), ("y", np.int32)], align=True)
    assert aligned_dtype.isalignedstruct

    d_in = cp.arange(20, dtype=np.int32).reshape(10, 2).view(aligned_dtype)
    d_out = cp.empty(1, dtype=aligned_dtype)
    h_init = np.void((0, 0), dtype=aligned_dtype)

    def op(a, b):
        return (a.x + b.x, a.y + b.y)

    # Should not raise
    cuda.compute.reduce_into(d_in, d_out, op, len(d_in), h_init)

    result = d_out.get()[0]
    # x values: 0, 2, 4, ..., 18 -> sum = 90
    # y values: 1, 3, 5, ..., 19 -> sum = 100
    assert result["x"] == 90
    assert result["y"] == 100


def test_scan_rejects_unaligned_h_init():
    """Test that scan operations reject unaligned h_init values."""
    # Create unaligned dtype
    unaligned_dtype = np.dtype([("a", np.float32), ("b", np.float64)])
    assert not unaligned_dtype.isalignedstruct

    d_in = cp.ones(10, dtype=np.float32)
    d_out = cp.empty(10, dtype=np.float32)
    h_init = np.void((0.0, 0.0), dtype=unaligned_dtype)

    def op(a, b):
        return a + b

    with pytest.raises(ValueError, match="align=True"):
        cuda.compute.exclusive_scan(d_in, d_out, op, h_init, len(d_in))


def test_transform_rejects_unaligned_return_type_annotation():
    """Test that transform rejects functions with unaligned return type annotations."""
    # Create unaligned dtype for return type annotation
    unaligned_dtype = np.dtype([("x", np.int32), ("y", np.int32)])
    assert not unaligned_dtype.isalignedstruct

    d_in = cp.arange(10, dtype=np.int32)
    h_out = np.empty(10, dtype=unaligned_dtype)
    d_out = cp.empty(10, dtype=unaligned_dtype)
    d_out.set(h_out)

    def op(a) -> unaligned_dtype:
        return (a, a * 2)

    with pytest.raises(ValueError, match="align=True"):
        cuda.compute.unary_transform(d_in, d_out, op, len(d_in))


def test_transform_rejects_unaligned_struct_input():
    """Test that transform rejects input arrays with unaligned struct dtypes."""
    # Create unaligned dtype
    unaligned_dtype = np.dtype([("x", np.int32), ("y", np.int32)])
    assert not unaligned_dtype.isalignedstruct

    # Create unaligned input array
    h_data = np.zeros(10, dtype=unaligned_dtype)
    d_in = cp.empty(10, dtype=unaligned_dtype)
    d_in.set(h_data)
    d_out = cp.empty(10, dtype=np.int32)

    def op(a):
        return a.x + a.y

    with pytest.raises(ValueError, match="align=True"):
        cuda.compute.unary_transform(d_in, d_out, op, len(d_in))


def test_transform_accepts_aligned_struct_input():
    """Test that transform accepts arrays with aligned struct dtypes."""
    # Create aligned dtype
    aligned_dtype = np.dtype([("x", np.int32), ("y", np.int32)], align=True)
    assert aligned_dtype.isalignedstruct

    # Create input array with struct dtype using cupy empty + set pattern
    h_data = np.empty(5, dtype=aligned_dtype)
    h_data["x"] = [0, 2, 4, 6, 8]
    h_data["y"] = [1, 3, 5, 7, 9]
    d_in = cp.empty(5, dtype=aligned_dtype)
    d_in.set(h_data)
    d_out = cp.empty(5, dtype=np.int32)

    def op(a):
        return a.x + a.y

    # Should not raise - input array with aligned struct dtype
    cuda.compute.unary_transform(d_in, d_out, op, len(d_in))

    result = d_out.get()
    # x + y: 1, 5, 9, 13, 17
    np.testing.assert_array_equal(result, [1, 5, 9, 13, 17])


def test_struct_dtype_basic_reduction():
    """Test basic reduction with aligned struct dtype."""
    pixel_dtype = np.dtype(
        [("r", np.int32), ("g", np.int32), ("b", np.int32)], align=True
    )

    d_in = cp.random.randint(0, 256, (10, 3), dtype=np.int32).view(pixel_dtype)
    d_out = cp.empty(1, dtype=pixel_dtype)
    h_init = np.void((0, 0, 0), dtype=pixel_dtype)

    def sum_pixels(a, b):
        return (a.r + b.r, a.g + b.g, a.b + b.b)

    cuda.compute.reduce_into(d_in, d_out, sum_pixels, len(d_in), h_init)

    result = d_out.get()[0]
    h_in = d_in.get()
    assert result["r"] == h_in["r"].sum()
    assert result["g"] == h_in["g"].sum()
    assert result["b"] == h_in["b"].sum()


def test_nested_struct_dtype():
    """Test nested struct dtypes are properly handled."""
    inner_dtype = np.dtype([("a", np.int32), ("b", np.float32)], align=True)
    outer_dtype = np.dtype([("x", np.int64), ("inner", inner_dtype)], align=True)

    assert inner_dtype.isalignedstruct
    assert outer_dtype.isalignedstruct

    d_in = cp.empty(5, dtype=outer_dtype)
    h_data = np.zeros(5, dtype=outer_dtype)
    for i in range(5):
        h_data[i]["x"] = i
        h_data[i]["inner"]["a"] = i * 2
        h_data[i]["inner"]["b"] = float(i * 3)
    d_in.set(h_data)

    d_out = cp.empty(1, dtype=outer_dtype)
    h_init = np.void((0, (0, 0.0)), dtype=outer_dtype)

    def sum_nested(s1, s2):
        return (
            s1.x + s2.x,
            (s1.inner.a + s2.inner.a, s1.inner.b + s2.inner.b),
        )

    cuda.compute.reduce_into(d_in, d_out, sum_nested, len(d_in), h_init)

    result = d_out.get()[0]
    assert result["x"] == sum(range(5))
    assert result["inner"]["a"] == sum(i * 2 for i in range(5))
    np.testing.assert_allclose(
        result["inner"]["b"], sum(float(i * 3) for i in range(5))
    )
