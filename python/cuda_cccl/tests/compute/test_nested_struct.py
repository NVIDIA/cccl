# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import cupy as cp
import numpy as np

import cuda.compute
from cuda.compute import ZipIterator


def test_reduce_nested_struct_direct():
    inner_dtype = np.dtype([("a", np.int32), ("b", np.float32)], align=True)
    outer_dtype = np.dtype([("x", np.int64), ("inner", inner_dtype)], align=True)

    def sum_nested(s1, s2):
        return (
            s1.x + s2.x,
            (s1.inner.a + s2.inner.a, s1.inner.b + s2.inner.b),
        )

    num_items = 10

    h_data = np.zeros(num_items, dtype=outer_dtype)
    for i in range(num_items):
        h_data[i]["x"] = i
        h_data[i]["inner"]["a"] = i * 2
        h_data[i]["inner"]["b"] = float(i * 3)

    itemsize = h_data.dtype.itemsize
    d_input = cp.empty(num_items * itemsize, dtype=np.uint8)
    d_input.set(h_data.view(np.uint8))
    d_input = d_input.view(outer_dtype)

    d_output = cp.empty(itemsize, dtype=np.uint8)
    d_output = d_output.view(outer_dtype)

    h_init = np.void((0, (0, 0.0)), dtype=outer_dtype)

    cuda.compute.reduce_into(d_input, d_output, sum_nested, num_items, h_init)

    result = d_output.view(np.uint8).get().view(outer_dtype)[0]

    expected_x = sum(range(num_items))
    expected_a = sum(i * 2 for i in range(num_items))
    expected_b = sum(float(i * 3) for i in range(num_items))

    assert result["x"] == expected_x
    assert result["inner"]["a"] == expected_a
    assert np.isclose(result["inner"]["b"], expected_b)


def test_nested_struct_in_zip_iterator():
    point_dtype = np.dtype([("x", np.int32), ("y", np.int32)], align=True)
    color_dtype = np.dtype(
        [("r", np.uint8), ("g", np.uint8), ("b", np.uint8)], align=True
    )
    pixel_dtype = np.dtype(
        [("position", point_dtype), ("color", color_dtype)], align=True
    )

    def sum_pixels(p1, p2):
        return (
            (p1.position.x + p2.position.x, p1.position.y + p2.position.y),
            (
                p1.color.r + p2.color.r,
                p1.color.g + p2.color.g,
                p1.color.b + p2.color.b,
            ),
        )

    num_items = 100

    d_points = cp.empty(num_items, dtype=point_dtype)
    d_colors = cp.empty(num_items, dtype=color_dtype)

    h_points = np.array([(i, i * 2) for i in range(num_items)], dtype=point_dtype)
    h_colors = np.array(
        [(i % 256, (i * 2) % 256, (i * 3) % 256) for i in range(num_items)],
        dtype=color_dtype,
    )

    d_points.set(h_points)
    d_colors.set(h_colors)

    zip_it = ZipIterator(d_points, d_colors)

    d_output = cp.empty(1, dtype=pixel_dtype)
    h_init = np.void(((0, 0), (0, 0, 0)), dtype=pixel_dtype)

    cuda.compute.reduce_into(zip_it, d_output, sum_pixels, num_items, h_init)

    result = d_output.get()[0]

    expected_x = sum(i for i in range(num_items))
    expected_y = sum(i * 2 for i in range(num_items))
    expected_r = sum(i % 256 for i in range(num_items)) % 256
    expected_g = sum((i * 2) % 256 for i in range(num_items)) % 256
    expected_b = sum((i * 3) % 256 for i in range(num_items)) % 256

    assert result["position"]["x"] == expected_x
    assert result["position"]["y"] == expected_y
    assert result["color"]["r"] == expected_r
    assert result["color"]["g"] == expected_g
    assert result["color"]["b"] == expected_b


def test_nested_struct_tuple_construction():
    """Test constructing nested structs using tuple syntax in device functions."""
    inner_dtype = np.dtype([("a", np.int32), ("b", np.float32)], align=True)
    outer_dtype = np.dtype([("x", np.int64), ("inner", inner_dtype)], align=True)

    def sum_nested_with_tuples(s1, s2):
        # Use tuple syntax for nested struct
        return (s1.x + s2.x, (s1.inner.a + s2.inner.a, s1.inner.b + s2.inner.b))

    num_items = 10

    h_data = np.zeros(num_items, dtype=outer_dtype)
    for i in range(num_items):
        h_data[i]["x"] = i
        h_data[i]["inner"]["a"] = i * 2
        h_data[i]["inner"]["b"] = float(i * 3)

    itemsize = h_data.dtype.itemsize
    d_input = cp.empty(num_items * itemsize, dtype=np.uint8)
    d_input.set(h_data.view(np.uint8))
    d_input = d_input.view(outer_dtype)

    d_output = cp.empty(itemsize, dtype=np.uint8)
    d_output = d_output.view(outer_dtype)

    h_init = np.void((0, (0, 0.0)), dtype=outer_dtype)

    cuda.compute.reduce_into(
        d_input, d_output, sum_nested_with_tuples, num_items, h_init
    )

    result = d_output.view(np.uint8).get().view(outer_dtype)[0]

    expected_x = sum(range(num_items))
    expected_a = sum(i * 2 for i in range(num_items))
    expected_b = sum(float(i * 3) for i in range(num_items))

    assert result["x"] == expected_x
    assert result["inner"]["a"] == expected_a
    assert np.isclose(result["inner"]["b"], expected_b)


def test_deeply_nested_tuple_construction():
    """Test constructing deeply nested structs (3 levels) using tuple syntax."""
    level1_dtype = np.dtype([("value", np.int32)], align=True)
    level2_dtype = np.dtype(
        [("data", np.float32), ("nested", level1_dtype)], align=True
    )
    level3_dtype = np.dtype([("id", np.int64), ("middle", level2_dtype)], align=True)

    def sum_deeply_nested(v1, v2):
        # Use nested tuple syntax: (int64, (float32, (int32,)))
        return (
            v1.id + v2.id,
            (
                v1.middle.data + v2.middle.data,
                (v1.middle.nested.value + v2.middle.nested.value,),
            ),
        )

    num_items = 10

    h_data = np.zeros(num_items, dtype=level3_dtype)
    for i in range(num_items):
        h_data[i]["id"] = i * 10
        h_data[i]["middle"]["data"] = float(i * 2.5)
        h_data[i]["middle"]["nested"]["value"] = i * 3

    itemsize = h_data.dtype.itemsize
    d_input = cp.empty(num_items * itemsize, dtype=np.uint8)
    d_input.set(h_data.view(np.uint8))
    d_input = d_input.view(level3_dtype)

    d_output = cp.empty(itemsize, dtype=np.uint8)
    d_output = d_output.view(level3_dtype)

    h_init = np.void((0, (0.0, (0,))), dtype=level3_dtype)

    cuda.compute.reduce_into(d_input, d_output, sum_deeply_nested, num_items, h_init)

    result = d_output.view(np.uint8).get().view(level3_dtype)[0]

    expected_id = sum(i * 10 for i in range(num_items))
    expected_data = sum(float(i * 2.5) for i in range(num_items))
    expected_value = sum(i * 3 for i in range(num_items))

    assert result["id"] == expected_id
    assert np.isclose(result["middle"]["data"], expected_data)
    assert result["middle"]["nested"]["value"] == expected_value


def test_tuple_construction_in_zip_iterator():
    """Test tuple construction with ZipIterator combining nested structs."""
    point_dtype = np.dtype([("x", np.int32), ("y", np.int32)], align=True)
    color_dtype = np.dtype(
        [("r", np.uint8), ("g", np.uint8), ("b", np.uint8)], align=True
    )
    pixel_dtype = np.dtype(
        [("position", point_dtype), ("color", color_dtype)], align=True
    )

    def sum_pixels_with_tuples(p1, p2):
        # Use tuple syntax for both nested structs
        return (
            (p1.position.x + p2.position.x, p1.position.y + p2.position.y),
            (
                p1.color.r + p2.color.r,
                p1.color.g + p2.color.g,
                p1.color.b + p2.color.b,
            ),
        )

    num_items = 100

    d_points = cp.empty(num_items, dtype=point_dtype)
    d_colors = cp.empty(num_items, dtype=color_dtype)

    h_points = np.array([(i, i * 2) for i in range(num_items)], dtype=point_dtype)
    h_colors = np.array(
        [(i % 256, (i * 2) % 256, (i * 3) % 256) for i in range(num_items)],
        dtype=color_dtype,
    )

    d_points.set(h_points)
    d_colors.set(h_colors)

    zip_it = ZipIterator(d_points, d_colors)

    d_output = cp.empty(1, dtype=pixel_dtype)
    h_init = np.void(((0, 0), (0, 0, 0)), dtype=pixel_dtype)

    cuda.compute.reduce_into(
        zip_it, d_output, sum_pixels_with_tuples, num_items, h_init
    )

    result = d_output.get()[0]

    expected_x = sum(i for i in range(num_items))
    expected_y = sum(i * 2 for i in range(num_items))
    expected_r = sum(i % 256 for i in range(num_items)) % 256
    expected_g = sum((i * 2) % 256 for i in range(num_items)) % 256
    expected_b = sum((i * 3) % 256 for i in range(num_items)) % 256

    assert result["position"]["x"] == expected_x
    assert result["position"]["y"] == expected_y
    assert result["color"]["r"] == expected_r
    assert result["color"]["g"] == expected_g
    assert result["color"]["b"] == expected_b
