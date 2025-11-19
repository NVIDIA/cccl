# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import cupy as cp
import numpy as np

import cuda.compute
from cuda.compute import ZipIterator, gpu_struct


def test_reduce_nested_struct_direct():
    Inner = gpu_struct({"a": np.int32, "b": np.float32})
    Outer = gpu_struct({"x": np.int64, "inner": Inner})

    def sum_nested(s1, s2):
        return Outer(
            s1.x + s2.x, Inner(s1.inner.a + s2.inner.a, s1.inner.b + s2.inner.b)
        )

    num_items = 10

    h_data = np.zeros(num_items, dtype=Outer.dtype)
    for i in range(num_items):
        h_data[i]["x"] = i
        h_data[i]["inner"]["a"] = i * 2
        h_data[i]["inner"]["b"] = float(i * 3)

    itemsize = h_data.dtype.itemsize
    d_input = cp.empty(num_items * itemsize, dtype=np.uint8)
    d_input.set(h_data.view(np.uint8))
    d_input = d_input.view(Outer.dtype)

    d_output = cp.empty(itemsize, dtype=np.uint8)
    d_output = d_output.view(Outer.dtype)

    h_init = Outer(0, Inner(0, 0.0))

    cuda.compute.reduce_into(d_input, d_output, sum_nested, num_items, h_init)

    result = d_output.view(np.uint8).get().view(Outer.dtype)[0]

    expected_x = sum(range(num_items))
    expected_a = sum(i * 2 for i in range(num_items))
    expected_b = sum(float(i * 3) for i in range(num_items))

    assert result["x"] == expected_x
    assert result["inner"]["a"] == expected_a
    assert np.isclose(result["inner"]["b"], expected_b)


def test_nested_struct_inline():
    """Test creating nested structs using inline dictionary syntax."""
    # Create a struct with an inline nested struct definition
    Outer = gpu_struct({"x": np.int64, "inner": {"a": np.int32, "b": np.float32}})

    # Get the nested struct type from the outer struct for construction
    Inner = type(Outer(0, (0, 0.0)).inner)

    def sum_nested(s1, s2):
        return Outer(
            s1.x + s2.x, Inner(s1.inner.a + s2.inner.a, s1.inner.b + s2.inner.b)
        )

    num_items = 10

    h_data = np.zeros(num_items, dtype=Outer.dtype)
    for i in range(num_items):
        h_data[i]["x"] = i
        h_data[i]["inner"]["a"] = i * 2
        h_data[i]["inner"]["b"] = float(i * 3)

    itemsize = h_data.dtype.itemsize
    d_input = cp.empty(num_items * itemsize, dtype=np.uint8)
    d_input.set(h_data.view(np.uint8))
    d_input = d_input.view(Outer.dtype)

    d_output = cp.empty(itemsize, dtype=np.uint8)
    d_output = d_output.view(Outer.dtype)

    h_init = Outer(0, Inner(0, 0.0))

    cuda.compute.reduce_into(d_input, d_output, sum_nested, num_items, h_init)

    result = d_output.view(np.uint8).get().view(Outer.dtype)[0]

    expected_x = sum(range(num_items))
    expected_a = sum(i * 2 for i in range(num_items))
    expected_b = sum(float(i * 3) for i in range(num_items))

    assert result["x"] == expected_x
    assert result["inner"]["a"] == expected_a
    assert np.isclose(result["inner"]["b"], expected_b)


def test_nested_struct_in_zip_iterator():
    Point = gpu_struct({"x": np.int32, "y": np.int32})
    Color = gpu_struct({"r": np.uint8, "g": np.uint8, "b": np.uint8})
    Pixel = gpu_struct({"position": Point, "color": Color})

    def sum_pixels(p1, p2):
        return Pixel(
            Point(p1.position.x + p2.position.x, p1.position.y + p2.position.y),
            Color(
                p1.color.r + p2.color.r,
                p1.color.g + p2.color.g,
                p1.color.b + p2.color.b,
            ),
        )

    num_items = 100

    d_points = cp.empty(num_items, dtype=Point.dtype)
    d_colors = cp.empty(num_items, dtype=Color.dtype)

    h_points = np.array([(i, i * 2) for i in range(num_items)], dtype=Point.dtype)
    h_colors = np.array(
        [(i % 256, (i * 2) % 256, (i * 3) % 256) for i in range(num_items)],
        dtype=Color.dtype,
    )

    d_points.set(h_points)
    d_colors.set(h_colors)

    zip_it = ZipIterator(d_points, d_colors)

    d_output = cp.empty(1, dtype=Pixel.dtype)
    h_init = Pixel(Point(0, 0), Color(0, 0, 0))

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


def test_dict_init_nested_struct():
    """Test initializing a nested struct with a dictionary."""
    Inner = gpu_struct({"a": np.int32, "b": np.float32})
    Outer = gpu_struct({"x": np.int64, "inner": Inner})

    # Initialize with nested dictionary
    obj = Outer({"x": 42, "inner": {"a": 10, "b": 3.14}})

    assert obj.x == 42
    assert obj.inner.a == 10
    assert np.isclose(obj.inner.b, 3.14)


def test_dict_init_per_field():
    """Test initializing a struct with a dictionary for a nested field."""
    Inner = gpu_struct({"a": np.int32, "b": np.float32})
    Outer = gpu_struct({"x": np.int64, "inner": Inner})

    # Mix positional value with dictionary for nested field
    obj = Outer(42, {"a": 10, "b": 3.14})

    assert obj.x == 42
    assert obj.inner.a == 10
    assert np.isclose(obj.inner.b, 3.14)


def test_dict_init_deeply_nested():
    """Test initializing deeply nested structs (3+ levels) with dictionaries."""
    Level1 = gpu_struct({"value": np.int32})
    Level2 = gpu_struct({"data": np.float32, "nested": Level1})
    Level3 = gpu_struct({"id": np.int64, "middle": Level2})

    # Initialize with deeply nested dictionary
    obj = Level3({"id": 100, "middle": {"data": 2.5, "nested": {"value": 42}}})

    assert obj.id == 100
    assert np.isclose(obj.middle.data, 2.5)
    assert obj.middle.nested.value == 42


def test_dict_init_mixed():
    """Test mixed initialization with some dicts and some direct values."""
    Inner1 = gpu_struct({"a": np.int32, "b": np.int32})
    Inner2 = gpu_struct({"c": np.float32, "d": np.float32})
    Outer = gpu_struct({"x": np.int64, "inner1": Inner1, "inner2": Inner2})

    # Mix different initialization styles
    inner1_obj = Inner1(1, 2)  # Direct instantiation
    # Mix direct object and dict
    obj = Outer(100, inner1_obj, {"c": 3.0, "d": 4.0})

    assert obj.x == 100
    assert obj.inner1.a == 1
    assert obj.inner1.b == 2
    assert np.isclose(obj.inner2.c, 3.0)
    assert np.isclose(obj.inner2.d, 4.0)


def test_dict_init_with_reduction():
    """Test that dict-initialized structs work correctly in reductions."""
    Inner = gpu_struct({"a": np.int32, "b": np.float32})
    Outer = gpu_struct({"x": np.int64, "inner": Inner})

    def sum_nested(s1, s2):
        return Outer(
            s1.x + s2.x, Inner(s1.inner.a + s2.inner.a, s1.inner.b + s2.inner.b)
        )

    num_items = 10
    h_data = np.zeros(num_items, dtype=Outer.dtype)
    for i in range(num_items):
        h_data[i]["x"] = i
        h_data[i]["inner"]["a"] = i * 2
        h_data[i]["inner"]["b"] = float(i * 3)

    itemsize = h_data.dtype.itemsize
    d_input = cp.empty(num_items * itemsize, dtype=np.uint8)
    d_input.set(h_data.view(np.uint8))
    d_input = d_input.view(Outer.dtype)

    d_output = cp.empty(itemsize, dtype=np.uint8)
    d_output = d_output.view(Outer.dtype)

    # Use dictionary initialization for the init value
    h_init = Outer({"x": 0, "inner": {"a": 0, "b": 0.0}})

    cuda.compute.reduce_into(d_input, d_output, sum_nested, num_items, h_init)

    result = d_output.view(np.uint8).get().view(Outer.dtype)[0]

    expected_x = sum(range(num_items))
    expected_a = sum(i * 2 for i in range(num_items))
    expected_b = sum(float(i * 3) for i in range(num_items))

    assert result["x"] == expected_x
    assert result["inner"]["a"] == expected_a
    assert np.isclose(result["inner"]["b"], expected_b)


def test_nested_struct_tuple_construction():
    """Test constructing nested structs using tuple syntax in device functions."""
    Inner = gpu_struct({"a": np.int32, "b": np.float32})
    Outer = gpu_struct({"x": np.int64, "inner": Inner})

    def sum_nested_with_tuples(s1, s2):
        # Use tuple syntax instead of Inner(...)
        return Outer(s1.x + s2.x, (s1.inner.a + s2.inner.a, s1.inner.b + s2.inner.b))

    num_items = 10

    h_data = np.zeros(num_items, dtype=Outer.dtype)
    for i in range(num_items):
        h_data[i]["x"] = i
        h_data[i]["inner"]["a"] = i * 2
        h_data[i]["inner"]["b"] = float(i * 3)

    itemsize = h_data.dtype.itemsize
    d_input = cp.empty(num_items * itemsize, dtype=np.uint8)
    d_input.set(h_data.view(np.uint8))
    d_input = d_input.view(Outer.dtype)

    d_output = cp.empty(itemsize, dtype=np.uint8)
    d_output = d_output.view(Outer.dtype)

    h_init = Outer(0, Inner(0, 0.0))

    cuda.compute.reduce_into(
        d_input, d_output, sum_nested_with_tuples, num_items, h_init
    )

    result = d_output.view(np.uint8).get().view(Outer.dtype)[0]

    expected_x = sum(range(num_items))
    expected_a = sum(i * 2 for i in range(num_items))
    expected_b = sum(float(i * 3) for i in range(num_items))

    assert result["x"] == expected_x
    assert result["inner"]["a"] == expected_a
    assert np.isclose(result["inner"]["b"], expected_b)


def test_deeply_nested_tuple_construction():
    """Test constructing deeply nested structs (3 levels) using tuple syntax."""
    Level1 = gpu_struct({"value": np.int32})
    Level2 = gpu_struct({"data": np.float32, "nested": Level1})
    Level3 = gpu_struct({"id": np.int64, "middle": Level2})

    def sum_deeply_nested(v1, v2):
        # Use nested tuple syntax: (float, (int,))
        return Level3(
            v1.id + v2.id,
            (
                v1.middle.data + v2.middle.data,
                (v1.middle.nested.value + v2.middle.nested.value,),
            ),
        )

    num_items = 10

    h_data = np.zeros(num_items, dtype=Level3.dtype)
    for i in range(num_items):
        h_data[i]["id"] = i * 10
        h_data[i]["middle"]["data"] = float(i * 2.5)
        h_data[i]["middle"]["nested"]["value"] = i * 3

    itemsize = h_data.dtype.itemsize
    d_input = cp.empty(num_items * itemsize, dtype=np.uint8)
    d_input.set(h_data.view(np.uint8))
    d_input = d_input.view(Level3.dtype)

    d_output = cp.empty(itemsize, dtype=np.uint8)
    d_output = d_output.view(Level3.dtype)

    h_init = Level3(0, Level2(0.0, Level1(0)))

    cuda.compute.reduce_into(d_input, d_output, sum_deeply_nested, num_items, h_init)

    result = d_output.view(np.uint8).get().view(Level3.dtype)[0]

    expected_id = sum(i * 10 for i in range(num_items))
    expected_data = sum(float(i * 2.5) for i in range(num_items))
    expected_value = sum(i * 3 for i in range(num_items))

    assert result["id"] == expected_id
    assert np.isclose(result["middle"]["data"], expected_data)
    assert result["middle"]["nested"]["value"] == expected_value


def test_mixed_tuple_and_direct_construction():
    """Test mixing tuple construction with direct struct construction."""
    Inner1 = gpu_struct({"a": np.int32, "b": np.int32})
    Inner2 = gpu_struct({"c": np.float32, "d": np.float32})
    Outer = gpu_struct({"x": np.int64, "inner1": Inner1, "inner2": Inner2})

    def sum_mixed(s1, s2):
        # Mix direct struct construction with tuple construction
        return Outer(
            s1.x + s2.x,
            Inner1(s1.inner1.a + s2.inner1.a, s1.inner1.b + s2.inner1.b),
            (s1.inner2.c + s2.inner2.c, s1.inner2.d + s2.inner2.d),
        )

    num_items = 10

    h_data = np.zeros(num_items, dtype=Outer.dtype)
    for i in range(num_items):
        h_data[i]["x"] = i
        h_data[i]["inner1"]["a"] = i * 2
        h_data[i]["inner1"]["b"] = i * 3
        h_data[i]["inner2"]["c"] = float(i * 4)
        h_data[i]["inner2"]["d"] = float(i * 5)

    itemsize = h_data.dtype.itemsize
    d_input = cp.empty(num_items * itemsize, dtype=np.uint8)
    d_input.set(h_data.view(np.uint8))
    d_input = d_input.view(Outer.dtype)

    d_output = cp.empty(itemsize, dtype=np.uint8)
    d_output = d_output.view(Outer.dtype)

    h_init = Outer(0, Inner1(0, 0), Inner2(0.0, 0.0))

    cuda.compute.reduce_into(d_input, d_output, sum_mixed, num_items, h_init)

    result = d_output.view(np.uint8).get().view(Outer.dtype)[0]

    expected_x = sum(range(num_items))
    expected_a = sum(i * 2 for i in range(num_items))
    expected_b = sum(i * 3 for i in range(num_items))
    expected_c = sum(float(i * 4) for i in range(num_items))
    expected_d = sum(float(i * 5) for i in range(num_items))

    assert result["x"] == expected_x
    assert result["inner1"]["a"] == expected_a
    assert result["inner1"]["b"] == expected_b
    assert np.isclose(result["inner2"]["c"], expected_c)
    assert np.isclose(result["inner2"]["d"], expected_d)


def test_tuple_construction_in_zip_iterator():
    """Test tuple construction with ZipIterator combining nested structs."""
    Point = gpu_struct({"x": np.int32, "y": np.int32})
    Color = gpu_struct({"r": np.uint8, "g": np.uint8, "b": np.uint8})
    Pixel = gpu_struct({"position": Point, "color": Color})

    def sum_pixels_with_tuples(p1, p2):
        # Use tuple syntax for both nested structs
        return Pixel(
            (p1.position.x + p2.position.x, p1.position.y + p2.position.y),
            (
                p1.color.r + p2.color.r,
                p1.color.g + p2.color.g,
                p1.color.b + p2.color.b,
            ),
        )

    num_items = 100

    d_points = cp.empty(num_items, dtype=Point.dtype)
    d_colors = cp.empty(num_items, dtype=Color.dtype)

    h_points = np.array([(i, i * 2) for i in range(num_items)], dtype=Point.dtype)
    h_colors = np.array(
        [(i % 256, (i * 2) % 256, (i * 3) % 256) for i in range(num_items)],
        dtype=Color.dtype,
    )

    d_points.set(h_points)
    d_colors.set(h_colors)

    zip_it = ZipIterator(d_points, d_colors)

    d_output = cp.empty(1, dtype=Pixel.dtype)
    h_init = Pixel(Point(0, 0), Color(0, 0, 0))

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


def test_all_tuple_construction():
    """Test constructing a struct where all fields use tuple syntax."""
    Inner1 = gpu_struct({"a": np.int32})
    Inner2 = gpu_struct({"b": np.float32})
    Outer = gpu_struct({"field1": Inner1, "field2": Inner2})

    def sum_all_tuples(s1, s2):
        # All fields use tuple syntax
        return Outer((s1.field1.a + s2.field1.a,), (s1.field2.b + s2.field2.b,))

    num_items = 5

    h_data = np.zeros(num_items, dtype=Outer.dtype)
    for i in range(num_items):
        h_data[i]["field1"]["a"] = i
        h_data[i]["field2"]["b"] = float(i * 2)

    itemsize = h_data.dtype.itemsize
    d_input = cp.empty(num_items * itemsize, dtype=np.uint8)
    d_input.set(h_data.view(np.uint8))
    d_input = d_input.view(Outer.dtype)

    d_output = cp.empty(itemsize, dtype=np.uint8)
    d_output = d_output.view(Outer.dtype)

    h_init = Outer(Inner1(0), Inner2(0.0))

    cuda.compute.reduce_into(d_input, d_output, sum_all_tuples, num_items, h_init)

    result = d_output.view(np.uint8).get().view(Outer.dtype)[0]

    expected_a = sum(range(num_items))
    expected_b = sum(float(i * 2) for i in range(num_items))

    assert result["field1"]["a"] == expected_a
    assert np.isclose(result["field2"]["b"], expected_b)
