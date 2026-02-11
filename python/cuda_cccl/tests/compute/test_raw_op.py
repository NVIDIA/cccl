# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import re
import struct

import cupy as cp
import numpy as np
import pytest

from cuda.compute import types
from cuda.compute.op import RawOp
from cuda.core import Device, Program, ProgramOptions

# Mark all tests in this module as no_numba
pytestmark = pytest.mark.no_numba


def get_arch():
    """Get the SM architecture string for the current device."""
    device = Device()
    device.set_current()
    cc_major, cc_minor = device.compute_capability
    return f"sm_{cc_major}{cc_minor}"


def compile_to_ltoir(source: str, arch: str) -> bytes:
    """Compile C++ source to LTOIR using cuda.core."""
    opts = ProgramOptions(
        arch=arch,
        relocatable_device_code=True,
        link_time_optimization=True,
    )
    prog = Program(source, "c++", options=opts)
    return prog.compile("ltoir").code


def extract_function_name(source: str) -> str:
    """Extract function name from C++ source."""
    match = re.search(r'extern\s+"C"\s+__device__\s+\w+\s+(\w+)\s*\(', source)
    if match:
        return match.group(1)
    raise ValueError("Could not extract function name from C++ source.")


def make_cpp_op(source: str, name: str = None) -> RawOp:
    """
    Compile C++ source to LTOIR and create a stateless RawOp.

    Args:
        source: C++ source code containing the operator function
        name: Optional function name. If not provided, will be extracted from source

    Returns:
        RawOp instance ready to use with CCCL algorithms
    """
    if name is None:
        name = extract_function_name(source)

    arch = get_arch()
    ltoir = compile_to_ltoir(source, arch)

    return RawOp(ltoir=ltoir, name=name)


def make_cpp_stateful_op(
    source: str, state: bytes, name: str = None, state_alignment: int = 8
) -> RawOp:
    """
    Compile C++ source to LTOIR and create a stateful RawOp.

    Args:
        source: C++ source code containing the operator function
        state: State data as bytes
        name: Optional function name. If not provided, will be extracted from source
        state_alignment: Memory alignment for state (default: 8)

    Returns:
        RawOp instance ready to use with CCCL algorithms
    """
    if name is None:
        name = extract_function_name(source)

    arch = get_arch()
    ltoir = compile_to_ltoir(source, arch)

    return RawOp(
        ltoir=ltoir,
        name=name,
        state=state,
        state_alignment=state_alignment,
    )


def test_cpp_op_basic_add():
    """Test a basic C++ addition operator with reduce_into."""
    cpp_source = """
    extern "C" __device__ void add_op(void* a, void* b, void* result) {
        *static_cast<int*>(result) = *static_cast<int*>(a) + *static_cast<int*>(b);
    }
    """

    op = make_cpp_op(cpp_source, "add_op")

    # Create test data
    num_items = 100
    h_input = np.arange(num_items, dtype=np.int32)
    d_input = cp.array(h_input)
    d_output = cp.zeros(1, dtype=np.int32)

    # Use the custom op with reduce_into
    import cuda.compute

    h_init = np.array(0, dtype=np.int32)
    cuda.compute.reduce_into(d_input, d_output, op, num_items, h_init)

    # Verify result
    result = d_output.get()[0]
    expected = np.sum(h_input)
    assert result == expected, f"Expected {expected}, got {result}"


def test_cpp_op_max():
    """Test a C++ max operator with reduce_into."""
    cpp_source = """
    extern "C" __device__ void max_op(void* a, void* b, void* result) {
        float va = *static_cast<float*>(a);
        float vb = *static_cast<float*>(b);
        *static_cast<float*>(result) = va > vb ? va : vb;
    }
    """

    op = make_cpp_op(cpp_source)

    # Create test data
    num_items = 100
    h_input = np.random.randn(num_items).astype(np.float32)
    d_input = cp.array(h_input)
    d_output = cp.zeros(1, dtype=np.float32)

    # Use the custom op with reduce_into
    import cuda.compute

    h_init = np.array(-np.inf, dtype=np.float32)
    cuda.compute.reduce_into(d_input, d_output, op, num_items, h_init)

    # Verify result
    result = d_output.get()[0]
    expected = np.max(h_input)
    assert np.isclose(result, expected), f"Expected {expected}, got {result}"


def test_cpp_op_multiply():
    """Test a C++ multiply operator."""
    cpp_source = """
    extern "C" __device__ void multiply(void* a, void* b, void* result) {
        *static_cast<int*>(result) = *static_cast<int*>(a) * *static_cast<int*>(b);
    }
    """

    op = make_cpp_op(cpp_source, "multiply")

    # Create test data - use small numbers to avoid overflow
    num_items = 5
    h_input = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    d_input = cp.array(h_input)
    d_output = cp.zeros(1, dtype=np.int32)

    # Use the custom op with reduce_into
    import cuda.compute

    h_init = np.array(1, dtype=np.int32)
    cuda.compute.reduce_into(d_input, d_output, op, num_items, h_init)

    # Verify result
    result = d_output.get()[0]
    expected = np.prod(h_input)
    assert result == expected, f"Expected {expected}, got {result}"


def test_cpp_op_complex_logic():
    """Test a C++ operator with more complex logic - bitwise OR (associative)."""
    cpp_source = """
    extern "C" __device__ void bitwise_or(void* a, void* b, void* result) {
        // Bitwise OR is associative: (a | b) | c = a | (b | c)
        int va = *static_cast<int*>(a);
        int vb = *static_cast<int*>(b);
        *static_cast<int*>(result) = va | vb;
    }
    """

    op = make_cpp_op(cpp_source, "bitwise_or")

    # Create test data with specific bit patterns
    num_items = 5
    h_input = np.array([1, 2, 4, 8, 16], dtype=np.int32)  # Powers of 2
    d_input = cp.array(h_input)
    d_output = cp.zeros(1, dtype=np.int32)

    # Use the custom op with reduce_into
    import cuda.compute

    h_init = np.array(0, dtype=np.int32)
    cuda.compute.reduce_into(d_input, d_output, op, num_items, h_init)

    # Expected: 1 | 2 | 4 | 8 | 16 = 31 (all bits set)
    result = d_output.get()[0]
    expected = 31
    assert result == expected, f"Expected {expected}, got {result}"


def test_cpp_op_different_types():
    """Test C++ operator with different numeric types."""
    cpp_source = """
    extern "C" __device__ void add_doubles(void* a, void* b, void* result) {
        *static_cast<double*>(result) = *static_cast<double*>(a) + *static_cast<double*>(b);
    }
    """

    op = make_cpp_op(cpp_source, "add_doubles")

    # Create test data
    num_items = 50
    h_input = np.random.randn(num_items).astype(np.float64)
    d_input = cp.array(h_input)
    d_output = cp.zeros(1, dtype=np.float64)

    # Use the custom op with reduce_into
    import cuda.compute

    h_init = np.array(0.0, dtype=np.float64)
    cuda.compute.reduce_into(d_input, d_output, op, num_items, h_init)

    # Verify result
    result = d_output.get()[0]
    expected = np.sum(h_input)
    assert np.isclose(result, expected), f"Expected {expected}, got {result}"


def test_cpp_op_name_extraction():
    """Test that function name is correctly extracted from C++ source."""
    cpp_source = """
    extern "C" __device__ void my_function_name(void* a, void* b, void* result) {
        *static_cast<int*>(result) = *static_cast<int*>(a) + *static_cast<int*>(b);
    }
    """

    # Don't provide name - it should be extracted
    op = make_cpp_op(cpp_source)

    # Create test data
    num_items = 10
    h_input = np.arange(num_items, dtype=np.int32)
    d_input = cp.array(h_input)
    d_output = cp.zeros(1, dtype=np.int32)

    # Use the custom op with reduce_into
    import cuda.compute

    h_init = np.array(0, dtype=np.int32)
    cuda.compute.reduce_into(d_input, d_output, op, num_items, h_init)

    # Verify result
    result = d_output.get()[0]
    expected = np.sum(h_input)
    assert result == expected, f"Expected {expected}, got {result}"


def test_cpp_op_min():
    """Test a C++ min operator."""
    cpp_source = """
    extern "C" __device__ void min_op(void* a, void* b, void* result) {
        int va = *static_cast<int*>(a);
        int vb = *static_cast<int*>(b);
        *static_cast<int*>(result) = va < vb ? va : vb;
    }
    """

    op = make_cpp_op(cpp_source, "min_op")

    # Create test data
    num_items = 100
    h_input = np.random.randint(-1000, 1000, num_items, dtype=np.int32)
    d_input = cp.array(h_input)
    d_output = cp.zeros(1, dtype=np.int32)

    # Use the custom op with reduce_into
    import cuda.compute

    h_init = np.array(np.iinfo(np.int32).max, dtype=np.int32)
    cuda.compute.reduce_into(d_input, d_output, op, num_items, h_init)

    # Verify result
    result = d_output.get()[0]
    expected = np.min(h_input)
    assert result == expected, f"Expected {expected}, got {result}"


def test_cpp_op_with_struct():
    """Test a C++ operator that works with struct types."""
    from cuda.compute import gpu_struct

    # Define a simple 2D point struct
    Point = gpu_struct({"x": np.int32, "y": np.int32})

    # C++ operator to add two points (field by field)
    cpp_source = """
    struct Point {
        int x;
        int y;
    };

    extern "C" __device__ void add_points(void* a, void* b, void* result) {
        Point* pa = static_cast<Point*>(a);
        Point* pb = static_cast<Point*>(b);
        Point* pr = static_cast<Point*>(result);
        pr->x = pa->x + pb->x;
        pr->y = pa->y + pb->y;
    }
    """

    op = make_cpp_op(cpp_source, "add_points")

    # Create test data
    num_items = 10
    h_data = np.zeros(num_items, dtype=Point.dtype)
    for i in range(num_items):
        h_data[i]["x"] = i
        h_data[i]["y"] = i * 2

    # Convert to device arrays using uint8 view
    itemsize = h_data.dtype.itemsize
    d_input = cp.empty(num_items * itemsize, dtype=np.uint8)
    d_input.set(h_data.view(np.uint8))
    d_input = d_input.view(Point.dtype)

    d_output = cp.empty(itemsize, dtype=np.uint8)
    d_output = d_output.view(Point.dtype)

    # Initial point (0, 0)
    h_init = Point(0, 0)

    # Use the custom op with reduce_into
    import cuda.compute

    cuda.compute.reduce_into(d_input, d_output, op, num_items, h_init)

    # Verify result
    result = d_output.view(np.uint8).get().view(Point.dtype)[0]
    expected_x = sum(range(num_items))  # 0+1+2+...+9 = 45
    expected_y = sum(i * 2 for i in range(num_items))  # 0+2+4+...+18 = 90

    assert result["x"] == expected_x, f"Expected x={expected_x}, got {result['x']}"
    assert result["y"] == expected_y, f"Expected y={expected_y}, got {result['y']}"


def test_cpp_op_with_transform_iterator():
    """Test that RawOp works with TransformIterator."""
    from cuda.compute import OpKind, TransformIterator

    # C++ unary operator that doubles a value
    cpp_source = """
    extern "C" __device__ void double_op(void* input, void* result) {
        *static_cast<int*>(result) = *static_cast<int*>(input) * 2;
    }
    """

    op = make_cpp_op(cpp_source, "double_op")

    # Create input data
    num_items = 10
    h_input = np.arange(num_items, dtype=np.int32)
    d_input = cp.array(h_input)

    # Create transform iterator with RawOp
    transform_iter = TransformIterator(d_input, op, value_type=types.int32)

    # Use the transform iterator with reduce
    import cuda.compute

    d_output = cp.zeros(1, dtype=np.int32)
    h_init = np.array(0, dtype=np.int32)

    # Sum the doubled values using built-in PLUS operator
    cuda.compute.reduce_into(transform_iter, d_output, OpKind.PLUS, num_items, h_init)

    # Verify result: sum of (0*2, 1*2, 2*2, ..., 9*2) = 2 * sum(0..9) = 2 * 45 = 90
    result = d_output.get()[0]
    expected = 2 * np.sum(h_input)
    assert result == expected, f"Expected {expected}, got {result}"


def test_cpp_stateful_op_reduce_with_constant():
    """Test stateful RawOp with a simple stateful reduce."""
    # State: a single int32 constant value (10) on device
    d_constant = cp.array([10], dtype=np.int32)
    constant_ptr = d_constant.__cuda_array_interface__["data"][0]
    state_data = struct.pack("P", constant_ptr)
    state_alignment = np.dtype(np.intp).alignment

    # C++ operator that adds inputs plus reads constant from state
    cpp_source = """
    extern "C" __device__ void add_with_state_constant(void* state, void* a, void* b, void* result) {
        // Extract constant pointer from state
        int* constant_ptr = *reinterpret_cast<int**>(state);
        int constant = *constant_ptr;

        int va = *static_cast<int*>(a);
        int vb = *static_cast<int*>(b);
        *static_cast<int*>(result) = va + vb + constant;
    }
    """

    op = make_cpp_stateful_op(
        cpp_source, state_data, "add_with_state_constant", state_alignment
    )

    # Create test data
    num_items = 5
    h_input = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    d_input = cp.array(h_input)
    d_output = cp.zeros(1, dtype=np.int32)

    # Use the stateful op with reduce_into
    import cuda.compute

    h_init = np.array(0, dtype=np.int32)
    cuda.compute.reduce_into(d_input, d_output, op, num_items, h_init)

    # Get result
    result = d_output.get()[0]
    # Each reduction adds 10, so we expect input sum + some multiple of 10
    # The exact value depends on tree structure, but should be > sum(inputs)
    sum_inputs = np.sum(h_input)
    assert result > sum_inputs, f"Expected result > {sum_inputs}, got {result}"


def test_cpp_stateful_op_select_with_counter():
    """Test stateful RawOp with select_if that atomically updates a counter."""
    # Create a device counter initialized to 0
    d_counter = cp.zeros(1, dtype=np.int32)

    # State: pointer to the counter
    counter_ptr = d_counter.__cuda_array_interface__["data"][0]
    state_data = struct.pack("P", counter_ptr)  # Pack pointer as bytes

    # Use proper pointer alignment for the platform
    state_alignment = np.dtype(np.intp).alignment

    # C++ select operator that counts selected items
    # Selects even numbers and atomically increments counter for each selection
    cpp_source = """
    extern "C" __device__ void select_even_with_count(void* state, void* input, void* result) {
        // Extract counter pointer from state
        int* counter = *reinterpret_cast<int**>(state);

        // Get input value
        int value = *static_cast<int*>(input);

        // Check if even
        bool is_even = (value % 2 == 0);

        // If selected, atomically increment the counter
        if (is_even) {
            atomicAdd(counter, 1);
        }

        // Store result as bool (uint8)
        *static_cast<unsigned char*>(result) = is_even ? 1 : 0;
    }
    """

    op = make_cpp_stateful_op(
        cpp_source,
        state_data,
        "select_even_with_count",
        state_alignment,
    )

    # Create test data: 0 to 19
    num_items = 20
    h_input = np.arange(num_items, dtype=np.int32)
    d_input = cp.array(h_input)

    # Allocate output arrays
    d_output = cp.empty(num_items, dtype=np.int32)
    d_num_selected = cp.zeros(1, dtype=np.int32)

    # Run select
    import cuda.compute

    cuda.compute.select(d_input, d_output, d_num_selected, op, num_items)

    # Get results
    num_selected = d_num_selected.get()[0]
    counter_value = d_counter.get()[0]

    # Verify: should have selected 10 even numbers (0, 2, 4, ..., 18)
    expected_count = 10
    assert num_selected == expected_count, (
        f"Expected {expected_count} selected, got {num_selected}"
    )
    assert counter_value == expected_count, (
        f"Expected counter={expected_count}, got {counter_value}"
    )

    # Verify the selected values are correct
    selected_values = d_output.get()[:num_selected]
    expected_selected = np.arange(0, 20, 2, dtype=np.int32)
    assert np.array_equal(selected_values, expected_selected), (
        "Selected values don't match"
    )
