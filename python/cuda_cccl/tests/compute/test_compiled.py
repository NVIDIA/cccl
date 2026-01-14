# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for CompiledOp and CompiledIterator (BYOC - Bring Your Own Compiler).

These tests verify that the BYOC path works without requiring Numba.
All tests in this module are marked with 'no_numba' and use the numba_absent
fixture to ensure no accidental Numba imports occur during execution.
"""

import builtins

import cupy as cp
import numpy as np
import pytest

from cuda.compute import (
    CompiledIterator,
    CompiledOp,
    OpKind,
    gpu_struct,
    reduce_into,
    types,
)
from cuda.core import Device, Program, ProgramOptions

# Mark all tests in this module as no_numba
pytestmark = pytest.mark.no_numba


@pytest.fixture(autouse=True)
def numba_absent(monkeypatch):
    """Fixture that blocks numba imports to verify BYOC path works without Numba."""
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "numba" or name.startswith("numba."):
            raise ModuleNotFoundError(f"No module named '{name}'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


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


# C++ source for a custom add operation (CCCL ABI: all void* parameters)
ADD_OP_SOURCE = """
extern "C" __device__ void my_add(void* a_ptr, void* b_ptr, void* result_ptr) {
    int a = *static_cast<int*>(a_ptr);
    int b = *static_cast<int*>(b_ptr);
    *static_cast<int*>(result_ptr) = a + b;
}
"""

ADD_OP_INT64_SOURCE = """
extern "C" __device__ void my_add_i64(void* a_ptr, void* b_ptr, void* result_ptr) {
    long long a = *static_cast<long long*>(a_ptr);
    long long b = *static_cast<long long*>(b_ptr);
    *static_cast<long long*>(result_ptr) = a + b;
}
"""

# C++ source for counting iterator advance/dereference
ADVANCE_SOURCE = """
extern "C" __device__ void advance(void* state_ptr, void* offset_ptr) {
    long long* state = static_cast<long long*>(state_ptr);
    long long offset = *static_cast<long long*>(offset_ptr);
    *state += offset;
}
"""

DEREF_SOURCE = """
extern "C" __device__ void dereference(void* state_ptr, void* result_ptr) {
    long long* state = static_cast<long long*>(state_ptr);
    long long* result = static_cast<long long*>(result_ptr);
    *result = *state;
}
"""


def test_import_numba_should_fail():
    # cuda.compute should be usable with compiled ops/iterators
    # without numba present.
    with pytest.raises(ModuleNotFoundError):
        import numba  # noqa: F401


def test_compiled_op_reduce_int32():
    """Test CompiledOp with reduce_into using int32."""
    arch = get_arch()
    add_ltoir = compile_to_ltoir(ADD_OP_SOURCE, arch)

    # New simplified API: just ltoir and name
    add_op = CompiledOp(add_ltoir, "my_add")

    d_in = cp.array([1, 2, 3, 4, 5], dtype=np.int32)
    d_out = cp.array([0], dtype=np.int32)
    h_init = np.array([0], dtype=np.int32)

    reduce_into(d_in, d_out, add_op, len(d_in), h_init)

    result = d_out.get()[0]
    expected = 15  # 1+2+3+4+5
    assert result == expected


def test_compiled_op_reduce_int64():
    """Test CompiledOp with reduce_into using int64."""
    arch = get_arch()
    add_ltoir = compile_to_ltoir(ADD_OP_INT64_SOURCE, arch)

    add_op = CompiledOp(add_ltoir, "my_add_i64")

    d_in = cp.array([10, 20, 30, 40, 50], dtype=np.int64)
    d_out = cp.array([0], dtype=np.int64)
    h_init = np.array([0], dtype=np.int64)

    reduce_into(d_in, d_out, add_op, len(d_in), h_init)

    result = d_out.get()[0]
    expected = 150
    assert result == expected


def test_compiled_op_reduce_with_init():
    """Test CompiledOp with non-zero initial value."""
    arch = get_arch()
    add_ltoir = compile_to_ltoir(ADD_OP_SOURCE, arch)

    add_op = CompiledOp(add_ltoir, "my_add")

    d_in = cp.array([1, 2, 3, 4, 5], dtype=np.int32)
    d_out = cp.array([0], dtype=np.int32)
    h_init = np.array([100], dtype=np.int32)

    reduce_into(d_in, d_out, add_op, len(d_in), h_init)

    result = d_out.get()[0]
    expected = 115  # 100 + 1+2+3+4+5
    assert result == expected


def test_compiled_iterator_counting():
    """Test CompiledIterator as a counting iterator with reduce_into."""
    arch = get_arch()
    advance_ltoir = compile_to_ltoir(ADVANCE_SOURCE, arch)
    deref_ltoir = compile_to_ltoir(DEREF_SOURCE, arch)

    # Create CompiledOp objects for advance and dereference
    advance_op = CompiledOp(advance_ltoir, "advance")
    deref_op = CompiledOp(deref_ltoir, "dereference")

    # New API: numpy state with auto-inferred alignment
    counting_iter = CompiledIterator(
        state=np.int64(10),
        value_type=types.int64,
        advance=advance_op,
        input_dereference=deref_op,
    )

    d_out = cp.array([0], dtype=np.int64)
    h_init = np.array([0], dtype=np.int64)

    reduce_into(counting_iter, d_out, OpKind.PLUS, 5, h_init)

    result = d_out.get()[0]
    expected = 10 + 11 + 12 + 13 + 14  # sum of 10..14
    assert result == expected


def test_compiled_iterator_different_offset():
    """Test CompiledIterator with different starting offset."""
    arch = get_arch()
    advance_ltoir = compile_to_ltoir(ADVANCE_SOURCE, arch)
    deref_ltoir = compile_to_ltoir(DEREF_SOURCE, arch)

    advance_op = CompiledOp(advance_ltoir, "advance")
    deref_op = CompiledOp(deref_ltoir, "dereference")

    # State: starting offset of 0
    counting_iter = CompiledIterator(
        state=np.int64(0),
        value_type=types.int64,
        advance=advance_op,
        input_dereference=deref_op,
    )

    d_out = cp.array([0], dtype=np.int64)
    h_init = np.array([0], dtype=np.int64)

    reduce_into(counting_iter, d_out, OpKind.PLUS, 10, h_init)

    result = d_out.get()[0]
    expected = sum(range(10))  # 0+1+2+...+9 = 45
    assert result == expected


def test_compiled_iterator_with_raw_bytes():
    """Test CompiledIterator with raw bytes state (requires explicit alignment)."""
    arch = get_arch()
    advance_ltoir = compile_to_ltoir(ADVANCE_SOURCE, arch)
    deref_ltoir = compile_to_ltoir(DEREF_SOURCE, arch)

    advance_op = CompiledOp(advance_ltoir, "advance")
    deref_op = CompiledOp(deref_ltoir, "dereference")

    # Raw bytes with explicit alignment
    state = np.int64(5).tobytes()
    counting_iter = CompiledIterator(
        state=state,
        value_type=types.int64,
        advance=advance_op,
        input_dereference=deref_op,
        state_alignment=8,
    )

    d_out = cp.array([0], dtype=np.int64)
    h_init = np.array([0], dtype=np.int64)

    reduce_into(counting_iter, d_out, OpKind.PLUS, 5, h_init)

    result = d_out.get()[0]
    expected = 5 + 6 + 7 + 8 + 9  # sum of 5..9
    assert result == expected


def test_compiled_op_validation_errors():
    """Test that CompiledOp raises appropriate validation errors."""
    arch = get_arch()
    add_ltoir = compile_to_ltoir(ADD_OP_SOURCE, arch)

    # Test invalid ltoir type
    with pytest.raises(TypeError, match="ltoir must be bytes"):
        CompiledOp("not bytes", "my_add")

    # Test empty ltoir
    with pytest.raises(ValueError, match="ltoir cannot be empty"):
        CompiledOp(b"", "my_add")

    # Test invalid name type
    with pytest.raises(TypeError, match="name must be str"):
        CompiledOp(add_ltoir, 123)

    # Test empty name
    with pytest.raises(ValueError, match="name cannot be empty"):
        CompiledOp(add_ltoir, "")


def test_compiled_iterator_validation_errors():
    """Test that CompiledIterator raises appropriate validation errors."""
    arch = get_arch()
    advance_ltoir = compile_to_ltoir(ADVANCE_SOURCE, arch)
    deref_ltoir = compile_to_ltoir(DEREF_SOURCE, arch)

    advance_op = CompiledOp(advance_ltoir, "advance")
    deref_op = CompiledOp(deref_ltoir, "dereference")

    # Test invalid state type
    with pytest.raises(TypeError, match="state must be bytes, numpy scalar"):
        CompiledIterator(
            state="not bytes",
            value_type=types.int64,
            advance=advance_op,
            input_dereference=deref_op,
        )

    # Test invalid alignment
    with pytest.raises(ValueError, match="state_alignment must be a power of 2"):
        CompiledIterator(
            state=np.int64(0),
            value_type=types.int64,
            advance=advance_op,
            input_dereference=deref_op,
            state_alignment=3,
        )

    # Test missing dereference
    with pytest.raises(ValueError, match="At least one of input_dereference"):
        CompiledIterator(
            state=np.int64(0),
            value_type=types.int64,
            advance=advance_op,
        )

    # Test raw bytes without alignment
    with pytest.raises(ValueError, match="state_alignment is required"):
        CompiledIterator(
            state=b"\x00\x00\x00\x00",
            value_type=types.int64,
            advance=advance_op,
            input_dereference=deref_op,
        )

    # Test non-CompiledOp advance
    with pytest.raises(TypeError, match="advance must be a CompiledOp"):
        CompiledIterator(
            state=np.int64(0),
            value_type=types.int64,
            advance=("advance", advance_ltoir),  # Old tuple format
            input_dereference=deref_op,
        )


# C++ source for struct operations
# Must match the Python gpu_struct layout exactly
STRUCT_ADD_SOURCE = """
struct Point {
    int x;
    int y;
};

extern "C" __device__ void point_add(void* a_ptr, void* b_ptr, void* result_ptr) {
    const Point& a = *static_cast<const Point*>(a_ptr);
    const Point& b = *static_cast<const Point*>(b_ptr);
    Point& result = *static_cast<Point*>(result_ptr);
    result.x = a.x + b.x;
    result.y = a.y + b.y;
}
"""


def test_compiled_op_with_gpu_struct():
    """Test CompiledOp with gpu_struct types for reduce_into."""
    arch = get_arch()
    add_ltoir = compile_to_ltoir(STRUCT_ADD_SOURCE, arch)

    # Define a gpu_struct type
    Point = gpu_struct({"x": np.int32, "y": np.int32}, name="Point")

    # Simplified API - just ltoir and name
    point_add_op = CompiledOp(add_ltoir, "point_add")

    # Create test data - use empty() + set() for structured dtypes with cupy
    h_points = np.array([(1, 2), (3, 4), (5, 6), (7, 8)], dtype=Point._dtype)
    d_points = cp.empty(len(h_points), dtype=Point._dtype)
    d_points.set(h_points)

    d_out = cp.empty(1, dtype=Point._dtype)
    h_init = np.zeros(1, dtype=Point._dtype)

    reduce_into(d_points, d_out, point_add_op, len(h_points), h_init)

    result = d_out.get()[0]
    expected_x = 1 + 3 + 5 + 7  # 16
    expected_y = 2 + 4 + 6 + 8  # 20
    assert result["x"] == expected_x
    assert result["y"] == expected_y


def test_compiled_op_properties():
    """Test CompiledOp property accessors."""
    arch = get_arch()
    add_ltoir = compile_to_ltoir(ADD_OP_SOURCE, arch)

    op = CompiledOp(add_ltoir, "my_add")

    assert op.name == "my_add"
    assert op.ltoir == add_ltoir
    assert op.func is None  # No underlying callable for compiled ops


def test_compiled_iterator_properties():
    """Test CompiledIterator property accessors."""
    arch = get_arch()
    advance_ltoir = compile_to_ltoir(ADVANCE_SOURCE, arch)
    deref_ltoir = compile_to_ltoir(DEREF_SOURCE, arch)

    advance_op = CompiledOp(advance_ltoir, "advance")
    deref_op = CompiledOp(deref_ltoir, "dereference")

    it = CompiledIterator(
        state=np.int64(42),
        value_type=types.int64,
        advance=advance_op,
        input_dereference=deref_op,
    )

    assert it.state == np.int64(42).tobytes()
    assert it.state_alignment == 8  # Auto-inferred from int64
    assert it.value_type == types.int64
    assert it.is_input_iterator is True
    assert it.is_output_iterator is False
