# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for CompiledOp and CompiledIterator (BYOC - Bring Your Own Compiler)."""

import cupy as cp
import numpy as np
import pytest

from cuda.compute import CompiledIterator, CompiledOp, OpKind, reduce_into, types
from cuda.core import Device, Program, ProgramOptions


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


def test_compiled_op_reduce_int32():
    """Test CompiledOp with reduce_into using int32."""
    arch = get_arch()
    add_ltoir = compile_to_ltoir(ADD_OP_SOURCE, arch)

    add_op = CompiledOp(
        ltoir=add_ltoir,
        name="my_add",
        arg_types=(types.int32, types.int32),
        return_type=types.int32,
    )

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

    add_op = CompiledOp(
        ltoir=add_ltoir,
        name="my_add_i64",
        arg_types=(types.int64, types.int64),
        return_type=types.int64,
    )

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

    add_op = CompiledOp(
        ltoir=add_ltoir,
        name="my_add",
        arg_types=(types.int32, types.int32),
        return_type=types.int32,
    )

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

    # State: starting offset
    offset = 10
    state = np.array([offset], dtype=np.int64).tobytes()

    counting_iter = CompiledIterator(
        state=state,
        state_alignment=8,
        value_type=types.int64,
        advance=("advance", advance_ltoir),
        input_dereference=("dereference", deref_ltoir),
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

    # State: starting offset of 0
    offset = 0
    state = np.array([offset], dtype=np.int64).tobytes()

    counting_iter = CompiledIterator(
        state=state,
        state_alignment=8,
        value_type=types.int64,
        advance=("advance", advance_ltoir),
        input_dereference=("dereference", deref_ltoir),
    )

    d_out = cp.array([0], dtype=np.int64)
    h_init = np.array([0], dtype=np.int64)

    reduce_into(counting_iter, d_out, OpKind.PLUS, 10, h_init)

    result = d_out.get()[0]
    expected = sum(range(10))  # 0+1+2+...+9 = 45
    assert result == expected


def test_compiled_op_validation_errors():
    """Test that CompiledOp raises appropriate validation errors."""
    arch = get_arch()
    add_ltoir = compile_to_ltoir(ADD_OP_SOURCE, arch)

    # Test invalid ltoir type
    with pytest.raises(TypeError, match="ltoir must be bytes"):
        CompiledOp(
            ltoir="not bytes",
            name="my_add",
            arg_types=(types.int32, types.int32),
            return_type=types.int32,
        )

    # Test empty ltoir
    with pytest.raises(ValueError, match="ltoir cannot be empty"):
        CompiledOp(
            ltoir=b"",
            name="my_add",
            arg_types=(types.int32, types.int32),
            return_type=types.int32,
        )

    # Test invalid name type
    with pytest.raises(TypeError, match="name must be str"):
        CompiledOp(
            ltoir=add_ltoir,
            name=123,
            arg_types=(types.int32, types.int32),
            return_type=types.int32,
        )

    # Test empty name
    with pytest.raises(ValueError, match="name cannot be empty"):
        CompiledOp(
            ltoir=add_ltoir,
            name="",
            arg_types=(types.int32, types.int32),
            return_type=types.int32,
        )

    # Test invalid arg_types
    with pytest.raises(TypeError, match="arg_types must be a tuple"):
        CompiledOp(
            ltoir=add_ltoir,
            name="my_add",
            arg_types=[types.int32, types.int32],
            return_type=types.int32,
        )


def test_compiled_iterator_validation_errors():
    """Test that CompiledIterator raises appropriate validation errors."""
    arch = get_arch()
    advance_ltoir = compile_to_ltoir(ADVANCE_SOURCE, arch)
    deref_ltoir = compile_to_ltoir(DEREF_SOURCE, arch)
    state = np.array([0], dtype=np.int64).tobytes()

    # Test invalid state type
    with pytest.raises(TypeError, match="state must be bytes"):
        CompiledIterator(
            state="not bytes",
            state_alignment=8,
            value_type=types.int64,
            advance=("advance", advance_ltoir),
            input_dereference=("dereference", deref_ltoir),
        )

    # Test invalid alignment
    with pytest.raises(ValueError, match="state_alignment must be a power of 2"):
        CompiledIterator(
            state=state,
            state_alignment=3,
            value_type=types.int64,
            advance=("advance", advance_ltoir),
            input_dereference=("dereference", deref_ltoir),
        )

    # Test missing dereference
    with pytest.raises(ValueError, match="At least one of input_dereference"):
        CompiledIterator(
            state=state,
            state_alignment=8,
            value_type=types.int64,
            advance=("advance", advance_ltoir),
        )
