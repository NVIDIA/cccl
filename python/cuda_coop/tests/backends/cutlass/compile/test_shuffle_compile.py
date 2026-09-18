# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

"""Shuffle compiler gates reject unsupported routes before linking."""

import pytest

cutlass = pytest.importorskip("cutlass")
from cutlass import cute
from cutlass.base_dsl.compiler import GPUArch
from cutlass.cute.runtime import make_ptr

from cuda import coop
from cuda.coop import cutlass as cutlass_coop
from cuda.coop.cutlass._compiler._types import ALL_PROVIDER_TYPES

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.compile]


def _compile(kernel, dtype=cutlass.Int32, block=(8, 3, 2)):
    @cute.jit
    def launch(memory: cute.Pointer):
        kernel(memory).launch(grid=1, block=block)

    pointer = make_ptr(dtype, 0, cute.AddressSpace.gmem, assumed_align=16)
    return cute.compile[(GPUArch("sm_80"),)](launch, pointer)


@pytest.mark.parametrize("dtype", tuple(ALL_PROVIDER_TYPES))
@pytest.mark.parametrize("mode", ("offset", "rotate", "up", "down"))
def test_profiles(dtype, mode):
    @cute.kernel
    def kernel(memory: cute.Pointer):
        value = dtype(cute.arch.thread_idx()[0])
        if cutlass.const_expr(mode in ("up", "down")):
            payload = cutlass_coop.ThreadData(
                2, dtype=dtype, values=[value, value], alignment=64
            )
            result = cutlass_coop.shuffle(
                cutlass_coop.this_block(), payload, mode=mode
            )[0]
        else:
            result = cutlass_coop.shuffle(cutlass_coop.this_block(), value, mode=mode)
        cute.make_tensor(memory, cute.make_layout(1))[0] = result

    assert _compile(kernel, dtype) is not None


@pytest.mark.parametrize("mode", ("offset", "rotate"))
@pytest.mark.parametrize(
    "dtype",
    (
        cutlass.Int8,
        cutlass.Int16,
        cutlass.Int32,
        cutlass.Int64,
        cutlass.Uint8,
        cutlass.Uint16,
        cutlass.Uint32,
    ),
)
def test_dynamic_distance(mode, dtype):
    @cute.kernel
    def kernel(memory: cute.Pointer):
        distance = dtype(cute.arch.thread_idx()[0] % 7 + 1)
        result = cutlass_coop.shuffle(
            cutlass_coop.this_block(), cutlass.Int32(1), mode=mode, distance=distance
        )
        cute.make_tensor(memory, cute.make_layout(1))[0] = result

    assert _compile(kernel) is not None


@pytest.mark.parametrize(
    "case,pattern",
    (
        ("warp", "block group"),
        ("scalar_up", "scalar shuffle"),
        ("array_rotate", "ThreadData shuffle"),
        ("array_distance", "exactly 1"),
        ("array_runtime", "compile-time 1"),
        ("rotate_zero", "distance"),
        ("rotate_size", "distance"),
        ("offset_large", "distance"),
        ("runtime_u64", "unsigned integer up to 32"),
        ("float_distance", "distance"),
        ("bool_distance", "distance"),
        ("common_scalar", "backend-qualified"),
        ("common_register", "ThreadData"),
        ("edge_output", "block_prefix"),
        ("storage", "temp_storage"),
    ),
)
def test_invalid_profiles(case, pattern):
    @cute.kernel
    def kernel(memory: cute.Pointer):
        group = cutlass_coop.this_block()
        value = cutlass.Int32(1)
        payload = cutlass_coop.ThreadData(2, dtype=cutlass.Int32, values=[value, value])
        if cutlass.const_expr(case == "warp"):
            cutlass_coop.shuffle(cutlass_coop.this_warp(), payload)
        elif cutlass.const_expr(case == "scalar_up"):
            cutlass_coop.shuffle(group, value, mode="up")
        elif cutlass.const_expr(case == "array_rotate"):
            cutlass_coop.shuffle(group, payload, mode="rotate")
        elif cutlass.const_expr(case == "array_distance"):
            cutlass_coop.shuffle(group, payload, distance=2)
        elif cutlass.const_expr(case == "array_runtime"):
            cutlass_coop.shuffle(group, payload, distance=cutlass.Int32(1))
        elif cutlass.const_expr(case == "rotate_zero"):
            cutlass_coop.shuffle(group, value, mode="rotate", distance=0)
        elif cutlass.const_expr(case == "rotate_size"):
            cutlass_coop.shuffle(group, value, mode="rotate", distance=48)
        elif cutlass.const_expr(case == "offset_large"):
            cutlass_coop.shuffle(group, value, mode="offset", distance=1 << 32)
        elif cutlass.const_expr(case == "runtime_u64"):
            cutlass_coop.shuffle(
                group, value, mode="offset", distance=cutlass.Uint64(1)
            )
        elif cutlass.const_expr(case == "float_distance"):
            cutlass_coop.shuffle(
                group, value, mode="rotate", distance=cutlass.Float32(1)
            )
        elif cutlass.const_expr(case == "bool_distance"):
            cutlass_coop.shuffle(group, value, mode="rotate", distance=True)
        elif cutlass.const_expr(case == "common_scalar"):
            coop.shuffle(group, value, mode="offset")
        elif cutlass.const_expr(case == "common_register"):
            coop.shuffle(group, cute.make_rmem_tensor(2, cutlass.Int32))
        elif cutlass.const_expr(case == "edge_output"):
            cutlass_coop.shuffle(group, payload, block_prefix=payload)
        elif cutlass.const_expr(case == "storage"):
            cutlass_coop.shuffle(
                group, payload, temp_storage=cutlass_coop.TempStorage()
            )

    with pytest.raises(Exception, match=pattern):
        _compile(kernel)


def test_one_thread_rotate():
    @cute.kernel
    def kernel(memory: cute.Pointer):
        cutlass_coop.shuffle(cutlass_coop.this_block(), cutlass.Int32(1), mode="rotate")

    with pytest.raises(Exception, match="at least two"):
        _compile(kernel, block=1)
