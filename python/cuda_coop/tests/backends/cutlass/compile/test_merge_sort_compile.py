# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

"""Compile real Merge Sort providers, partial shims, and inferred results."""

import pytest

cutlass = pytest.importorskip("cutlass")

from cutlass import cute
from cutlass.base_dsl.common import DSLRuntimeError
from cutlass.base_dsl.compiler import GPUArch
from cutlass.cute.runtime import make_ptr

from cuda import coop
from cuda.coop import cutlass as cutlass_coop
from cuda.coop.cutlass._compiler._types import ALL_PROVIDER_TYPES

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.compile]


def _pointer(dtype=cutlass.Int32):
    return make_ptr(dtype, 0, cute.AddressSpace.gmem, assumed_align=16)


@pytest.mark.parametrize("api", (coop, cutlass_coop), ids=("common", "qualified"))
@pytest.mark.parametrize("width", (1, 2, 4, 8, 16, 32, 64))
@pytest.mark.parametrize("partial", (False, True))
def test_groups_and_partial_pairs(api, width, partial):
    @cute.kernel
    def kernel(memory: cute.Pointer, count: cutlass.Int64):
        if cutlass.const_expr(width == 64):
            group = api.this_block()
        else:
            group = api.this_warp().group_by(width)
        keys = api.ThreadData(3, dtype=cutlass.Int32)
        values = api.ThreadData(3, dtype=cutlass.Float64)
        for item in cutlass.range_constexpr(3):
            keys[item] = cutlass.Int32(item + 1)
            values[item] = cutlass.Float64(item + 1)
        if cutlass.const_expr(partial):
            result, payload = api.merge_sort_pairs(
                group, keys, values, valid_items=count, oob_default=99
            )
        else:
            result, payload = api.merge_sort_pairs(group, keys, values)
        cute.make_tensor(memory, cute.make_layout(1))[0] = result[0] + cutlass.Int32(
            payload[0]
        )

    @cute.jit
    def launch(memory: cute.Pointer, count: cutlass.Int64):
        kernel(memory, count).launch(grid=1, block=(8, 4, 2))

    assert (
        cute.compile[(GPUArch("sm_80"),)](launch, _pointer(), cutlass.Int64(1))
        is not None
    )


@pytest.mark.parametrize("dtype", tuple(ALL_PROVIDER_TYPES))
def test_inferred_key_type(dtype):
    @cute.kernel
    def kernel(memory: cute.Pointer):
        keys = coop.ThreadData(2)
        keys[0] = dtype(2)
        keys[1] = dtype(1)
        result = coop.merge_sort_keys(coop.this_block(), keys)
        total = coop.sum(coop.this_block(), result)
        cute.make_tensor(memory, cute.make_layout(1))[0] = total

    @cute.jit
    def launch(memory: cute.Pointer):
        kernel(memory).launch(grid=1, block=32)

    assert cute.compile[(GPUArch("sm_80"),)](launch, _pointer(dtype)) is not None


def test_mixed_partial_group_bundle():
    @cute.kernel
    def kernel(memory: cute.Pointer):
        keys = coop.ThreadData(2, dtype=cutlass.Int32)
        keys[0] = cutlass.Int32(2)
        keys[1] = cutlass.Int32(1)
        first = coop.merge_sort_keys(
            coop.this_block(), keys, valid_items=61, oob_default=99
        )
        result = coop.merge_sort_keys(
            coop.this_warp().group_by(8), first, valid_items=15, oob_default=99
        )
        cute.make_tensor(memory, cute.make_layout(1))[0] = result[0]

    @cute.jit
    def launch(memory: cute.Pointer):
        kernel(memory).launch(grid=1, block=32)

    assert cute.compile[(GPUArch("sm_80"),)](launch, _pointer()) is not None


@pytest.mark.parametrize(
    "case, message",
    (
        ("count_u64", "valid_items.*signed integer"),
        ("count_float", "valid_items.*signed integer"),
        ("sentinel", "oob_default.*dtype"),
        ("unpaired", "provided together"),
        ("warp_storage", "only to block"),
        ("extent", "matching.*extents"),
        ("block_shape", "power-of-two"),
        ("incomplete_warp", "complete"),
    ),
)
def test_invalid_profiles(case, message):
    block = 48 if case == "block_shape" else 24 if case == "incomplete_warp" else 32
    count_type = cutlass.Uint64 if case == "count_u64" else cutlass.Float32

    @cute.kernel
    def kernel(memory: cute.Pointer, count: count_type):
        keys = coop.ThreadData(2, dtype=cutlass.Int32)
        keys[0], keys[1] = cutlass.Int32(2), cutlass.Int32(1)
        if cutlass.const_expr(case in {"count_u64", "count_float"}):
            result = coop.merge_sort_keys(
                coop.this_block(), keys, valid_items=count, oob_default=99
            )
        elif cutlass.const_expr(case == "sentinel"):
            result = coop.merge_sort_keys(
                coop.this_block(), keys, valid_items=1, oob_default=cutlass.Int64(99)
            )
        elif cutlass.const_expr(case == "unpaired"):
            result = coop.merge_sort_keys(coop.this_block(), keys, valid_items=1)
        elif cutlass.const_expr(case == "warp_storage"):
            result = coop.merge_sort_keys(
                coop.this_warp(), keys, temp_storage=coop.TempStorage()
            )
        elif cutlass.const_expr(case == "extent"):
            values = coop.ThreadData(1, dtype=cutlass.Int32)
            values[0] = cutlass.Int32(1)
            result, result_values = coop.merge_sort_pairs(
                coop.this_block(), keys, values
            )
        elif cutlass.const_expr(case == "incomplete_warp"):
            result = coop.merge_sort_keys(coop.this_warp().group_by(8), keys)
        else:
            result = coop.merge_sort_keys(coop.this_block(), keys)
        cute.make_tensor(memory, cute.make_layout(1))[0] = result[0]

    @cute.jit
    def launch(memory: cute.Pointer, count: count_type):
        kernel(memory, count).launch(grid=1, block=block)

    with pytest.raises(
        (TypeError, ValueError, DSLRuntimeError, NotImplementedError), match=message
    ):
        cute.compile[(GPUArch("sm_80"),)](launch, _pointer(), count_type(1))


@pytest.mark.parametrize("api", (coop, cutlass_coop), ids=("common", "qualified"))
def test_register_input_boundary(api):
    @cute.kernel
    def kernel(memory: cute.Pointer):
        fragment = cute.make_rmem_tensor(2, cutlass.Int32)
        fragment[0], fragment[1] = cutlass.Int32(2), cutlass.Int32(1)
        result = api.merge_sort_keys(api.this_block(), fragment)
        cute.make_tensor(memory, cute.make_layout(1))[0] = result[0]

    @cute.jit
    def launch(memory: cute.Pointer):
        kernel(memory).launch(grid=1, block=32)

    if api is coop:
        with pytest.raises(TypeError, match="ThreadData"):
            cute.compile[(GPUArch("sm_80"),)](launch, _pointer())
    else:
        assert cute.compile[(GPUArch("sm_80"),)](launch, _pointer()) is not None
