# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Fresh movement results compose through native arrays, aliases and tuples."""

from __future__ import annotations

import numba_cuda_mlir.tools as compiler_tools
import pytest
from numba_cuda_mlir import cuda, types

import cuda.coop.numba_mlir as numba_coop
from cuda import coop
from cuda.coop.numba_mlir._pipeline import testing

from .._support import BLOCK, CHIP, requires_typed_pipeline

pytestmark = requires_typed_pipeline


@pytest.fixture(autouse=True)
def _fixed_compiler_target(monkeypatch):
    monkeypatch.setattr(
        compiler_tools,
        "get_gpu_compute_capability",
        lambda as_type=str: (12, 0) if as_type is tuple else CHIP,
    )


@pytest.mark.parametrize("items_per_thread", [1, 4])
def test_fresh_results_compose_with_aliases_and_tuples(items_per_thread):
    def kernel(source, output, preserved):
        group = coop.this_block()
        value = coop.ThreadData(items_per_thread)
        coop.load(group, source, value)
        exchanged = coop.exchange(group, value)
        pair = (value, exchanged)
        alias = pair[1]
        alias[0] = types.float64(7)
        shifted = coop.shuffle(group, alias, mode="down")
        coop.store(group, output, shifted)
        coop.store(group, preserved, pair[0])

    testing.check(
        kernel,
        types.void(*([types.float64[::1]] * 3)),
        r"""
        CHECK: memref.alloca() : memref<ITEMSxf64>
        CHECK: func.call @{{\w+_block_load\w+__abi}}(
        CHECK: memref.alloca() : memref<ITEMSxf64>
        CHECK: func.call @{{\w+_block_exchange\w+__abi}}(
        CHECK: memref.alloca() : memref<ITEMSxf64>
        CHECK: func.call @{{\w+_block_shuffle\w+__abi}}(
        CHECK: func.call @{{\w+_block_store\w+__abi}}(
        CHECK: func.call @{{\w+_block_store\w+__abi}}(
        """.replace("ITEMS", str(items_per_thread)),
        dump="mlir",
        block=BLOCK,
        chip=CHIP,
    )


def test_qualified_local_array_result_extent_survives_chained_calls():
    def kernel(source, output):
        group = coop.this_block()
        value = cuda.local.array(4, types.int64)
        numba_coop.load(group, source, value)
        first = numba_coop.exchange(group, value)
        second = numba_coop.shuffle(group, first, mode="up")
        third = numba_coop.exchange(group, second, mode="blocked_to_striped")
        numba_coop.store(group, output, third)

    compiled = testing.compile_kernel(
        kernel,
        types.void(types.int64[::1], types.int64[::1]),
        block=BLOCK,
        chip=CHIP,
    )
    assert compiled.plan.count("items_per_thread=4") == 5


def test_scatter_keeps_independent_rank_and_flag_dtypes():
    def kernel(source, ranks, flags, output):
        group = coop.this_block()
        value = coop.ThreadData(items_per_thread=4)
        rank = coop.ThreadData(items_per_thread=4)
        valid = coop.ThreadData(items_per_thread=4)
        coop.load(group, source, value)
        coop.load(group, ranks, rank)
        coop.load(group, flags, valid)
        result = numba_coop.exchange(
            group,
            value,
            mode="scatter_to_striped_flagged",
            ranks=rank,
            valid_flags=valid,
        )
        coop.store(group, output, result)

    compiled = testing.compile_kernel(
        kernel,
        types.void(
            types.float64[::1],
            types.int64[::1],
            types.uint8[::1],
            types.float64[::1],
        ),
        block=BLOCK,
        chip=CHIP,
    )
    assert "ScatterToStripedFlagged" in compiled.mlir


@pytest.mark.parametrize("mode", ["offset", "rotate"])
def test_scalar_shuffle_retains_wide_runtime_distance(mode):
    def kernel(source, output, distance):
        result = numba_coop.shuffle(
            coop.this_block(),
            source[cuda.threadIdx.x],
            mode=mode,
            distance=distance,
        )
        output[cuda.threadIdx.x] = result

    testing.check(
        kernel,
        types.void(types.int64[::1], types.int64[::1], types.int64),
        r"""
        CHECK: func.call @{{\w+_block_shuffle\w+__abi}}(
        CHECK-SAME: (!llvm.ptr, !llvm.ptr, i64, i64) -> i32
        CHECK-NEXT: {{.*}} = llvm.load {{.*}} : !llvm.ptr -> i64
        """,
        dump="mlir",
        block=BLOCK,
        chip=CHIP,
    )
