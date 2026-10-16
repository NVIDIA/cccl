# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""What the coop lowering emits into the kernel's MLIR.

Each site lowers to a ``func.call`` of its provider's external-ABI symbol;
scratch is a workgroup ``memref.global``; the plan's barriers follow their
call directly; and the launch's thread count becomes ``nvvm.maxntid``.
"""

from __future__ import annotations

from numba_cuda_mlir import cuda

from cuda import coop
from cuda.coop.numba_mlir._pipeline import testing

from .._support import BLOCK, CHIP, f32_in_out, requires_typed_pipeline

pytestmark = requires_typed_pipeline


def test_each_site_calls_its_provider():
    @cuda.jit
    def k1(source, output):
        block = coop.this_block()
        items = coop.ThreadData(4)
        coop.load(block, source, items)
        coop.store(block, output, items)

    testing.check(
        k1,
        f32_in_out(),
        r"""
        CHECK:       gpu.func @{{.*}}k1{{.*}} kernel attributes
        CHECK-NOT:   gpu.address_space<workgroup>
        CHECK:       func.call @[[LOAD:\w+_block_load\w+__abi]](
        CHECK-NOT:   gpu.barrier
        CHECK:       func.call @[[STORE:\w+_block_store\w+__abi]](
        CHECK-NOT:   gpu.barrier
        CHECK:       gpu.return
        CHECK-DAG:   func.func private @[[LOAD]](
        CHECK-DAG:   func.func private @[[STORE]](
        """,
        dump="mlir",
        block=BLOCK,
        chip=CHIP,
    )


def test_scratch_is_a_workgroup_global():
    @cuda.jit
    def kt(source, output):
        block = coop.this_block()
        items = coop.ThreadData(4)
        coop.load(block, source, items, algorithm="transpose")
        coop.store(block, output, items, algorithm="transpose")

    testing.check(
        kt,
        f32_in_out(),
        r"""
        CHECK:       memref.get_global @[[SHARED:\w+]]
        CHECK-SAME:    : memref<[[SIZE:[0-9]+]]xi8,
        CHECK-SAME:    #gpu.address_space<workgroup>>
        CHECK:       func.call @{{\w+}}BLOCK_LOAD_TRANSPOSE{{\w+}}__abi(
        CHECK:       memref.global "private" @[[SHARED]] : memref<[[SIZE]]xi8,
        CHECK-SAME:    #gpu.address_space<workgroup>> {alignment = 16 : i64}
        """,
        dump="mlir",
        block=BLOCK,
        chip=CHIP,
    )


def test_barriers_follow_their_transposing_call():
    @cuda.jit
    def kt(source, output):
        block = coop.this_block()
        items = coop.ThreadData(4)
        coop.load(block, source, items, algorithm="transpose")
        coop.store(block, output, items, algorithm="transpose")

    testing.check(
        kt,
        f32_in_out(),
        r"""
        CHECK:       func.call @{{\w+}}BLOCK_LOAD_TRANSPOSE{{\w+}}__abi(
        CHECK-NEXT:  gpu.barrier
        CHECK-NOT:   gpu.barrier
        CHECK:       func.call @{{\w+}}BLOCK_STORE_TRANSPOSE{{\w+}}__abi(
        CHECK-NEXT:  gpu.barrier
        CHECK-NEXT:  gpu.return
        """,
        dump="mlir",
        block=BLOCK,
        chip=CHIP,
    )
