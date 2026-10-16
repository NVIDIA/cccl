# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""How ``valid_items=``, ``oob_default=``, ``offset=`` and a scalar ``store``
value reach the plan and the kernel's MLIR.

The plan shows each control on its site as ``runtime`` or as its static
value. A runtime control is an operand of the provider's ``func.call``, at
its ABI type; a static one specializes the provider and is not an operand.
"""

from __future__ import annotations

from numba_cuda_mlir import cuda, types

from cuda import coop
from cuda.coop.numba_mlir._pipeline import testing

from .._support import BLOCK, CHIP, f32_in_out, requires_typed_pipeline

pytestmark = requires_typed_pipeline


def _runtime_signature(*controls):
    return types.void(*f32_in_out().args, *controls)


def test_controls_remain_runtime():
    @cuda.jit
    def partial(source, output, n, pad, off):
        block = coop.this_block()
        items = coop.ThreadData(4)
        coop.load(
            block,
            source,
            items,
            algorithm="transpose",
            valid_items=n,
            oob_default=pad,
            offset=off,
        )
        coop.store(block, output, items, valid_items=n, offset=off)

    testing.check(
        partial,
        _runtime_signature(types.int32, types.float32, types.int64),
        """
        CHECK-LABEL: coop-plan partial
        CHECK:         site #0 load @L{{[0-9]+}} algorithm=transpose
        CHECK-SAME:      dtype=float32 items_per_thread=4 block=128x1x1
        CHECK-SAME:      valid_items=runtime oob_default=runtime offset=runtime
        CHECK-SAME:      region=implicit.0{{$}}
        CHECK-NEXT:    barrier after #0
        CHECK-NEXT:    site #1 store @L{{[0-9]+}} algorithm=direct
        CHECK-SAME:      dtype=float32 items_per_thread=4 block=128x1x1
        CHECK-SAME:      valid_items=runtime offset=runtime{{$}}
        """,
        block=BLOCK,
        chip=CHIP,
    )


def test_static_controls_show_their_values():
    @cuda.jit
    def fixed(source, output):
        block = coop.this_block()
        items = coop.ThreadData(4)
        coop.load(block, source, items, valid_items=100, oob_default=-1.5)
        coop.store(block, output, items, valid_items=100, offset=256)

    testing.check(
        fixed,
        f32_in_out(),
        """
        CHECK-LABEL: coop-plan fixed
        CHECK:         site #0 load @L{{[0-9]+}} algorithm=direct
        CHECK-SAME:      items_per_thread=4 block=128x1x1
        CHECK-SAME:      valid_items=100 oob_default=-1.5{{$}}
        CHECK-NEXT:    site #1 store @L{{[0-9]+}} algorithm=direct
        CHECK-SAME:      items_per_thread=4 block=128x1x1
        CHECK-SAME:      valid_items=100 offset=256{{$}}
        """,
        block=BLOCK,
        chip=CHIP,
    )


def test_a_scalar_store_stores_one_item_per_thread():
    @cuda.jit
    def fill(output, value):
        coop.store(coop.this_block(), output, value)
        coop.store(coop.this_block(), output, 2, offset=128)

    testing.check(
        fill,
        types.void(types.float32[::1], types.float32),
        """
        CHECK-LABEL: coop-plan fill
        CHECK-NEXT:    block (128, 1, 1) from launch
        CHECK:         site #0 store @L{{[0-9]+}} algorithm=direct
        CHECK-SAME:      dtype=float32 items_per_thread=1 block=128x1x1
        CHECK-SAME:      value=runtime{{$}}
        CHECK-NEXT:    site #1 store @L{{[0-9]+}} algorithm=direct
        CHECK-SAME:      dtype=float32 items_per_thread=1 block=128x1x1
        CHECK-SAME:      value=2.0 offset=128{{$}}
        """,
        block=BLOCK,
        chip=CHIP,
    )


def test_runtime_controls_are_call_operands_at_their_abi_types():
    @cuda.jit
    def partial(source, output, n, pad, off):
        block = coop.this_block()
        items = coop.ThreadData(4)
        coop.load(
            block,
            source,
            items,
            algorithm="transpose",
            valid_items=n,
            oob_default=pad,
            offset=off,
        )
        coop.store(block, output, items, valid_items=n)

    testing.check(
        partial,
        _runtime_signature(types.int32, types.float32, types.int64),
        r"""
        CHECK:       gpu.func @{{.*}}partial{{.*}}(
        CHECK-SAME:    %[[N:arg[0-9]+]]: i32, %[[PAD:arg[0-9]+]]: f32,
        CHECK-SAME:    %[[OFF:arg[0-9]+]]: i64
        CHECK:       %[[N64:[0-9]+]] = arith.extsi %[[N]] : i32 to i64
        CHECK:       func.call @[[LOAD:\w+BLOCK_LOAD_TRANSPOSE\w+]](
        CHECK-SAME:    %[[N64]], %[[PAD]], %[[OFF]])
        CHECK-SAME:    : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, f32,
        CHECK-SAME:    i64) -> i32
        CHECK:       func.call @[[STORE:\w+BLOCK_STORE_DIRECT\w+]](
        CHECK-SAME:    , %{{[0-9]+}})
        CHECK-SAME:    : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> i32
        CHECK-DAG:   func.func private @[[LOAD]](
        CHECK-DAG:   func.func private @[[STORE]](
        """,
        dump="mlir",
        block=BLOCK,
        chip=CHIP,
    )
    assert "Boundedint64Toint32_0_512_Exactfloat32_Offsetint64Runtime" in (
        testing.compile_kernel(
            partial,
            _runtime_signature(types.int32, types.float32, types.int64),
            block=BLOCK,
            chip=CHIP,
        ).mlir
    )


def test_static_controls_specialize_the_provider_and_are_not_operands():
    @cuda.jit
    def fixed(source, output):
        block = coop.this_block()
        items = coop.ThreadData(4)
        coop.load(block, source, items, valid_items=100, oob_default=0.0)
        coop.store(block, output, items, offset=64)

    testing.check(
        fixed,
        f32_in_out(),
        r"""
        CHECK:       func.call @{{\w+}}BLOCK_LOAD_DIRECT{{\w+}}_F100_{{\w+}}(
        CHECK-SAME:    : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
        CHECK:       func.call @{{\w+}}BLOCK_STORE_DIRECT{{\w+}}Static64__abi(
        CHECK-SAME:    : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
        """,
        dump="mlir",
        block=BLOCK,
        chip=CHIP,
    )


def test_a_scalar_store_value_is_boxed_into_a_one_item_array():
    @cuda.jit
    def fill(output, value):
        coop.store(coop.this_block(), output, value)

    testing.check(
        fill,
        types.void(types.float32[::1], types.float32),
        r"""
        CHECK:       gpu.func @{{.*}}fill{{.*}}(
        CHECK-SAME:    %[[VALUE:arg[0-9]+]]: f32
        CHECK:       %[[BOX:[0-9a-z_]+]] = memref.alloca() : memref<1xf32>
        CHECK:       memref.store %[[VALUE]], %[[BOX]][%{{[0-9a-z_]+}}]
        CHECK:       func.call @{{\w+}}BLOCK_STORE_DIRECT{{\w+}}_A1_float32
        """,
        dump="mlir",
        block=BLOCK,
        chip=CHIP,
    )
