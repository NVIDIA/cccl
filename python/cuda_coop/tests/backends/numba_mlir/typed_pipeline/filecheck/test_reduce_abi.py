# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Scalar reductions preserve result width and link their callback images."""

from __future__ import annotations

import numba_cuda_mlir.tools as compiler_tools
import pytest
from numba_cuda_mlir import cuda, types

import cuda.coop.numba_mlir as numba_coop
from cuda import coop
from cuda.coop.numba_mlir._pipeline import _link, testing

from .._support import BLOCK, CHIP, requires_typed_pipeline

pytestmark = requires_typed_pipeline


@pytest.fixture(autouse=True)
def _fixed_compiler_target(monkeypatch):
    # Numba resolves the host capability even with an explicit chip. Keep
    # these compile-only checks independent of any visible CUDA device.
    monkeypatch.setattr(
        compiler_tools,
        "get_gpu_compute_capability",
        lambda as_type=str: (12, 0) if as_type is tuple else CHIP,
    )


@pytest.mark.parametrize(
    ("dtype", "mlir_type"),
    [(types.int32, "i32"), (types.int64, "i64"), (types.float64, "f64")],
)
def test_scalar_sum_result_uses_its_full_width(dtype, mlir_type):
    def kernel(source, output):
        total = coop.sum(coop.this_block(), source[cuda.threadIdx.x])
        if cuda.threadIdx.x == 0:
            output[0] = total

    testing.check(
        kernel,
        types.void(dtype[::1], dtype[::1]),
        r"""
        CHECK: %[[RET:\w+]] = llvm.alloca {{.*}} x TYPE :
        CHECK: func.call @{{\w+_block_reduce\w+__abi}}(%[[RET]],
        CHECK-NEXT: %[[VALUE:\w+]] = llvm.load %[[RET]] : !llvm.ptr -> TYPE
        CHECK: memref.store %[[VALUE]],
        """.replace("TYPE", mlir_type),
        dump="mlir",
        block=BLOCK,
        chip=CHIP,
    )


@pytest.mark.parametrize("items_per_thread", [1, 4])
def test_thread_data_sum_returns_a_scalar(items_per_thread):
    def kernel(source, output):
        block = coop.this_block()
        values = coop.ThreadData(items_per_thread)
        coop.load(block, source, values)
        total = coop.sum(block, values)
        if cuda.threadIdx.x == 0:
            output[0] = total

    testing.check(
        kernel,
        types.void(types.float32[::1], types.float32[::1]),
        r"""
        CHECK: func.call @{{\w+_block_load\w+__abi}}(
        CHECK: %[[RET:\w+]] = llvm.alloca {{.*}} x f32 :
        CHECK: func.call @{{\w+_block_reduce\w+__abi}}(%[[RET]],
        CHECK-NEXT: %[[VALUE:\w+]] = llvm.load %[[RET]] : !llvm.ptr -> f32
        CHECK: memref.store %[[VALUE]],
        """,
        dump="mlir",
        block=BLOCK,
        chip=CHIP,
    )


def test_callback_bundle_reuse_preserves_captured_values(monkeypatch):
    def make_kernel(offset):
        @cuda.jit(device=True)
        def add(left, right):
            return left + right + offset

        def kernel(source, output):
            result = numba_coop.reduce(
                coop.this_block(), source[cuda.threadIdx.x], binary_op=add
            )
            if cuda.threadIdx.x == 0:
                output[0] = result

        return kernel

    bundles = []
    compile_bundle = _link.compile_bundle

    def record_bundle(*args, **kwargs):
        bundle = compile_bundle(*args, **kwargs)
        bundles.append(bundle)
        return bundle

    monkeypatch.setattr(_link, "compile_bundle", record_bundle)
    _link.clear()
    first = make_kernel(0)
    for kernel in (first, first, make_kernel(1)):
        testing.compile_kernel(
            kernel,
            types.void(types.int32[::1], types.int32[::1]),
            block=BLOCK,
            chip=CHIP,
        )

    assert len(bundles) == 3
    assert bundles[0] is bundles[1]
    assert bundles[0].key != bundles[2].key
    assert len(bundles[0].support_link_items) == 1
    assert len(bundles[2].support_link_items) == 1
    assert (
        bundles[0].support_link_items[0].data
        != bundles[2].support_link_items[0].data
    )
