# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Call resolution reports every kernel, and its dumps are FileCheck-able.

These cases pin the dump formats on kernels without cooperative calls; the
per-operation tests check the same dumps on real cooperative kernels.
"""

from __future__ import annotations

import pytest

from ._support import requires_typed_pipeline

pytestmark = requires_typed_pipeline


def _copy_kernel():
    from numba_cuda_mlir import cuda

    @cuda.jit
    def copy(dst, src):
        i = cuda.threadIdx.x
        dst[i] = src[i]

    return copy


def _signature():
    from numba_cuda_mlir import types

    return types.void(types.float32[::1], types.float32[::1])


def test_plan_without_coop_calls():
    from cuda.coop.numba_mlir._pipeline import testing

    testing.check(
        _copy_kernel(),
        _signature(),
        """
        CHECK-LABEL: coop-plan copy
        CHECK-NEXT:    block not requested
        CHECK-NEXT:    no coop sites
        CHECK-NOT:     launch_bounds
        """,
        chip="sm_120",
    )


def test_typed_ir_shows_every_assigned_type():
    from cuda.coop.numba_mlir._pipeline import testing

    testing.check(
        _copy_kernel(),
        _signature(),
        """
        CHECK-LABEL: typed-ir copy
        CHECK:       dst = arg(0, name=dst)  :: array(float32, 1d, C)
        CHECK:       src = arg(1, name=src)  :: array(float32, 1d, C)
        CHECK:       = freevar(cuda: <module 'numba_cuda_mlir.cuda'>)
        CHECK:       i = getattr(value={{.*}}, attr=x)  :: int32
        CHECK-NOT:   0x
        """,
        dump="typed_ir",
        chip="sm_120",
    )


def test_mlir_dump_is_the_lowered_module():
    from cuda.coop.numba_mlir._pipeline import testing

    testing.check(
        _copy_kernel(),
        _signature(),
        """
        CHECK: gpu.func @{{.*}}copy
        CHECK: memref.load
        CHECK: memref.store
        """,
        dump="mlir",
        chip="sm_120",
    )


def test_prefixes_select_one_configuration():
    from cuda.coop.numba_mlir._pipeline import testing

    text = "barrier after #0\nsite #1 store\n"
    checks = """
    FIRST:      barrier after #0
    SECOND-NOT: barrier
    SECOND:     site #1
    """
    testing.filecheck(checks, text, prefixes=("FIRST",))
    with pytest.raises(AssertionError, match="FileCheck failed"):
        testing.filecheck(checks, text, prefixes=("SECOND",))
