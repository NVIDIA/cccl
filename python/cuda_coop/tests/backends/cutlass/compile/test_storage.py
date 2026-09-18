# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

"""C++ scratch layout and block participation errors are compile-time failures."""

import pytest

cutlass = pytest.importorskip("cutlass")

from cutlass import cute
from cutlass.base_dsl.compiler import GPUArch
from cutlass.cute.runtime import make_ptr

from cuda.coop import cutlass as cutlass_coop

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.compile]


@pytest.mark.parametrize("sharing", ("shared", "exclusive"))
@pytest.mark.parametrize(
    "algorithm", ("transpose", "warp_transpose", "warp_transpose_timesliced")
)
def test_undersized_storage_is_rejected_before_kernel_execution(sharing, algorithm):
    @cute.kernel
    def kernel(memory: cute.Pointer):
        storage = cutlass_coop.TempStorage(1, sharing=sharing)
        payload = cutlass_coop.ThreadData(4)
        cutlass_coop.load(
            cutlass_coop.this_block(),
            memory,
            payload,
            algorithm=algorithm,
            temp_storage=storage,
        )

    @cute.jit
    def launch(memory: cute.Pointer):
        kernel(memory).launch(grid=1, block=64)

    pointer = make_ptr(cutlass.Int32, 0, cute.AddressSpace.gmem, assumed_align=16)
    with pytest.raises(Exception, match="(?i)(capacity|size|smaller)"):
        cute.compile[(GPUArch("sm_80"),)](launch, pointer)


@pytest.mark.parametrize("algorithm", ("warp_transpose", "warp_transpose_timesliced"))
def test_warp_transpose_rejects_partial_physical_warps(algorithm):
    @cute.kernel
    def kernel(memory: cute.Pointer):
        cutlass_coop.load(
            cutlass_coop.this_block(),
            memory,
            cutlass_coop.ThreadData(4),
            algorithm=algorithm,
        )

    @cute.jit
    def launch(memory: cute.Pointer):
        kernel(memory).launch(grid=1, block=33)

    pointer = make_ptr(cutlass.Int32, 0, cute.AddressSpace.gmem, assumed_align=16)
    with pytest.raises(Exception, match="multiple of 32"):
        cute.compile[(GPUArch("sm_80"),)](launch, pointer)
