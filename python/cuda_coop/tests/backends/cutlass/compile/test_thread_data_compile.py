# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compile payload reconstruction across dynamic loops and nested CuTe calls.

A typed null pointer and an explicit SM80 target let the kernels compile
without a launch. For each supported element dtype, tracing must rebuild
ThreadData from the compiler's MLIR values after a runtime loop and a nested
JIT call, while preserving its declared alignment and item type. The tests
also convert one TensorSSA vector both inside and after a loop.
"""

import pytest

cutlass = pytest.importorskip("cutlass")

from cutlass import cute
from cutlass.base_dsl.compiler import GPUArch
from cutlass.cute.runtime import make_ptr

from cuda import coop
from cuda.coop import cutlass as cutlass_coop
from cuda.coop.cutlass._compiler._types import ALL_PROVIDER_TYPES

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.compile]


@pytest.mark.parametrize("dtype", tuple(ALL_PROVIDER_TYPES))
@pytest.mark.parametrize(
    "api", (coop, cutlass_coop), ids=("common", "qualified")
)
def test_dynamic_payload(dtype, api):
    """Carry a payload through a runtime loop, branch, and nested JIT helper.

    Rebind the helper result so tracing must reconstruct the payload from
    MLIR values. Check dtype and alignment during tracing after the loop.
    Successful compilation confirms the representation. The kernel is not
    launched, so no output values are checked.
    """

    @cute.jit
    def increment(payload):
        payload[0] = dtype(payload[0] + dtype(1))
        return payload

    @cute.kernel
    def kernel(
        memory: cute.Pointer,
        iterations: cutlass.Int32,
        items_per_thread: cutlass.Constexpr,
    ):
        payload = api.ThreadData(items_per_thread, dtype=dtype, alignment=64)
        payload[0] = 1
        payload[1] = 3
        for iteration in range(iterations):
            payload = increment(payload)
            if iteration % 2 == 0:
                payload[1] = dtype(payload[1] + dtype(2))
        if cutlass.const_expr(
            payload.alignment != 64 or payload.dtype is not dtype
        ):
            raise AssertionError(
                "ThreadData metadata changed across control flow"
            )
        output = cute.make_tensor(memory, cute.make_layout(2))
        output[0] = payload[0]
        output[1] = payload[1]

    @cute.jit
    def launch(
        memory: cute.Pointer,
        iterations: cutlass.Int32,
        items_per_thread: cutlass.Constexpr,
    ):
        kernel(memory, iterations, items_per_thread).launch(grid=1, block=1)

    pointer = make_ptr(dtype, 0, cute.AddressSpace.gmem, assumed_align=16)
    assert cute.compile[(GPUArch("sm_80"),)](launch, pointer, 3, 2) is not None


def test_tensor_ssa_conversion_across_regions():
    """Convert one SSA vector inside and after a runtime loop.

    The outer vector remains in use after both conversions. This exposes a
    conversion that incorrectly reuses values defined only inside the loop.
    An SM80 target and null pointer check compilation without execution.
    """

    @cute.kernel
    def kernel(memory: cute.Pointer, iterations: cutlass.Int32):
        payload = cutlass_coop.ThreadData.from_values(
            cutlass.Int32(1), cutlass.Int32(3), dtype=cutlass.Int32
        )
        vector = payload.to_tensor_ssa()
        output = cute.make_tensor(memory, cute.make_layout(6))
        for _ in range(iterations):
            inside = cutlass_coop.ThreadData.from_vector(vector)
            output[0], output[1] = inside[0], inside[1]
        outside = cutlass_coop.ThreadData.from_vector(vector)
        output[2], output[3] = outside[0], outside[1]
        output[4], output[5] = vector[0], vector[1]

    @cute.jit
    def launch(memory: cute.Pointer, iterations: cutlass.Int32):
        kernel(memory, iterations).launch(grid=1, block=1)

    pointer = make_ptr(
        cutlass.Int32, 0, cute.AddressSpace.gmem, assumed_align=16
    )
    assert (
        cute.compile[(GPUArch("sm_80"),)](launch, pointer, cutlass.Int32(3))
        is not None
    )
