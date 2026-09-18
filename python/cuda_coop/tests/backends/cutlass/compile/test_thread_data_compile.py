# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

"""ThreadData carries typed lanes through CuTe calls, branches, and loops."""

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
@pytest.mark.parametrize("api", (coop, cutlass_coop), ids=("common", "qualified"))
def test_dynamic_payload(dtype, api):
    @cute.jit
    def increment(payload):
        payload[0] = dtype(payload[0] + dtype(1))
        return payload

    @cute.kernel
    def kernel(memory: cute.Pointer, iterations: cutlass.Int32):
        payload = api.ThreadData(2, dtype=dtype, alignment=64)
        payload[0] = 1
        payload[1] = 3
        for iteration in range(iterations):
            payload = increment(payload)
            if iteration % 2 == 0:
                payload[1] = dtype(payload[1] + dtype(2))
        if cutlass.const_expr(payload.alignment != 64 or payload.dtype is not dtype):
            raise AssertionError("ThreadData metadata changed across control flow")
        output = cute.make_tensor(memory, cute.make_layout(2))
        output[0] = payload[0]
        output[1] = payload[1]

    @cute.jit
    def launch(memory: cute.Pointer, iterations: cutlass.Int32):
        kernel(memory, iterations).launch(grid=1, block=1)

    pointer = make_ptr(dtype, 0, cute.AddressSpace.gmem, assumed_align=16)
    assert cute.compile[(GPUArch("sm_80"),)](launch, pointer, 3) is not None
