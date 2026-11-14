# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check numeric casts before register export and dtype selector adaptation."""

import cutlass
import numpy as np
import pytest
from cutlass import cute

import cuda.coop.cutlass as coop
from tests.runtime_support import device_array

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.runtime, pytest.mark.gpu]


@pytest.mark.parametrize(
    "selector", [cutlass.Int64, np.int64, np.dtype("int64"), int]
)
def test_casting_constructors_accept_runtime_values(selector):
    source_value = 7 if selector is int else 0xFFFFFFFF

    @cute.kernel
    def kernel(destination: cute.Pointer, value: cutlass.Uint32):
        payload = coop.ThreadData.from_fn(1, lambda _: value, dtype=selector)
        registers = cute.make_rmem_tensor(1, cutlass.Uint32)
        registers[0] = value
        from_registers = coop.ThreadData.from_register_tensor(
            registers, dtype=selector
        )
        from_vector = coop.ThreadData.from_vector(
            registers.load(), dtype=selector
        )
        from_payload = coop.ThreadData.from_payload(
            coop.ThreadData.from_values(value), dtype=selector
        )
        output = cute.make_tensor(destination, cute.make_layout(4))
        output[0] = payload.to_tensor_ssa(dtype=cutlass.Int64)[0]
        output[1] = from_registers.to_tensor_ssa(dtype=cutlass.Int64)[0]
        output[2] = from_vector.to_tensor_ssa(dtype=cutlass.Int64)[0]
        output[3] = from_payload.to_tensor_ssa(dtype=cutlass.Int64)[0]

    @cute.jit
    def launch(destination: cute.Pointer, value: cutlass.Uint32):
        kernel(destination, value).launch(grid=1, block=1)

    output = np.zeros(4, dtype=np.int64)
    with device_array(output) as destination:
        launch(destination, cutlass.Uint32(source_value))
    np.testing.assert_array_equal(output, [source_value] * 4)
