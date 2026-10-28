# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check producer snapshots, runtime loops, and register export."""

import cutlass
import numpy as np
import pytest
from cutlass import cute

import cuda.coop.cutlass as coop
from tests.runtime_support import device_array

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.runtime, pytest.mark.gpu]


class _RegisterProducer:
    def __init__(self, registers):
        self.registers = registers
        self.calls = 0

    @property
    def shape(self):
        return self.registers.shape

    @property
    def dtype(self):
        return self.registers.element_type

    def __cuda_coop_thread_data_load__(self):
        self.calls += 1
        return self.registers.load()


@pytest.mark.parametrize("items_per_thread", (1, 4))
@pytest.mark.parametrize("iterations", (0, 2))
def test_producer_snapshots_and_metadata_export(items_per_thread, iterations):
    @cute.kernel
    def kernel(destination: cute.Pointer, iterations: cutlass.Int32):
        thread, _, _ = cute.arch.thread_idx()
        registers = cute.make_rmem_tensor(items_per_thread, cutlass.Int32)
        for item in cutlass.range_constexpr(items_per_thread):
            registers[item] = thread * items_per_thread + item * 3
        source = _RegisterProducer(registers)
        values = coop.ThreadData.load(source)
        if cutlass.const_expr(source.calls != 1):
            raise AssertionError("producer hook called more than once")
        registers[0] = cutlass.Int32(-100)
        for iteration in range(iterations):
            values[0] += cutlass.Int32(1)
        snapshot = values.to_tensor_ssa(like=source)
        values[0] += cutlass.Int32(100)
        output = cute.make_tensor(
            destination, cute.make_layout(2 * 32 * items_per_thread)
        )
        for item in cutlass.range_constexpr(items_per_thread):
            index = thread * items_per_thread + item
            output[index] = snapshot[item]
            output[32 * items_per_thread + index] = values[item]

    @cute.jit
    def launch(destination: cute.Pointer, iterations: cutlass.Int32):
        kernel(destination, iterations).launch(grid=1, block=32)

    output = np.zeros(2 * 32 * items_per_thread, dtype=np.int32)
    with device_array(output) as destination:
        launch(destination, cutlass.Int32(iterations))
    expected = (
        np.arange(32, dtype=np.int32)[:, None] * items_per_thread
        + np.arange(items_per_thread, dtype=np.int32)[None, :] * 3
    )
    expected[:, 0] += iterations
    np.testing.assert_array_equal(
        output[: 32 * items_per_thread], expected.ravel()
    )
    expected[:, 0] += 100
    np.testing.assert_array_equal(
        output[32 * items_per_thread :], expected.ravel()
    )
