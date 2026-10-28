# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compile producer-owned register loads across runtime control flow."""

import cutlass
import pytest
from cutlass import cute
from cutlass.base_dsl.compiler import GPUArch
from cutlass.cute.runtime import make_ptr

import cuda.coop.cutlass as coop

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.compile]


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
def test_producer_load_and_like_survive_runtime_loop(items_per_thread):
    @cute.kernel
    def kernel(destination: cute.Pointer, iterations: cutlass.Int32):
        registers = cute.make_rmem_tensor(items_per_thread, cutlass.Int32)
        for item in cutlass.range_constexpr(items_per_thread):
            registers[item] = cutlass.Int32(item)
        source = _RegisterProducer(registers)
        values = coop.ThreadData.load(source)
        if cutlass.const_expr(source.calls != 1):
            raise AssertionError("producer hook called more than once")
        for iteration in range(iterations):
            values[0] += cutlass.Int32(1)
        snapshot = values.to_tensor_ssa(like=source)
        registers.store(snapshot)
        coop.store(coop.this_block(), destination, values)

    @cute.jit
    def launch(destination: cute.Pointer, iterations: cutlass.Int32):
        kernel(destination, iterations).launch(grid=1, block=32)

    pointer = make_ptr(
        cutlass.Int32, 0, cute.AddressSpace.gmem, assumed_align=16
    )
    assert (
        cute.compile[(GPUArch("sm_80"),)](launch, pointer, cutlass.Int32(2))
        is not None
    )
