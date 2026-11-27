# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Require convergence semantics before collective calls reach optimization."""

import cutlass
import pytest
from cutlass import cute
from cutlass._mlir import ir
from cutlass.base_dsl.compiler import GPUArch
from cutlass.cute.runtime import make_ptr

import cuda.coop.cutlass as coop

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.compile]


def test_provider_calls_retain_convergence_semantics():
    @cute.kernel
    def kernel(memory: cute.Pointer):
        block = coop.this_block()
        rank = block.rank_as(cutlass.Int32)
        output = cute.make_tensor(memory, cute.make_layout(2))
        if rank == 0:
            output[1] = 7
        total = coop.sum(block, rank)
        block.sync()
        if rank == 0:
            output[0] = total

    @cute.jit
    def launch(memory: cute.Pointer):
        kernel(memory).launch(grid=1, block=64)

    calls = []
    declarations = []

    def inspect(dsl, module, name):
        def visit(operation):
            if operation.name in {"func.call", "llvm.call"}:
                symbol = operation.attributes["callee"].value
                if symbol.startswith("cuda_coop_cutlass_"):
                    assert operation.name == "llvm.call"
                    assert "convergent" in operation.attributes
                    calls.append(symbol)
            elif operation.name in {"func.func", "llvm.func"}:
                symbol = operation.attributes["sym_name"].value
                if symbol.startswith("cuda_coop_cutlass_"):
                    assert operation.name == "llvm.func"
                    assert "convergent" in operation.attributes
                    declarations.append(symbol)
            return ir.WalkResult.ADVANCE

        module.operation.walk(visit)

    pointer = make_ptr(
        cutlass.Int32, 0, cute.AddressSpace.gmem, assumed_align=16
    )
    cute.compile[(GPUArch("sm_80"),)](
        launch, pointer, trace_finalize_hooks=inspect
    )
    assert len(calls) == 3
    assert set(calls) == set(declarations)
