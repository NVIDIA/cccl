# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Preserve cooperative participation through optimization of extern calls."""

from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm


def lower_convergent_provider_calls(module, requests):
    """Give provider declarations and calls LLVM convergence semantics.

    CuTe FFI emits ordinary ``func`` declarations and calls. Their bodies can
    contain barriers, shuffles, and reductions that must execute together.
    Without convergence semantics, optimization can duplicate a call into
    divergent branches and leave participating threads at different barriers.
    Provider signatures already use LLVM-compatible scalar and pointer types,
    so lower these externs directly to LLVM's first-class convergent operations.
    """

    symbols = {request.symbol_name for request in requests}
    operations = []

    def collect(operation):
        if operation.name in {"func.func", "func.call"}:
            operations.append(operation)
        return ir.WalkResult.ADVANCE

    module.operation.walk(collect)
    for operation in operations:
        attribute = "sym_name" if operation.name == "func.func" else "callee"
        symbol = operation.attributes[attribute].value
        if symbol not in symbols:
            continue
        with ir.InsertionPoint(operation), operation.location:
            if operation.name == "func.func":
                signature = ir.FunctionType(
                    ir.TypeAttr(operation.attributes["function_type"]).value
                )
                result_type = (
                    signature.results[0]
                    if signature.results
                    else ir.Type.parse("!llvm.void")
                )
                llvm.LLVMFuncOp(
                    symbol,
                    ir.TypeAttr.get(
                        llvm.FunctionType.get(result_type, signature.inputs)
                    ),
                    convergent=True,
                    no_inline="no_inline" in operation.attributes,
                )
            else:
                call = llvm.CallOp(
                    operation.results[0].type if operation.results else None,
                    list(operation.operands),
                    [],
                    [],
                    callee=symbol,
                    convergent=True,
                    no_inline="no_inline" in operation.attributes,
                )
                for old, new in zip(operation.results, call.results):
                    old.replace_all_uses_with(new)
        operation.erase()
