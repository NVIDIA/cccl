# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Lower typed cooperative calls to MLIR.

Lowering takes no decisions. Everything it needs per call site (the
provider symbol and ABI, the scratch operand, the barrier, the link items)
was decided by call resolution and stored in
``state.metadata["coop_plan"]``:

* ``ThreadData(...)`` is a ``memref.alloca`` of its type's extent, aligned as
  the constructor's own ``alignment=`` asks, and its ``items_per_thread``
  is that extent as a constant;
* a ``load`` or ``store`` is a ``func.call`` to the provider, with the
  scratch pointer first when the provider takes one, then the memory and
  payload pointers, then the runtime controls converted to their ABI
  types, followed by a ``gpu.barrier`` when the plan has one. A scalar
  ``store`` value is boxed into a one-item array first;
* scalar-returning providers use the same hidden return pointer as Numba's
  external ABI, with storage matching the provider result dtype.
"""

from __future__ import annotations

import struct
from typing import Any

from . import _bridges as nb
from . import _results
from ._templates import (
    _CONSTRUCTOR_PARAMETERS,
    OPERATIONS,
    constructor_alignment,
    kernel_primitives,
)
from ._types import ThreadDataType

SCRATCH_KEYWORD = "__coop_scratch"


def _lower_thread_data(builder, target, args, kwargs):
    payload = builder.get_numba_type(target.name)
    if not isinstance(payload, ThreadDataType) or not payload.is_precise():
        raise AssertionError(f"coop.ThreadData lowered as {payload}")
    # Constant controls are normalized only for this call. Alignment is
    # allocation-specific and deliberately absent from the payload type key.
    expression = builder.func_ir.get_definition(target)
    signature = builder.fndesc.calltypes[expression]
    bound = dict(zip(_CONSTRUCTOR_PARAMETERS, signature.args[: len(args)]))
    bound.update(
        zip(
            (name for name, _ in (kwargs or [])),
            signature.args[len(args) :],
            strict=True,
        )
    )
    alignment = constructor_alignment("coop.ThreadData", bound.get("alignment"))
    element = builder.get_storage_type(payload.dtype)
    memref_type = nb.MemRefType.get([payload.items], element)
    with builder.alloca_insertion_point():
        if alignment is None:
            value = nb.memref.alloca(memref_type, [], [])
        else:
            value = nb.memref.alloca(
                memref_type,
                [],
                [],
                alignment=max(struct.calcsize("P"), alignment),
            )
    builder.store_var(target, value)


def _lower_items_per_thread(context, builder, target, value):
    extent = builder.get_numba_type(target.name)
    builder.store_var(
        target,
        nb.arith.constant(builder.get_mlir_type(extent), extent.literal_value),
    )


def _call_entry(builder, target) -> Any:
    plan = builder.metadata.get("coop_plan")
    entry = None if plan is None else plan.calls.get(target.name)
    if entry is None:
        raise AssertionError(
            "a cooperative call reached lowering without a storage plan"
        )
    return entry


def _box(builder, var, dtype, value):
    element = builder.get_storage_type(dtype)
    with builder.alloca_insertion_point():
        array = nb.memref.alloca(nb.MemRefType.get([1], element), [], [])
    if value is None:
        item = nb.cast_value(
            builder,
            builder.load_var(var),
            builder.get_numba_type(var.name),
            dtype,
        )
    else:
        item = nb.arith.constant(element, value.item())
    zero = nb.arith.constant(nb.IndexType.get(), 0)
    nb.memref.store(item, array, [zero])
    return array


def _operand(builder, operand, abi_type, args, keywords, allocations):
    tag = operand[0]
    if tag == "scratch":
        return nb.memref_data_pointer(
            builder.load_var(keywords[SCRATCH_KEYWORD])
        )
    if tag == "allocation":
        return nb.memref_data_pointer(allocations[operand[1]])
    var = _results.argument(operand[1], args, keywords)
    if tag == "pointer":
        return nb.memref_data_pointer(builder.load_var(var))
    if tag == "box":
        _, _, dtype, value = operand
        return nb.memref_data_pointer(_box(builder, var, dtype, value))
    if tag != "value":
        raise AssertionError(f"unknown cooperative operand {tag!r}")
    return nb.cast_value(
        builder,
        builder.load_var(var),
        builder.get_numba_type(var.name),
        abi_type,
    )


def _lower_operation(builder, target, args, kwargs):
    entry = _call_entry(builder, target)
    keywords = dict(kwargs or [])
    allocations = _results.allocate(builder, entry.allocations, args, keywords)
    operands = [
        _operand(builder, operand, abi_type, args, keywords, allocations)
        for operand, abi_type in zip(
            entry.operands, entry.abi_arguments, strict=True
        )
    ]
    function_type = nb.external_abi_function_type(
        builder, nb.signature(entry.return_type, *entry.abi_arguments)
    )
    returns_value = entry.return_type != nb.types.void
    result_storage_type = (
        builder.get_storage_type(entry.return_type)
        if returns_value
        else nb.mlir_types.i8()
    )
    with builder.alloca_insertion_point():
        result = nb.llvm.alloca(
            res=nb.PointerType.get(),
            array_size=nb.arith.constant(nb.mlir_types.i32(), 1),
            elem_type=result_storage_type,
        )
    callee = nb.get_or_insert_function(
        entry.symbol, function_type, builder.mlir_gpu_module
    )
    nb.func.call(
        result=list(function_type.results),
        callee=callee.name.value,
        operands_=[result, *operands],
    )
    returned = None
    if returns_value:
        stored = nb.llvm.load(res=result_storage_type, addr=result)
        returned = builder.from_storage(entry.return_type, stored)
    if entry.result is not None:
        builder.store_var(
            target,
            _results.assemble(builder, entry.result, allocations, returned),
        )
    elif returns_value:
        builder.store_var(target, returned)
    if entry.barrier_after:
        nb.gpu.barrier()
    for item in entry.link_items:
        builder.link_external_item(item)


_registered = False


def register() -> None:
    """Register lowering for every typed cooperative call."""

    global _registered
    if _registered:
        return
    any_args = nb.types.VarArg(nb.types.Any)
    for obj, qualified_name in kernel_primitives().items():
        short = qualified_name.rsplit(".", 1)[1]
        if short == "ThreadData":
            nb.lowering_registry.lower(obj, any_args)(_lower_thread_data)
    nb.lowering_registry.lower_getattr(ThreadDataType, "items_per_thread")(
        _lower_items_per_thread
    )
    for op in OPERATIONS:
        nb.lowering_registry.lower(op, any_args)(_lower_operation)
    _registered = True


__all__ = ["SCRATCH_KEYWORD", "register"]
