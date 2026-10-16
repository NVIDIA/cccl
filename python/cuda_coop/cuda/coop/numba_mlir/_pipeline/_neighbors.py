# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Type fresh neighbor results and bind the existing block providers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from . import _bridges as nb
from . import _callbacks, _templates
from ._errors import TypedPipelineError
from ._inference import operation_static_arguments
from ._results import ArrayAllocation
from ._types import PRODUCER, CoopGroupType, ThreadDataType

KINDS = frozenset({"adjacent_difference", "discontinuity"})
_BOUNDARIES = ("tile_predecessor_item", "tile_successor_item")


def _operation(kind, qualified):
    def operation(group, values, **kwargs):
        raise NotImplementedError("cuda.coop: kernel only")

    operation.__name__ = f"_coop_{'qualified_' if qualified else ''}{kind}"
    return operation


OPERATIONS = {
    _operation(kind, qualified): (kind, qualified)
    for kind in sorted(KINDS)
    for qualified in (False, True)
}


def _mode(kind, kws):
    adjacent = kind == "adjacent_difference"
    name, default = ("direction", "left") if adjacent else ("mode", "heads")
    value = kws.get(name)
    return default if _templates._omitted(value) else _callbacks.constant(value)


def _payload(kind, args, qualified):
    from .._compiler._parameters import _validate_common_numeric_dtype

    if len(args) != 2 or not isinstance(args[0], CoopGroupType):
        raise TypedPipelineError(f"coop.{kind}: expected group and values")
    value = args[1]
    if not qualified and not isinstance(value, ThreadDataType):
        raise TypedPipelineError(f"cuda.coop.{kind} requires ThreadData")
    if not isinstance(value, nb.types.Array):
        raise TypedPipelineError(f"coop.{kind}: values must be a fixed array")
    if value.ndim != 1 or value.layout != "C":
        raise TypedPipelineError(
            f"coop.{kind}: values must be contiguous and one-dimensional"
        )
    if not value.dtype.is_precise():
        return None
    return _validate_common_numeric_dtype(
        value.dtype, operation=kind, parameter="values"
    )


def _result_type(kind, value, mode):
    dtype = value.dtype if kind == "adjacent_difference" else nb.types.int32
    result = (
        ThreadDataType(dtype, value.items, PRODUCER)
        if isinstance(value, ThreadDataType)
        else nb.types.Array(dtype, 1, "C")
    )
    if mode == "heads_and_tails":
        return nb.types.BaseTuple.from_types((result, result))
    return result


@dataclass(frozen=True, eq=False)
class NeighborRequest:
    kind: str
    dtype: Any
    items_per_thread: int
    group: CoopGroupType
    call: Any
    operator: Any
    controls: tuple
    storage_options: tuple

    @property
    def key(self):
        return (
            self.call.semantic_key,
            tuple((name, value.semantic_key) for name, value in self.controls),
            self.storage_options,
        )

    @property
    def algorithm(self):
        return self.call.operation.primitive.mode

    @property
    def allocations(self):
        dtype = (
            self.dtype if self.kind == "adjacent_difference" else nb.types.int32
        )
        return tuple(
            ArrayAllocation(dtype, self.items_per_thread)
            for _ in self.call.operation.primitive.result_names
        )

    @property
    def result(self):
        if len(self.allocations) == 1:
            return ("allocation", 0)
        return ("tuple", (("allocation", 0), ("allocation", 1)))

    @property
    def provider_operands(self):
        from ..._core import BindingKind

        controls = dict(self.controls)
        operands: list[tuple] = [("pointer", 1)]
        operands.extend(("allocation", i) for i in range(len(self.allocations)))
        valid = controls["valid_items"]
        operands.append(
            ("value", "valid_items")
            if valid.kind is BindingKind.RUNTIME
            else (
                "constant",
                valid.value if valid.kind is BindingKind.STATIC else 0,
            )
        )
        for name in _BOUNDARIES:
            binding = controls[name]
            operands.append(
                ("value", name)
                if binding.kind is BindingKind.RUNTIME
                else (
                    "constant",
                    binding.value
                    if binding.kind is BindingKind.STATIC
                    else np.dtype(str(self.dtype)).type(0),
                )
            )
        return tuple(operands)

    def complete(self, block, plan):
        from .._lowering import _neighbors

        primitive = plan.call.operation.primitive
        suffix = "_both" if len(primitive.result_names) == 2 else ""
        return getattr(_neighbors, "block_" + self.kind + suffix), {
            "dtype": self.dtype,
            "threads_per_block": block,
            "items_per_thread": self.items_per_thread,
            "mode": primitive.mode,
            "partial": primitive.partial,
            "predecessor": primitive.predecessor,
            "successor": primitive.successor,
            "op": self.operator,
        }


def _controls(analysis, stmt, kind, dtype):
    from ..._core import ArgumentBinding
    from .._compiler._parameters import (
        _validate_runtime_integer_dtype,
        coerce_static_scalar,
    )
    from ._sites import _RUNTIME

    result, arguments = [], dict(stmt.value.kws)
    for name in ("valid_items", *_BOUNDARIES):
        variable = arguments.get(name)
        if variable is None or _templates._omitted(
            analysis.typemap[variable.name]
        ):
            result.append((name, ArgumentBinding.omitted()))
            continue
        ty = nb.unliteral(analysis.typemap[variable.name])
        scalar = analysis._provenance(variable)
        if scalar is _RUNTIME:
            if name == "valid_items":
                _validate_runtime_integer_dtype(
                    ty, operation=kind, parameter=name
                )
            elif ty != dtype:
                raise TypedPipelineError(
                    f"{kind} {name} dtype must match input dtype {dtype}"
                )
            binding = ArgumentBinding.runtime()
        else:
            value = scalar.value
            if name in _BOUNDARIES:
                value = coerce_static_scalar(
                    value,
                    dtype,
                    operation=kind,
                    parameter=name,
                    source_dtype=scalar.dtype,
                )
            binding = ArgumentBinding.static(value)
        result.append((name, binding))
    return tuple(result)


def site_request(analysis, stmt, kind, qualified):
    from ..._core import BindingKind, make_group_primitive_call
    from ..._core.block.neighbors import BlockNeighborSemantics
    from ..._core.group.neighbors import GroupNeighborSemantics
    from .._lowering._core import NumbaMlirCoreAdapter
    from .._lowering._neighbors import neighbor_operator

    call = stmt.value
    args = [analysis.typemap[arg.name] for arg in call.args]
    kws = {name: analysis.typemap[arg.name] for name, arg in call.kws}
    dtype = _payload(kind, args, qualified)
    if dtype is None:
        raise TypedPipelineError(f"coop.{kind}: payload dtype is unresolved")
    items = (
        args[1].items
        if isinstance(args[1], ThreadDataType)
        else analysis._array_extent(stmt, kind, index=1)
    )
    controls = _controls(analysis, stmt, kind, dtype)
    bindings = dict(controls)
    operator = _callbacks.constant(
        kws.get("difference_op" if kind == "adjacent_difference" else "flag_op")
    )
    primitive = BlockNeighborSemantics(
        operation=kind,
        dtype=NumbaMlirCoreAdapter().core_dtype(dtype),
        items_per_thread=items,
        mode=_mode(kind, kws),
        operator=neighbor_operator(kind, operator),
        partial=bindings["valid_items"].kind is not BindingKind.OMITTED,
        predecessor=bindings[_BOUNDARIES[0]].kind is not BindingKind.OMITTED,
        successor=bindings[_BOUNDARIES[1]].kind is not BindingKind.OMITTED,
    )
    return NeighborRequest(
        kind,
        dtype,
        items,
        args[0],
        make_group_primitive_call(
            args[0].group,
            GroupNeighborSemantics(primitive, bindings["valid_items"]),
        ),
        operator,
        controls,
        tuple(_templates.storage_options(kws.get("temp_storage")).items()),
    )


def _typing(kind, qualified):
    def generic(self, args, kws):
        from ..._core.block.neighbors import validate_neighbor_options

        dtype = _payload(kind, args, qualified)
        if dtype is None:
            return None
        mode = _mode(kind, kws)
        validate_neighbor_options(kind, mode)
        return nb.signature(
            _result_type(kind, args[1], mode), *args, *kws.values()
        )

    return generic


def _result_extent(analysis, call):
    extent = analysis._value_extent(call.args[1], "discontinuity")
    value_type = analysis.typemap[call.args[1].name]
    if isinstance(value_type, ThreadDataType):
        extent = value_type.items
    kws = {name: analysis.typemap[arg.name] for name, arg in call.kws}
    return (
        (extent, extent)
        if _mode("discontinuity", kws) == "heads_and_tails"
        else extent
    )


def register():
    _callbacks.register()
    for operation, (kind, qualified) in OPERATIONS.items():
        _templates.OPERATIONS[operation] = kind
        _templates.RESULT_EXTENT_SOURCES[operation] = (
            _result_extent if kind == "discontinuity" else 1
        )
        if qualified:
            _templates.QUALIFIED_OPERATIONS.add(operation)
        operation_static_arguments[operation] = {
            "difference_op": _callbacks.StaticCallableType,
            "flag_op": _callbacks.StaticCallableType,
        }
        _templates.register_template(
            operation,
            _typing(kind, qualified),
            operation.__name__,
            prefer_literal=True,
        )
    for public, name in _templates.kernel_primitives().items():
        kind = _templates.group_operation(public)
        if kind not in KINDS:
            continue
        qualified = name.startswith("cuda.coop.numba_mlir.")
        operation = next(
            op for op, info in OPERATIONS.items() if info == (kind, qualified)
        )
        nb.attach_inline_wrapper(
            public, _templates.result_wrapper(public, name, operation)
        )
        _templates.register_template(
            public, _templates._not_inlined(name), name
        )
