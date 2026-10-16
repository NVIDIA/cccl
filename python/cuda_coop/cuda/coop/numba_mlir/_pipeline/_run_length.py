# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Type decoded windows and bind the existing checked block decode drivers."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any

from ..._core import ArgumentBinding, BindingKind, make_group_primitive_call
from ..._core.block._common import normalize_positive_int
from ..._core.group.run_length import GroupRunLengthDecodeSemantics
from .._compiler._parameters import (
    _validate_common_numeric_dtype,
    normalize_dtype_param,
)
from .._lowering._core import NumbaMlirCoreAdapter
from . import _bridges as nb
from . import _templates
from ._errors import TypedPipelineError
from ._inference import (
    DtypeSource,
    OperationInference,
    operation_inference_contracts,
    operation_literal_arguments,
    static_call_arguments,
)
from ._results import ArrayAllocation
from ._types import PRODUCER, CoopGroupType, ThreadDataType

KINDS = frozenset({"run_length_decode", "run_length_decode_into"})


def _decode(group, run_values, run_lengths, **kwargs):
    raise NotImplementedError("cuda.coop: kernel only")


def _qualified_decode(group, run_values, run_lengths, **kwargs):
    raise NotImplementedError("cuda.coop: kernel only")


def _decode_into(group, run_values, run_lengths, destination, **kwargs):
    raise NotImplementedError("cuda.coop: kernel only")


def _qualified_decode_into(
    group, run_values, run_lengths, destination, **kwargs
):
    raise NotImplementedError("cuda.coop: kernel only")


OPERATIONS = {
    _decode: ("run_length_decode", False),
    _qualified_decode: ("run_length_decode", True),
    _decode_into: ("run_length_decode_into", False),
    _qualified_decode_into: ("run_length_decode_into", True),
}


def _integer_dtype(dtype, name):
    dtype = nb.unliteral(dtype)
    if not isinstance(dtype, nb.types.Integer) or dtype.bitwidth > 64:
        raise TypedPipelineError(
            f"run_length_decode {name} must have an integer dtype up to 64 bits"
        )
    return dtype


def _offset_dtype(value):
    if _templates._omitted(value):
        return nb.types.uint32
    if isinstance(value, nb.types.DTypeSpec):
        value = value.dtype
    elif isinstance(value, nb.types.Literal):
        value = normalize_dtype_param(value.literal_value)
    if value not in (nb.types.uint32, nb.types.uint64):
        raise TypedPipelineError(
            "decoded_offset_dtype must be uint32 or uint64"
        )
    return value


def _decoded_extent(kws):
    value = kws.get("decoded_items_per_thread")
    if not isinstance(value, nb.types.IntegerLiteral):
        raise TypedPipelineError(
            "decoded_items_per_thread must be a positive compile-time integer"
        )
    return normalize_positive_int(
        "decoded_items_per_thread", value.literal_value
    )


def _static_integer(value):
    if isinstance(value, Integral) and not isinstance(value, bool):
        return nb.types.IntegerLiteral(int(value))
    return None


def _payload(kind, name, value, qualified):
    if not isinstance(value, nb.types.Array):
        raise TypedPipelineError(
            f"{kind} {name} must be a fixed per-thread array"
        )
    if not qualified and not isinstance(value, ThreadDataType):
        raise TypedPipelineError(
            f"cuda.coop.{kind} requires {name} to be ThreadData"
        )
    if value.ndim != 1 or value.layout != "C":
        raise TypedPipelineError(
            f"{kind} {name} must be contiguous and one-dimensional"
        )
    if not value.dtype.is_precise():
        return None
    if name == "run_lengths":
        return _integer_dtype(value.dtype, name)
    return _validate_common_numeric_dtype(
        value.dtype, operation=kind, parameter=name
    )


def _extent(value, name, extents):
    return (
        value.items if isinstance(value, ThreadDataType) else extents.get(name)
    )


def _destination(value, dtype, name):
    if (
        not isinstance(value, nb.types.Array)
        or value.ndim != 1
        or value.layout != "C"
        or not value.mutable
        or value.dtype != dtype
    ):
        raise TypedPipelineError(
            f"run_length_decode_into {name} must be a writable contiguous "
            f"one-dimensional array with dtype {dtype}"
        )


@dataclass(frozen=True, eq=False)
class DecodeRequest:
    kind: str
    dtype: Any
    run_length_dtype: Any
    offset_dtype: Any
    control_dtype: Any
    runs_per_thread: int
    items_per_thread: int
    group: CoopGroupType
    call: Any
    controls: tuple
    storage_options: tuple
    has_total: bool
    has_relative: bool

    @property
    def bulk(self):
        return self.kind == "run_length_decode_into"

    @property
    def algorithm(self):
        return "bulk" if self.bulk else "window"

    @property
    def key(self):
        return self.call.semantic_key, self.storage_options, self.has_total

    @property
    def allocations(self):
        if self.bulk:
            return ()
        allocations = [ArrayAllocation(self.dtype, self.items_per_thread)]
        if not self.has_total:
            allocations.append(ArrayAllocation(self.offset_dtype, 1))
        if not self.has_relative:
            allocations.append(
                ArrayAllocation(self.offset_dtype, self.items_per_thread)
            )
        return tuple(allocations)

    @property
    def result(self):
        return ("return",) if self.bulk else ("allocation", 0)

    @property
    def provider_operands(self):
        operands: list[tuple] = [("pointer", 1), ("pointer", 2)]
        if self.bulk:
            operands.extend((("pointer", 3), ("length", 3)))
            if self.has_relative:
                operands.extend(
                    (
                        ("pointer", "relative_offsets"),
                        ("length", "relative_offsets"),
                    )
                )
        else:
            operands.append(("allocation", 0))
            allocation = 1
            for name, present in (
                ("total_decoded_size", self.has_total),
                ("relative_offsets", self.has_relative),
            ):
                if present:
                    operands.append(("pointer", name))
                else:
                    operands.append(("allocation", allocation))
                    allocation += 1
        for name, binding in self.controls:
            if binding.kind is BindingKind.RUNTIME:
                operands.append(("value", name))
        return tuple(operands)

    def complete(self, block, plan):
        from .._lowering import _run_length

        factory = getattr(
            _run_length,
            self.kind + ("_offsets" if self.bulk and self.has_relative else ""),
        )
        return factory, {
            "item_dtype": self.dtype,
            "run_length_dtype": self.run_length_dtype,
            "decoded_offset_dtype": self.offset_dtype,
            "control_dtype": self.control_dtype,
            "threads_per_block": block,
            "runs_per_thread": self.runs_per_thread,
            "decoded_items_per_thread": self.items_per_thread,
            "offset": self.controls[0][1],
            "relative_offsets": self.has_relative,
        }


def request_for(kind, args, kws, *, qualified, extents, controls):
    bulk = kind == "run_length_decode_into"
    if len(args) != (4 if bulk else 3) or not isinstance(
        args[0], CoopGroupType
    ):
        raise TypedPipelineError(f"coop.{kind}: expected group and run arrays")
    dtypes, sizes = [], []
    for name, value in zip(
        ("run_values", "run_lengths"), args[1:3], strict=True
    ):
        dtype = _payload(kind, name, value, qualified)
        if dtype is None:
            return None
        dtypes.append(dtype)
        sizes.append(_extent(value, name, extents))
    if sizes[0] is None or sizes[0] != sizes[1]:
        raise TypedPipelineError(
            f"{kind} run values and lengths require matching fixed extents"
        )
    extent = _decoded_extent(kws)
    offset_dtype = _offset_dtype(kws.get("decoded_offset_dtype"))
    has_total = not bulk and not _templates._omitted(
        kws.get("total_decoded_size")
    )
    has_relative = not _templates._omitted(kws.get("relative_offsets"))
    if bulk:
        _destination(args[3], dtypes[0], "destination")
        if has_relative:
            _destination(
                kws["relative_offsets"], offset_dtype, "relative_offsets"
            )
    else:
        for name, expected in (
            ("total_decoded_size", 1),
            ("relative_offsets", extent),
        ):
            value = kws.get(name)
            if _templates._omitted(value):
                continue
            if (
                not isinstance(value, nb.types.Array)
                or _extent(value, name, extents) != expected
            ):
                raise TypedPipelineError(
                    f"{name} requires a fixed per-thread payload "
                    f"of extent {expected}"
                )
            if value.ndim != 1 or value.layout != "C" or not value.mutable:
                raise TypedPipelineError(
                    f"{name} requires a writable contiguous per-thread payload"
                )
            if value.dtype != offset_dtype:
                raise TypedPipelineError(
                    f"{name} dtype must match decoded_offset_dtype"
                )
    name, binding = controls[0]
    control_dtype = (
        _integer_dtype(kws[name], name)
        if binding.kind is BindingKind.RUNTIME
        else nb.types.uint64
    )
    adapter = NumbaMlirCoreAdapter()
    semantics = GroupRunLengthDecodeSemantics(
        item_dtype=adapter.core_dtype(dtypes[0]),
        run_length_dtype=adapter.core_dtype(dtypes[1]),
        runs_per_thread=sizes[0],
        decoded_items_per_thread=extent,
        offset=binding,
        decoded_offset_dtype=adapter.core_dtype(offset_dtype),
        control_dtype=adapter.core_dtype(control_dtype),
        bulk=bulk,
        relative_offsets=has_relative,
    )
    return DecodeRequest(
        kind,
        dtypes[0],
        dtypes[1],
        offset_dtype,
        control_dtype,
        sizes[0],
        extent,
        args[0],
        make_group_primitive_call(args[0].group, semantics),
        controls,
        tuple(_templates.storage_options(kws.get("temp_storage")).items()),
        has_total,
        has_relative,
    )


def site_request(analysis, stmt, kind, qualified):
    from ._sites import _RUNTIME

    call = stmt.value
    args = [analysis.typemap[arg.name] for arg in call.args]
    kws = {name: analysis.typemap[value.name] for name, value in call.kws}
    args, kws = static_call_arguments(
        analysis.func_ir,
        call,
        analysis.typemap[call.func.name].typing_key,
        args,
        kws,
    )
    variables: dict[str, Any] = dict(
        zip(("group", "run_values", "run_lengths", "destination"), call.args)
    )
    variables.update(call.kws)
    name = (
        "destination_offset"
        if kind == "run_length_decode_into"
        else "decoded_window_offset"
    )
    variable = variables.get(name)
    provenance = (
        analysis._provenance(variable) if variable is not None else None
    )
    binding = (
        ArgumentBinding.static(0)
        if provenance is None
        else ArgumentBinding.runtime()
        if provenance is _RUNTIME
        else ArgumentBinding.static(provenance.value)
    )
    names = ["run_values", "run_lengths"]
    if kind == "run_length_decode":
        names.extend(
            name
            for name in ("total_decoded_size", "relative_offsets")
            if name in kws and not _templates._omitted(kws[name])
        )
    return request_for(
        kind,
        args,
        kws,
        qualified=qualified,
        extents={
            name: analysis._value_extent(variables[name], kind)
            for name in names
        },
        controls=((name, binding),),
    )


def _typing(kind, qualified):
    def generic(self, args, kws):
        bulk = kind == "run_length_decode_into"
        if len(args) != (4 if bulk else 3) or not isinstance(
            args[0], CoopGroupType
        ):
            raise TypedPipelineError(
                f"coop.{kind}: expected group and run arrays"
            )
        dtype = _payload(kind, "run_values", args[1], qualified)
        lengths = _payload(kind, "run_lengths", args[2], qualified)
        if dtype is None or lengths is None:
            return None
        result = (
            _offset_dtype(kws.get("decoded_offset_dtype"))
            if bulk
            else ThreadDataType(dtype, _decoded_extent(kws), PRODUCER)
        )
        return nb.signature(result, *args, *kws.values())

    return generic


def register():
    for operation, (kind, qualified) in OPERATIONS.items():
        _templates.OPERATIONS[operation] = kind
        if qualified:
            _templates.QUALIFIED_OPERATIONS.add(operation)
        operation_literal_arguments[operation] = {
            "decoded_items_per_thread": _static_integer
        }
        if kind == "run_length_decode":
            operation_inference_contracts[operation] = OperationInference(
                checked_producers=tuple(
                    (name, DtypeSource("decoded_offset_dtype", _offset_dtype))
                    for name in ("total_decoded_size", "relative_offsets")
                )
            )
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
