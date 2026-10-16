# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Type Scan results and bind native operands to shared Scan providers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from . import _bridges as nb
from . import _callbacks, _templates
from ._callbacks import constant
from ._errors import TypedPipelineError
from ._inference import (
    OperationInference,
    operation_inference_contracts,
    operation_static_arguments,
)
from ._reductions import _payload
from ._results import ArrayAllocation
from ._types import PRODUCER, CoopGroupType, ThreadDataType

KINDS = frozenset(
    {
        "scan",
        "inclusive_scan",
        "exclusive_scan",
        "inclusive_sum",
        "exclusive_sum",
    }
)


def _operation(kind, qualified):
    def operation(group, value, **kwargs):
        raise NotImplementedError("cuda.coop: kernel only")

    operation.__name__ = f"_coop_{'qualified_' if qualified else ''}{kind}"
    return operation


OPERATIONS = {
    _operation(kind, qualified): (kind, qualified)
    for kind in sorted(KINDS)
    for qualified in (False, True)
}


def _result_type(value, dtype):
    if isinstance(value, ThreadDataType):
        return ThreadDataType(dtype, value.items, PRODUCER)
    if isinstance(value, nb.types.Array):
        return nb.types.Array(dtype, 1, "C")
    return dtype


def _auxiliary(value, name, dtype, extents):
    if _templates._omitted(value):
        return False
    if not isinstance(value, nb.types.Array):
        raise TypedPipelineError(
            f"coop.scan {name} must be a one-item ThreadData or local array"
        )
    items = (
        value.items if isinstance(value, ThreadDataType) else extents.get(name)
    )
    if items != 1:
        raise TypedPipelineError(
            f"coop.scan {name} must contain exactly one item"
        )
    if value.dtype != dtype:
        raise TypedPipelineError(
            f"coop.scan {name} dtype {value.dtype} does not match "
            f"the required dtype {dtype}"
        )
    return True


@dataclass(frozen=True, eq=False)
class ScanRequest:
    kind: str
    dtype: Any
    items_per_thread: int
    algorithm: str | None
    group: CoopGroupType
    controls: tuple
    call: Any
    is_array: bool
    mode: str
    scan_op: Any
    aggregate: bool
    storage_options: tuple

    @property
    def key(self):
        return self.call.semantic_key, self.storage_options

    @property
    def allocations(self):
        return (
            (ArrayAllocation(self.dtype, self.items_per_thread),)
            if self.is_array
            else ()
        )

    @property
    def result(self):
        return ("allocation", 0) if self.is_array else ("return",)

    @property
    def provider_operands(self):
        from ..._core import BindingKind

        result: list[tuple] = [("pointer" if self.is_array else "value", 1)]
        if self.is_array:
            result.append(("allocation", 0))
        controls = dict(self.controls)
        initial = controls.get("initial_value")
        if initial is not None and initial.kind is BindingKind.RUNTIME:
            result.append(("value", "initial_value"))
        valid = controls.get("valid_items")
        if valid is not None and valid.kind is BindingKind.RUNTIME:
            result.append(("value", "valid_items"))
        if self.aggregate:
            result.append(("pointer", "aggregate_output"))
        return tuple(result)

    def complete(self, block, plan):
        from ..._core import GroupLoweringTarget
        from .._lowering import _scan as providers

        kwargs = {
            "dtype": self.dtype,
            "mode": self.mode,
            "scan_op": self.scan_op,
        }
        controls = dict(self.controls)
        if "initial_value" in controls:
            kwargs["initial_value"] = controls["initial_value"]
        if plan.target is GroupLoweringTarget.CUB_BLOCK:
            factory = (
                providers.block_scan_array
                if self.is_array
                else providers.block_scan_scalar
            )
            kwargs.update(
                threads_per_block=block,
                items_per_thread=self.items_per_thread,
                value_kind="array" if self.is_array else "scalar",
                algorithm=plan.call.operation.cub_algorithm.name.lower(),
            )
            if self.aggregate:
                kwargs["block_aggregate"] = True
        else:
            factory = providers.warp_scan
            kwargs.update(
                threads_in_warp=plan.topology.logical_width,
                threads_per_block=block,
            )
            if "valid_items" in controls:
                kwargs["valid_items"] = controls["valid_items"]
            if self.aggregate:
                kwargs["warp_aggregate"] = True
        return factory, kwargs


_REQUESTS = {}


def request_for(kind, args, kws, *, qualified, extents=None, controls=()):
    from ..._core import (
        ArgumentBinding,
        BindingKind,
        GroupScanSemantics,
        make_group_primitive_call,
        make_scan_semantics,
    )
    from .._lowering import _scan as providers
    from .._lowering._core import NumbaMlirCoreAdapter

    if len(args) != 2:
        raise TypedPipelineError(f"coop.{kind}: expected group and value")
    group, value = args[:2]
    if not isinstance(group, CoopGroupType):
        raise TypedPipelineError(f"coop.{kind}: group must be a thread group")
    dtype = _payload(kind, value, qualified)
    if dtype is None:
        return None
    extents = extents or {}
    is_array = isinstance(value, nb.types.Array)
    items = (
        (
            value.items
            if isinstance(value, ThreadDataType)
            else extents.get("value")
        )
        if is_array
        else 1
    )
    if items is None:
        raise TypedPipelineError(f"coop.{kind}: payload extent is not static")
    if group.group.kind in {"warp", "threads_within_warp"} and is_array:
        raise TypedPipelineError(
            "coop WarpScan supports one scalar value per lane"
        )
    mode = providers._scan_mode(
        constant(kws.get("mode")) if kind == "scan" else kind.split("_", 1)[0]
    )
    scan_op = None if kind.endswith("_sum") else constant(kws.get("scan_op"))
    if not qualified and scan_op is not None and not isinstance(scan_op, str):
        raise TypedPipelineError(f"cuda.coop.{kind} scan_op must be a string")
    operator_kind = providers.normalize_scan_operation(scan_op)
    providers.validate_scan_operator_dtype(scan_op, dtype)
    aggregate = _auxiliary(
        kws.get("aggregate_output"), "aggregate_output", dtype, extents
    )
    bindings = dict(controls)
    initial = bindings.get("initial_value", ArgumentBinding.omitted())
    if mode == "inclusive" and initial.kind is not BindingKind.OMITTED:
        raise TypedPipelineError(
            "coop.scan inclusive scan does not accept initial_value"
        )
    if (
        mode == "exclusive"
        and operator_kind != "sum"
        and initial.kind is BindingKind.OMITTED
    ):
        raise TypedPipelineError(
            "coop.scan non-sum exclusive scans require initial_value"
        )
    algorithm = constant(kws.get("algorithm"))
    if algorithm is not None:
        algorithm = providers._block_scan_algorithm(algorithm).name.lower()
    adapter = NumbaMlirCoreAdapter()
    primitive = make_scan_semantics(
        dtype=adapter.core_dtype(dtype),
        mode=mode,
        value_kind="array" if is_array else "scalar",
        items_per_thread=items,
        scan_operator=providers._scan_operator(
            scan_op, force_sum_operator=initial.kind is not BindingKind.OMITTED
        ),
        initial_value=providers._initial_value(initial, dtype),
        aggregate=aggregate,
    )
    call = make_group_primitive_call(
        group.group,
        GroupScanSemantics(
            primitive,
            cub_algorithm=algorithm,
            valid_items=bindings.get("valid_items", ArgumentBinding.omitted()),
        ),
    )
    storage = tuple(_templates.storage_options(kws.get("temp_storage")).items())
    key = (kind, call.semantic_key, storage)
    request = _REQUESTS.get(key)
    if request is None:
        request = _REQUESTS[key] = ScanRequest(
            kind,
            dtype,
            items,
            algorithm,
            group,
            tuple(controls),
            call,
            is_array,
            mode,
            scan_op,
            aggregate,
            storage,
        )
    return request


def bind_controls(analysis, stmt, dtype):
    """Classify seeds and lane counts from typed scalar provenance."""

    from ..._core import ArgumentBinding
    from .._compiler._parameters import (
        _validate_runtime_integer_dtype,
        coerce_static_scalar,
    )
    from ._sites import _RUNTIME

    result = []
    for name, variable in stmt.value.kws:
        if name not in {"initial_value", "valid_items"}:
            continue
        ty = analysis.typemap[variable.name]
        if _templates._omitted(ty):
            continue
        provenance = analysis._provenance(variable)
        if provenance is _RUNTIME:
            if name == "initial_value" and nb.unliteral(ty) != dtype:
                raise TypedPipelineError(
                    f"coop.scan runtime initial_value dtype {ty} does not "
                    f"match value dtype {dtype}"
                )
            if name == "valid_items":
                _validate_runtime_integer_dtype(
                    ty, operation="scan", parameter=name
                )
            binding = ArgumentBinding.runtime()
        else:
            value = provenance.value
            if name == "initial_value":
                value = coerce_static_scalar(
                    value,
                    dtype,
                    operation="scan",
                    parameter=name,
                    source_dtype=provenance.dtype,
                )
            binding = ArgumentBinding.static(value)
        result.append((name, binding))
    return tuple(result)


def _typing(kind, qualified):
    def generic(self, args, kws):
        if len(args) != 2:
            raise TypedPipelineError(f"coop.{kind}: expected group and value")
        dtype = _payload(kind, args[1], qualified)
        if dtype is None:
            return None
        return nb.signature(_result_type(args[1], dtype), *args, *kws.values())

    return generic


def register():
    _callbacks.register()
    for operation, (kind, qualified) in OPERATIONS.items():
        _templates.OPERATIONS[operation] = kind
        _templates.RESULT_EXTENT_SOURCES[operation] = 1
        if qualified:
            _templates.QUALIFIED_OPERATIONS.add(operation)
        operation_inference_contracts[operation] = OperationInference(
            checked_producers=(("aggregate_output", 1),)
        )
        operation_static_arguments[operation] = {
            "scan_op": _callbacks.StaticCallableType,
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
