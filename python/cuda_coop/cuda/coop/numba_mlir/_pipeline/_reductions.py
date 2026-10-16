# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Type reduction results and select CUB providers from native facts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from . import _bridges as nb
from . import _templates
from ._errors import TypedPipelineError
from ._types import CoopGroupType, ThreadDataType


def _reduce(group, value, **kwargs):
    raise NotImplementedError("cuda.coop: kernel only")


def _sum(group, value, **kwargs):
    raise NotImplementedError("cuda.coop: kernel only")


def _qualified_reduce(group, value, **kwargs):
    raise NotImplementedError("cuda.coop: kernel only")


def _qualified_sum(group, value, **kwargs):
    raise NotImplementedError("cuda.coop: kernel only")


OPERATIONS = {
    _reduce: ("reduce", False),
    _sum: ("sum", False),
    _qualified_reduce: ("reduce", True),
    _qualified_sum: ("sum", True),
}


def _constant(ty):
    from ._callbacks import constant

    return constant(ty)


def _payload(kind, value, qualified):
    from .._compiler._parameters import _validate_common_numeric_dtype

    if isinstance(value, nb.types.Array):
        if not qualified and not isinstance(value, ThreadDataType):
            raise TypedPipelineError(
                f"cuda.coop.{kind} accepts only a scalar or fixed-size "
                "ThreadData value payload; use cuda.coop.numba_mlir for "
                "backend-qualified local arrays"
            )
        dtype = value.dtype
    else:
        dtype = nb.unliteral(value)
    if not dtype.is_precise():
        return None
    return _validate_common_numeric_dtype(
        dtype, operation=kind, parameter="value"
    )


@dataclass(frozen=True, eq=False)
class ReductionRequest:
    """A scalar-return operation with explicit provider input bindings."""

    kind: str
    dtype: Any
    items_per_thread: int
    algorithm: str | None
    group: CoopGroupType
    controls: tuple
    call: Any
    operator_kind: str
    binary_op: Any
    is_array: bool

    @property
    def key(self):
        return self.call.semantic_key

    @property
    def provider_operands(self):
        result = [("pointer" if self.is_array else "value", 1)]
        result.extend(
            ("value", name)
            for name, binding in self.controls
            if binding.kind.value == "runtime"
        )
        return tuple(result)

    def complete(self, block, plan):
        from ..._core import GroupLoweringTarget
        from .._lowering import _reduce as providers

        warp = plan.target is GroupLoweringTarget.CUB_WARP
        if not warp and plan.target is not GroupLoweringTarget.CUB_BLOCK:
            raise TypedPipelineError(
                f"coop.{self.kind}: unsupported reduction target {plan.target}"
            )
        if self.operator_kind == "sum":
            factory = providers.warp_sum if warp else providers.sum
        elif self.operator_kind == "callback":
            factory = providers.warp_reduce if warp else providers.reduce
        else:
            factory = (
                providers.warp_reduce_builtin
                if warp
                else providers.block_reduce_builtin
            )
        kwargs = {
            "dtype": self.dtype,
            "threads_per_block": block,
            "items_per_thread": self.items_per_thread,
            "value_kind": "array" if self.is_array else "scalar",
        }
        if warp:
            kwargs["threads_in_warp"] = plan.topology.logical_width
        else:
            kwargs["algorithm"] = self.algorithm or "warp_reductions"
        if self.operator_kind != "sum":
            kwargs["binary_op"] = (
                self.binary_op
                if self.operator_kind == "callback"
                else self.operator_kind
            )
        bindings = dict(self.controls)
        if "valid_items" in bindings:
            kwargs["valid_items" if warp else "num_valid"] = bindings[
                "valid_items"
            ]
        return factory, kwargs


_REQUESTS = {}


def request_for(kind, args, kws, *, qualified, extent=None, controls=()):
    from ..._core import (
        ArgumentBinding,
        CxxOperator,
        Dependency,
        GroupReduceSemantics,
        PythonOperator,
        make_group_primitive_call,
        make_reduce_semantics,
    )
    from .._lowering._core import NumbaMlirCoreAdapter
    from .._lowering._reduce import (
        _BUILTIN_REDUCE_OPERATORS,
        normalize_reduce_operation,
        validate_reduce_operator_dtype,
    )
    from .._semantic import _normalize_numba_callable, _numba_semantic_token

    if len(args) != 2:
        raise TypedPipelineError(f"coop.{kind}: expected group and value")
    group, value = args
    if not isinstance(group, CoopGroupType):
        raise TypedPipelineError(f"coop.{kind}: group must be a thread group")
    storage = _templates.storage_options(kws.get("temp_storage"))
    dtype = _payload(kind, value, qualified)
    if dtype is None:
        return None
    is_array = isinstance(value, nb.types.Array)
    items = value.items if isinstance(value, ThreadDataType) else extent
    if not is_array:
        items = 1
    if items is None:
        return None
    binary_op = None if kind == "sum" else _constant(kws.get("binary_op"))
    if (
        not qualified
        and binary_op is not None
        and not isinstance(binary_op, str)
    ):
        raise TypedPipelineError("cuda.coop.reduce binary_op must be a string")
    try:
        operator_kind = normalize_reduce_operation(binary_op)
    except NotImplementedError:
        operator_kind = "callback"
    semantic_kind = "sum" if operator_kind == "sum" else "reduce"
    operator = None
    if operator_kind == "callback":
        operator = PythonOperator(
            op_tokenizer=_numba_semantic_token,
            ret_dtype=Dependency("T"),
            arg_dtypes=(Dependency("T"), Dependency("T")),
            op=_normalize_numba_callable(binary_op),
            name="binary_op",
        )
    else:
        validate_reduce_operator_dtype(operator_kind, dtype)
        if operator_kind != "sum":
            operator = CxxOperator(
                cpp=_BUILTIN_REDUCE_OPERATORS[operator_kind],
                dtype=Dependency("T"),
                name="binary_op",
            )
    algorithm = _constant(kws.get("algorithm"))
    if algorithm is not None:
        if not isinstance(algorithm, str):
            raise TypedPipelineError(f"coop.{kind}: algorithm must be a string")
        algorithm = algorithm.strip().lower().replace("-", "_")
        if algorithm not in {
            "raking",
            "raking_commutative_only",
            "warp_reductions",
        }:
            raise TypedPipelineError(
                f"coop.{kind}: unsupported algorithm {algorithm!r}"
            )
    primitive = make_reduce_semantics(
        dtype=NumbaMlirCoreAdapter().core_dtype(dtype),
        items_per_thread=items,
        operation=semantic_kind,
        value_kind="array" if is_array else "scalar",
        reduce_operator=operator,
        valid_items=dict(controls).get(
            "valid_items", ArgumentBinding.omitted()
        ),
    )
    call = make_group_primitive_call(
        group.group,
        GroupReduceSemantics(primitive, cub_algorithm=algorithm, **storage),
    )
    key = (kind, call.semantic_key)
    request = _REQUESTS.get(key)
    if request is None:
        request = _REQUESTS[key] = ReductionRequest(
            kind,
            dtype,
            items,
            algorithm,
            group,
            tuple(controls),
            call,
            operator_kind,
            binary_op,
            is_array,
        )
    return request


def _typing(kind, qualified):
    def generic(self, args, kws):
        if len(args) != 2:
            raise TypedPipelineError(f"coop.{kind}: expected group and value")
        dtype = _payload(kind, args[1], qualified)
        if dtype is None:
            return None
        # Plain local-array extents are recovered from typed IR at resolution.
        if not isinstance(args[1], nb.types.Array) or isinstance(
            args[1], ThreadDataType
        ):
            request_for(kind, args, kws, qualified=qualified)
        return nb.signature(dtype, *args, *kws.values())

    return generic


def register():
    from ._callbacks import StaticCallableType
    from ._inference import operation_static_arguments

    for operation, (kind, qualified) in OPERATIONS.items():
        _templates.OPERATIONS[operation] = kind
        if qualified:
            _templates.QUALIFIED_OPERATIONS.add(operation)
            operation_static_arguments[operation] = {
                "binary_op": StaticCallableType,
            }
        _templates.register_template(
            operation,
            _typing(kind, qualified),
            operation.__name__,
            prefer_literal=True,
        )
    for public, name in _templates.kernel_primitives().items():
        kind = _templates.group_operation(public)
        if kind not in {"reduce", "sum"}:
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
