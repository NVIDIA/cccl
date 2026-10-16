# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Type batched Warp Reduce results with their smaller independent extent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from . import _bridges as nb
from . import _templates
from ._callbacks import constant
from ._errors import TypedPipelineError
from ._results import ArrayAllocation
from ._types import PRODUCER, CoopGroupType, ThreadDataType


def _reduce_batched(group, value, **kwargs):
    raise NotImplementedError("cuda.coop: kernel only")


def _qualified_reduce_batched(group, value, **kwargs):
    raise NotImplementedError("cuda.coop: kernel only")


OPERATIONS = {_reduce_batched: False, _qualified_reduce_batched: True}


def _facts(args, kws, qualified):
    from .._compiler._parameters import _validate_common_numeric_dtype
    from .._lowering._reduce_batched import reduction_operator

    if len(args) != 2 or not isinstance(args[0], CoopGroupType):
        raise TypedPipelineError(
            "coop.reduce_batched: expected group and value"
        )
    group, value = args
    if (
        not isinstance(value, nb.types.Array)
        or value.ndim != 1
        or value.layout != "C"
    ):
        raise TypedPipelineError("reduce_batched requires a fixed-size payload")
    if not qualified and not isinstance(value, ThreadDataType):
        raise TypedPipelineError("cuda.coop.reduce_batched requires ThreadData")
    if not value.dtype.is_precise():
        return None
    dtype = _validate_common_numeric_dtype(
        value.dtype, operation="reduce_batched"
    )
    width = group.group.static_size
    if width is None or group.group.kind not in {"warp", "threads_within_warp"}:
        raise TypedPipelineError(
            f"reduce_batched does not support group kind {group.group.kind!r}; "
            "use a statically sized Warp group"
        )
    layout = constant(kws.get("output_layout", nb.types.literal("striped")))
    if layout not in {"striped", "blocked"}:
        raise TypedPipelineError(
            "reduce_batched output_layout must be striped or blocked"
        )
    binary_op = constant(kws.get("binary_op"))
    operator = reduction_operator(
        binary_op, dtype, is_common_root=not qualified
    )
    return dtype, width, layout, binary_op, operator


@dataclass(frozen=True, eq=False)
class BatchedReduceRequest:
    kind: str
    dtype: Any
    items_per_thread: int
    output_items: int
    output_layout: str
    binary_op: Any
    group: CoopGroupType
    call: Any
    controls: tuple = ()
    algorithm: None = None

    @property
    def key(self):
        return self.call.semantic_key

    @property
    def allocations(self):
        return (ArrayAllocation(self.dtype, self.output_items),)

    @property
    def result(self):
        return ("allocation", 0)

    @property
    def provider_operands(self):
        return ("pointer", 1), ("allocation", 0)

    def complete(self, block, plan):
        from .._lowering import _reduce_batched as providers

        return providers.reduce_batched, {
            "dtype": self.dtype,
            "threads_per_block": block,
            "threads_in_warp": plan.topology.logical_width,
            "batches": self.items_per_thread,
            "binary_op": self.binary_op,
            "output_layout": self.output_layout,
        }


_REQUESTS = {}


def request_for(args, kws, *, qualified, extent=None):
    from ..._core import make_group_primitive_call
    from ..._core.group.reduce_batched import GroupReduceBatchedSemantics
    from ..._core.warp.reduce_batched import WarpReduceBatchedSemantics
    from .._lowering._core import NumbaMlirCoreAdapter

    facts = _facts(args, kws, qualified)
    if facts is None:
        return None
    dtype, width, layout, binary_op, operator = facts
    items = args[1].items if isinstance(args[1], ThreadDataType) else extent
    if items is None:
        raise TypedPipelineError("reduce_batched requires a static batch count")
    call = make_group_primitive_call(
        args[0].group,
        GroupReduceBatchedSemantics(
            WarpReduceBatchedSemantics(
                NumbaMlirCoreAdapter().core_dtype(dtype),
                items,
                operator,
                layout,
            )
        ),
    )
    key = call.semantic_key
    if key not in _REQUESTS:
        _REQUESTS[key] = BatchedReduceRequest(
            "reduce_batched",
            dtype,
            items,
            (items + width - 1) // width,
            layout,
            binary_op,
            args[0],
            call,
        )
    return _REQUESTS[key]


def _typing(qualified):
    def generic(self, args, kws):
        facts = _facts(args, kws, qualified)
        if facts is None:
            return None
        dtype, width, _, _, _ = facts
        result = (
            ThreadDataType(
                dtype, (args[1].items + width - 1) // width, PRODUCER
            )
            if isinstance(args[1], ThreadDataType)
            else nb.types.Array(dtype, 1, "C")
        )
        return nb.signature(result, *args, *kws.values())

    return generic


def result_extent(analysis, expression):
    value = expression.args[1]
    ty = analysis.typemap[value.name]
    items = (
        ty.items
        if isinstance(ty, ThreadDataType)
        else analysis._value_extent(value, "reduce_batched")
    )
    width = analysis.typemap[expression.args[0].name].group.static_size
    return (
        None if items is None or width is None else (items + width - 1) // width
    )


def register():
    from ._callbacks import StaticCallableType
    from ._inference import operation_static_arguments

    for operation, qualified in OPERATIONS.items():
        _templates.OPERATIONS[operation] = "reduce_batched"
        _templates.RESULT_EXTENT_SOURCES[operation] = result_extent
        if qualified:
            _templates.QUALIFIED_OPERATIONS.add(operation)
            operation_static_arguments[operation] = {
                "binary_op": StaticCallableType
            }
        _templates.register_template(
            operation,
            _typing(qualified),
            operation.__name__,
            prefer_literal=True,
        )
    for public, name in _templates.kernel_primitives().items():
        if _templates.group_operation(public) != "reduce_batched":
            continue
        qualified = name.startswith("cuda.coop.numba_mlir.")
        operation = _qualified_reduce_batched if qualified else _reduce_batched
        nb.attach_inline_wrapper(
            public, _templates.result_wrapper(public, name, operation)
        )
        _templates.register_template(
            public, _templates._not_inlined(name), name
        )
