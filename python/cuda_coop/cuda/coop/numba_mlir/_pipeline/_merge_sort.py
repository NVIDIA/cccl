# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Native MergeSort result types and checked partial-tile operands."""

from dataclasses import dataclass
from typing import Any

from . import _bridges as nb
from . import _callbacks, _sorting, _templates
from ._errors import TypedPipelineError
from ._results import ArrayAllocation

KINDS = frozenset({"merge_sort_keys", "merge_sort_pairs"})
OPERATIONS = _sorting.operations(KINDS)


@dataclass(frozen=True, eq=False)
class MergeSortRequest:
    kind: str
    dtype: Any
    value_dtype: Any
    items_per_thread: int
    group: Any
    call: Any
    controls: tuple
    descending: bool
    compare_op: Any
    storage_options: tuple

    @property
    def key(self):
        return self.call.semantic_key, self.controls, self.storage_options

    @property
    def algorithm(self):
        return self.kind

    @property
    def allocations(self):
        dtypes = (self.dtype,) + (
            (self.value_dtype,) if self.value_dtype is not None else ()
        )
        return tuple(
            ArrayAllocation(dtype, self.items_per_thread, copy_from=i + 1)
            for i, dtype in enumerate(dtypes)
        )

    @property
    def result(self):
        if self.value_dtype is None:
            return ("allocation", 0)
        return ("tuple", (("allocation", 0), ("allocation", 1)))

    @property
    def provider_operands(self):
        from ..._core import BindingKind

        operands = tuple(
            ("allocation", i) for i in range(len(self.allocations))
        )
        if dict(self.controls)["valid_items"].kind is not BindingKind.OMITTED:
            operands += tuple(
                _sorting.scalar_operand(name, bound)
                for name, bound in self.controls
            )
        return operands

    def complete(self, block, plan):
        from ..._core import BindingKind, GroupLoweringTarget
        from .._lowering import _merge_sort

        namespace = (
            "block" if plan.target is GroupLoweringTarget.CUB_BLOCK else "warp"
        )
        partial = (
            dict(self.controls)["valid_items"].kind is not BindingKind.OMITTED
        )
        factory = getattr(
            _merge_sort,
            f"{namespace}_{self.kind}" + ("_partial" if partial else ""),
        )
        kwargs = {
            "key_dtype": self.dtype,
            "threads_per_block": block,
            "items_per_thread": self.items_per_thread,
            "descending": self.descending,
            "compare_op": self.compare_op,
        }
        if self.value_dtype is not None:
            kwargs["value_dtype"] = self.value_dtype
        if namespace == "warp":
            kwargs["threads_in_warp"] = plan.topology.logical_width
        return factory, kwargs


def bind_controls(analysis, stmt, dtype):
    from ..._core import ArgumentBinding, BindingKind
    from .._compiler._parameters import (
        _validate_common_numeric_dtype,
        coerce_static_sort_sentinel,
    )
    from .._compiler._scalar_provenance import _typed_static_value

    valid = _sorting.integer_binding(
        analysis, stmt, "merge_sort", "valid_items"
    )
    sentinel = _sorting.binding(analysis, stmt, "oob_default")
    if (valid.kind is BindingKind.OMITTED) != (
        sentinel.kind is BindingKind.OMITTED
    ):
        raise TypedPipelineError(
            "Merge Sort valid_items and oob_default must be provided together"
        )
    if sentinel.kind is BindingKind.STATIC:
        provenance = analysis._provenance(dict(stmt.value.kws)["oob_default"])
        try:
            value = coerce_static_sort_sentinel(
                _typed_static_value(provenance), dtype, operation="merge_sort"
            )
        except (TypeError, ValueError) as exc:
            raise TypedPipelineError(str(exc)) from None
        sentinel = ArgumentBinding.static(value)
    elif sentinel.kind is BindingKind.RUNTIME:
        ty = analysis.typemap[dict(stmt.value.kws)["oob_default"].name]
        actual = _validate_common_numeric_dtype(
            ty, operation="merge_sort", parameter="oob_default"
        )
        if actual != dtype:
            raise TypedPipelineError(
                f"Merge Sort oob_default dtype {actual} does not "
                f"match keys dtype {dtype}"
            )
    return (("valid_items", valid), ("oob_default", sentinel))


def request_for(kind, args, kws, *, qualified, extents=None, controls=()):
    from ..._core import (
        BindingKind,
        GroupMergeSortSemantics,
        make_block_merge_sort_semantics,
        make_group_primitive_call,
    )
    from .._lowering._core import NumbaMlirCoreAdapter
    from .._lowering._merge_sort import comparison_operator

    payload = _sorting.payloads(kind, args, qualified, extents)
    if payload is None:
        return None
    dtypes, items, _ = payload
    if items is None:
        raise TypedPipelineError(f"coop.{kind}: payload extent must be static")
    descending = _callbacks.constant(kws.get("descending"))
    compare_op = _callbacks.constant(kws.get("compare_op"))
    comparator = comparison_operator(descending, compare_op)
    valid = dict(controls)["valid_items"]
    partial = valid.kind is not BindingKind.OMITTED
    adapter = NumbaMlirCoreAdapter()
    key_dtype = dtypes[0]
    value_dtype = dtypes[1] if len(dtypes) > 1 else None
    semantics = GroupMergeSortSemantics(
        make_block_merge_sort_semantics(
            key_dtype=adapter.core_dtype(key_dtype),
            value_dtype=adapter.core_dtype(value_dtype),
            items_per_thread=items,
            compare_operator=comparator,
            valid_items=0 if partial else None,
            oob_default=0 if partial else None,
        ),
        valid_items=valid,
    )
    return MergeSortRequest(
        kind,
        key_dtype,
        value_dtype,
        items,
        args[0],
        make_group_primitive_call(args[0].group, semantics),
        tuple(controls),
        descending,
        compare_op,
        tuple(_templates.storage_options(kws.get("temp_storage")).items()),
    )


def _typing(kind, qualified):
    def generic(self, args, kws):
        payload = _sorting.payloads(kind, args, qualified)
        if payload is None:
            return None
        dtypes, _, _ = payload
        results = tuple(
            _sorting.result_type(value, dtype)
            for value, dtype in zip(args[1:], dtypes, strict=True)
        )
        result = (
            results[0]
            if len(results) == 1
            else nb.types.BaseTuple.from_types(results)
        )
        return nb.signature(result, *args, *kws.values())

    return generic


def site_request(analysis, stmt, kind, qualified):
    args, kws, extents = _sorting.site_arguments(analysis, stmt, kind)
    return request_for(
        kind,
        args,
        kws,
        qualified=qualified,
        extents=extents,
        controls=bind_controls(analysis, stmt, args[1].dtype),
    )


def register():
    _sorting.register(OPERATIONS, _typing, callbacks=("compare_op",))
