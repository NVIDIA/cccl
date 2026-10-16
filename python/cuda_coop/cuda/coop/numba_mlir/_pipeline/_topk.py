# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Type fresh TopK results and preserve checked count provider operands."""

from dataclasses import dataclass
from typing import Any

from . import _bridges as nb
from . import _sorting, _templates
from ._errors import TypedPipelineError
from ._results import ArrayAllocation

KINDS = frozenset(
    f"topk_{selection}_{payload}"
    for selection in ("min", "max")
    for payload in ("keys", "pairs")
)
OPERATIONS = _sorting.operations(KINDS)


@dataclass(frozen=True, eq=False)
class TopKRequest:
    kind: str
    dtype: Any
    value_dtype: Any
    items_per_thread: int
    group: Any
    call: Any
    controls: tuple
    storage_options: tuple

    @property
    def key(self):
        return self.call.semantic_key, self.storage_options

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

        return tuple(
            ("allocation", i) for i in range(len(self.allocations))
        ) + tuple(
            ("value", name)
            for name, bound in self.controls
            if bound.kind is BindingKind.RUNTIME
        )

    def complete(self, block, plan):
        from .._lowering import _topk

        controls = dict(self.controls)
        kwargs = {
            "key_dtype": self.dtype,
            "value_dtype": self.value_dtype,
            "threads_per_block": block,
            "items_per_thread": self.items_per_thread,
            "selection": self.kind.split("_")[1],
            "k": controls["k"],
            "num_valid": controls["valid_items"],
        }
        return (
            (
                _topk.topk_pairs
                if self.value_dtype is not None
                else _topk.topk_keys
            ),
            kwargs,
        )


def bind_controls(analysis, stmt, dtype):
    return tuple(
        (name, _sorting.integer_binding(analysis, stmt, "topk", name))
        for name in ("k", "valid_items")
    )


def request_for(kind, args, kws, *, qualified, extents=None, controls=()):
    from ..._core import GroupTopKSemantics, make_group_primitive_call
    from .._lowering._core import NumbaMlirCoreAdapter

    payload = _sorting.payloads(kind, args, qualified, extents)
    if payload is None:
        return None
    dtypes, items, _ = payload
    if items is None:
        raise TypedPipelineError(f"coop.{kind}: payload extent must be static")
    adapter = NumbaMlirCoreAdapter()
    bounds = dict(controls)
    key_dtype = dtypes[0]
    value_dtype = dtypes[1] if len(dtypes) > 1 else None
    semantics = GroupTopKSemantics(
        key_dtype=adapter.core_dtype(key_dtype),
        value_dtype=adapter.core_dtype(value_dtype),
        items_per_thread=items,
        selection=kind.split("_")[1],
        k=bounds["k"],
        valid_items=bounds["valid_items"],
    )
    return TopKRequest(
        kind,
        key_dtype,
        value_dtype,
        items,
        args[0],
        make_group_primitive_call(args[0].group, semantics),
        tuple(controls),
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


def register():
    _sorting.register(OPERATIONS, _typing)


def site_request(analysis, stmt, kind, qualified):
    args, kws, extents = _sorting.site_arguments(analysis, stmt, kind)
    return request_for(
        kind,
        args,
        kws,
        qualified=qualified,
        extents=extents,
        controls=bind_controls(analysis, stmt, None),
    )
