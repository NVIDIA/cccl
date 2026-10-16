# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Type Radix results and keep runtime bit operands out of specialization."""

from dataclasses import dataclass
from math import prod
from typing import Any

from . import _bridges as nb
from . import _callbacks, _sorting, _templates
from ._errors import TypedPipelineError
from ._results import ArrayAllocation
from ._types import ThreadDataType

KINDS = frozenset({"radix_sort_keys", "radix_sort_pairs", "radix_rank_keys"})
OPERATIONS = _sorting.operations(KINDS)


def _payload(kind, args, qualified, extents=None):
    payload = _sorting.payloads(kind, args, qualified, extents, scalar=True)
    if payload is None:
        return None
    integral = {
        nb.types.int32,
        nb.types.uint32,
        nb.types.int64,
        nb.types.uint64,
    }
    allowed = (
        integral
        if kind == "radix_rank_keys" or not qualified
        else integral | {nb.types.float32, nb.types.float64}
    )
    if payload[0][0] not in allowed:
        raise TypedPipelineError(
            f"cuda.coop.{kind} keys require "
            + (
                "int32, uint32, int64, or uint64"
                if allowed == integral
                else "32- or 64-bit integer or float dtype"
            )
        )
    return payload


def _bool(kws, name):
    value = _callbacks.constant(kws[name]) if name in kws else False
    if not isinstance(value, bool):
        raise TypedPipelineError(f"radix {name} must be a compile-time bool")
    return value


@dataclass(frozen=True, eq=False)
class RadixRequest:
    kind: str
    dtype: Any
    value_dtype: Any
    items_per_thread: int
    is_array: bool
    group: Any
    call: Any
    controls: tuple
    descending: bool
    blocked_to_striped: bool
    prefix_extent: int | None
    digit_bits: int | None
    fixed_radix_bits: int | None
    storage_options: tuple

    @property
    def key(self):
        # Sort's provider key intentionally coalesces static bit intervals.
        # Site requests retain them so each call supplies its own operands.
        return self.call.semantic_key, self.controls, self.storage_options

    @property
    def algorithm(self):
        return self.kind

    @property
    def allocations(self):
        if self.kind == "radix_rank_keys":
            return (ArrayAllocation(nb.types.int32, self.items_per_thread),)
        dtypes = (self.dtype,) + (
            (self.value_dtype,) if self.value_dtype is not None else ()
        )
        return tuple(
            ArrayAllocation(dtype, self.items_per_thread, copy_from=i + 1)
            for i, dtype in enumerate(dtypes)
        )

    @property
    def result(self):
        recipes = tuple(
            (
                ("allocation", i)
                if self.is_array
                else ("scalar", i, allocation.dtype)
            )
            for i, allocation in enumerate(self.allocations)
        )
        return recipes[0] if len(recipes) == 1 else ("tuple", recipes)

    @property
    def provider_operands(self):
        from ..._core import BindingKind

        if self.kind != "radix_rank_keys":
            return tuple(
                ("allocation", i) for i in range(len(self.allocations))
            ) + tuple(
                _sorting.scalar_operand(name, bound)
                for name, bound in self.controls
            )
        operands = (
            ("pointer", 1) if self.is_array else ("box", 1, self.dtype, None),
            ("allocation", 0),
        )
        if self.prefix_extent is not None:
            operands += (("pointer", "exclusive_digit_prefix"),)
        begin = dict(self.controls)["begin_bit"]
        if begin.kind is BindingKind.RUNTIME:
            operands += (("value", "begin_bit"),)
        return operands

    def complete(self, block, plan):
        from ..._core import BindingKind
        from ..._core.block.radix_rank import block_radix_rank_bins_per_thread
        from .._lowering import _radix_sort

        kwargs = {
            "dtype": self.dtype,
            "threads_per_block": block,
            "items_per_thread": self.items_per_thread,
            "descending": self.descending,
        }
        if self.kind == "radix_rank_keys":
            if self.prefix_extent is not None:
                assert self.digit_bits is not None
                expected = block_radix_rank_bins_per_thread(
                    self.digit_bits, prod(block)
                )
                if self.prefix_extent != expected:
                    raise TypedPipelineError(
                        f"exclusive_digit_prefix must contain {expected} "
                        "items per thread"
                    )
            controls = dict(self.controls)
            begin = controls["begin_bit"]
            end = controls["end_bit"]
            kwargs.update(
                begin_bit=(
                    begin.value if begin.kind is BindingKind.STATIC else begin
                ),
                end_bit=end.value if end.kind is BindingKind.STATIC else None,
                radix_bits=self.fixed_radix_bits,
                with_exclusive_digit_prefix=self.prefix_extent is not None,
            )
        else:
            kwargs["blocked_to_striped"] = self.blocked_to_striped
            if self.value_dtype is not None:
                kwargs["value_dtype"] = self.value_dtype
        return getattr(_radix_sort, self.kind), kwargs


def bind_controls(analysis, stmt, dtype):
    return tuple(
        (name, _sorting.integer_binding(analysis, stmt, "radix", name))
        for name in ("begin_bit", "end_bit")
    )


def request_for(kind, args, kws, *, qualified, extents=None, controls=()):
    from ..._core import (
        ArgumentBinding,
        BindingKind,
        GroupOperandKind,
        make_group_primitive_call,
    )
    from ..._core.api.radix_sort import _radix_bounds
    from ..._core.block.radix import make_radix_bit_range
    from ..._core.block.radix_rank import make_block_radix_rank_semantics
    from ..._core.block.radix_sort import make_block_radix_sort_semantics
    from ..._core.group.radix_sort import (
        GroupRadixRankSemantics,
        GroupRadixSortSemantics,
    )
    from .._lowering._core import NumbaMlirCoreAdapter

    payload = _payload(kind, args, qualified, extents)
    if payload is None:
        return None
    dtypes, items, is_array = payload
    if items is None:
        raise TypedPipelineError(f"coop.{kind}: payload extent must be static")
    key_dtype = dtypes[0]
    value_dtype = dtypes[1] if len(dtypes) > 1 else None
    descending = _bool(kws, "descending")
    rank = kind == "radix_rank_keys"
    bounds = dict(controls)
    begin, end = bounds["begin_bit"], bounds["end_bit"]
    if begin.kind is BindingKind.OMITTED:
        begin = ArgumentBinding.static(0)
    adapter = NumbaMlirCoreAdapter()
    core_kwargs = {
        "key_dtype": adapter.core_dtype(key_dtype),
        "items_per_thread": items,
        "descending": descending,
        "key_bit_width": key_dtype.bitwidth,
    }
    operand_kind = (
        GroupOperandKind.ARRAY if is_array else GroupOperandKind.SCALAR
    )
    prefix_extent = digit_bits = fixed_radix_bits = None
    striped = False
    if rank:
        radix_bits = _callbacks.constant(kws.get("radix_bits"))
        if end.kind is BindingKind.RUNTIME:
            raise TypedPipelineError(
                "radix_rank_keys end_bit must be a compile-time integer"
            )
        end_value = end.value if end.kind is BindingKind.STATIC else None
        if begin.kind is BindingKind.STATIC:
            begin_value, end_value = _radix_bounds(
                kind, key_dtype.bitwidth, begin.value, end_value, radix_bits
            )
            begin, end = (
                ArgumentBinding.static(begin_value),
                ArgumentBinding.static(end_value),
            )
            digit_bits = end_value - begin_value
        else:
            if radix_bits is None:
                raise TypedPipelineError(
                    "radix_rank_keys begin_bit must be compile-time "
                    "when radix_bits is omitted"
                )
            _, digit_bits = _radix_bounds(
                kind, key_dtype.bitwidth, 0, None, radix_bits
            )
            fixed_radix_bits = digit_bits
        prefix = kws.get("exclusive_digit_prefix")
        if not _templates._omitted(prefix):
            if (
                not isinstance(prefix, nb.types.Array)
                or prefix.ndim != 1
                or prefix.layout != "C"
            ):
                raise TypedPipelineError(
                    "exclusive_digit_prefix must be a contiguous "
                    "one-dimensional array"
                )
            if prefix.dtype != nb.types.int32:
                raise TypedPipelineError(
                    "exclusive_digit_prefix must have int32 dtype"
                )
            prefix_extent = (
                prefix.items
                if isinstance(prefix, ThreadDataType)
                else (extents or {}).get("exclusive_digit_prefix")
            )
            if prefix_extent is None:
                raise TypedPipelineError(
                    "exclusive_digit_prefix requires a static extent"
                )
        primitive = make_block_radix_rank_semantics(
            **core_kwargs,
            begin_bit=begin,
            end_bit=end_value,
            radix_bits=fixed_radix_bits,
            block_threads=None,
            exclusive_digit_prefix_items_per_thread=prefix_extent,
        )
        semantics = GroupRadixRankSemantics(
            primitive, operand_kind=operand_kind
        )
    else:
        if end.kind is BindingKind.OMITTED:
            end = ArgumentBinding.static(key_dtype.bitwidth)
        make_radix_bit_range(
            begin_bit=begin, end_bit=end, bit_width=key_dtype.bitwidth
        )
        striped = _bool(kws, "blocked_to_striped")
        primitive = make_block_radix_sort_semantics(
            **core_kwargs,
            value_dtype=adapter.core_dtype(value_dtype),
            begin_bit=begin,
            end_bit=end,
            blocked_to_striped=striped,
            bit_policy="both",
        )
        semantics = GroupRadixSortSemantics(
            primitive, operand_kind=operand_kind
        )
    return RadixRequest(
        kind,
        key_dtype,
        value_dtype,
        items,
        is_array,
        args[0],
        make_group_primitive_call(args[0].group, semantics),
        (("begin_bit", begin), ("end_bit", end)),
        descending,
        striped,
        prefix_extent,
        digit_bits,
        fixed_radix_bits,
        tuple(_templates.storage_options(kws.get("temp_storage")).items()),
    )


def _typing(kind, qualified):
    def generic(self, args, kws):
        payload = _payload(kind, args, qualified)
        if payload is None:
            return None
        dtypes, _, _ = payload
        if kind == "radix_rank_keys":
            dtypes = (nb.types.int32,)
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
    prefix = dict(stmt.value.kws).get("exclusive_digit_prefix")
    if prefix is not None and not _templates._omitted(
        kws["exclusive_digit_prefix"]
    ):
        extents["exclusive_digit_prefix"] = analysis._value_extent(prefix, kind)
    return request_for(
        kind,
        args,
        kws,
        qualified=qualified,
        extents=extents,
        controls=bind_controls(analysis, stmt, None),
    )


def register():
    from ._inference import (
        DtypeSource,
        OperationInference,
        operation_inference_contracts,
    )

    for operation, (kind, _) in OPERATIONS.items():
        if kind == "radix_rank_keys":
            operation_inference_contracts[operation] = OperationInference(
                checked_producers=(
                    (
                        "exclusive_digit_prefix",
                        DtypeSource(1, lambda ty: nb.types.int32),
                    ),
                )
            )
    _sorting.register(OPERATIONS, _typing)
