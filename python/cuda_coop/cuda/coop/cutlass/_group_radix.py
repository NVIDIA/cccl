# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""CUTLASS group-first block radix sort and rank entrypoints."""

from __future__ import annotations

import dataclasses
import operator
from typing import Any

from cuda.coop._core import (
    AlgorithmSpec,
    GroupLoweringPlan,
    GroupOperandKind,
    GroupRadixRankSemantics,
    GroupRadixSortSemantics,
    LaunchFacts,
    RuntimeValue,
    make_group_primitive_call,
    plan_group_primitive,
)
from cuda.coop._core.block import (
    make_block_radix_rank_semantics,
    make_block_radix_sort_semantics,
)

from ._internal._thread_data import _coerce_thread_payload
from ._thread_group import ThreadGroup, _resolve_collective_group_from_launch

_SCOPE = __name__.rsplit(".", 1)[0]
_MAX_SAFE_RADIX_RANK_BITS = 8
_WIDE_RANK_SMEM_CONFIG = "cudaSharedMemBankSizeEightByte"


def _normalize_order(descending: Any, *, primitive_name: str) -> bool:
    if not isinstance(descending, bool):
        raise TypeError(f"{_SCOPE}.{primitive_name} descending must be a bool")
    return descending


def _static_radix_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(
            f"{_SCOPE}.radix_rank {name} must be a trace-time static integer"
        )
    try:
        value = operator.index(value)
    except Exception as exc:
        raise TypeError(
            f"{_SCOPE}.radix_rank {name} must be a trace-time static integer"
        ) from exc
    if isinstance(value, bool):
        raise TypeError(
            f"{_SCOPE}.radix_rank {name} must be a trace-time static integer"
        )
    return int(value)


def _resolve_rank_bits(
    *,
    begin_bit: Any,
    end_bit: Any | None,
    radix_bits: Any | None,
) -> tuple[int, int]:
    begin = _static_radix_int(begin_bit, name="begin_bit")
    if begin < 0:
        raise ValueError("begin_bit must be non-negative")

    width = None
    if radix_bits is not None:
        width = _static_radix_int(radix_bits, name="radix_bits")
        if width <= 0:
            raise ValueError("radix_bits must be positive")
    if end_bit is None:
        end = begin + (4 if width is None else width)
    else:
        end = _static_radix_int(end_bit, name="end_bit")
        if width is not None and end != begin + width:
            raise ValueError("radix_bits must match end_bit - begin_bit")
    if end <= begin:
        raise ValueError("end_bit must be greater than begin_bit")
    width = end - begin
    if width > _MAX_SAFE_RADIX_RANK_BITS:
        raise ValueError(
            "radix_rank bit width must be <= 8; wider CUB specializations "
            "are outside the qualified resource-parity contract"
        )
    return begin, end


def _make_group_radix_sort_plan(
    *,
    group: ThreadGroup,
    launch: LaunchFacts,
    key_dtype: Any,
    value_dtype: Any | None,
    items_per_thread: int,
    operand_kind: GroupOperandKind,
    descending: bool,
    key_bit_width: int,
    source: str,
) -> GroupLoweringPlan:
    primitive = make_block_radix_sort_semantics(
        key_dtype=key_dtype,
        value_dtype=value_dtype,
        items_per_thread=items_per_thread,
        descending=descending,
        begin_bit=RuntimeValue("begin_bit"),
        end_bit=RuntimeValue("end_bit"),
        key_bit_width=key_bit_width,
        bit_policy="explicit",
    )
    return plan_group_primitive(
        make_group_primitive_call(
            group,
            GroupRadixSortSemantics(primitive, operand_kind=operand_kind),
            source=source,
        ),
        launch,
    )


def _make_group_radix_rank_plan(
    *,
    group: ThreadGroup,
    launch: LaunchFacts,
    cub_key_dtype: Any,
    input_dtype: Any,
    items_per_thread: int,
    operand_kind: GroupOperandKind,
    begin_bit: int,
    end_bit: int,
    key_bit_width: int,
    descending: bool,
    exclusive_digit_prefix_items_per_thread: int | None,
    source: str,
) -> GroupLoweringPlan:
    assert launch.exact_block_threads is not None
    primitive = make_block_radix_rank_semantics(
        key_dtype=cub_key_dtype,
        items_per_thread=items_per_thread,
        begin_bit=begin_bit,
        end_bit=end_bit,
        key_bit_width=key_bit_width,
        descending=descending,
        block_threads=launch.exact_block_threads,
        exclusive_digit_prefix_items_per_thread=(
            exclusive_digit_prefix_items_per_thread
        ),
    )
    plan = plan_group_primitive(
        make_group_primitive_call(
            group,
            GroupRadixRankSemantics(
                primitive,
                input_dtype=input_dtype,
                operand_kind=operand_kind,
            ),
            source=source,
        ),
        launch,
    )
    if plan.unsupported is not None:
        return plan
    implementation = plan.implementation
    if not isinstance(implementation, AlgorithmSpec):
        raise TypeError("CUTLASS radix rank plans require an AlgorithmSpec")
    radix_bits = implementation.template_arguments["RADIX_BITS"]
    if radix_bits < 8:
        return plan

    # CUB's four-byte packing gives an eight-bit rank a 129-word memoized
    # raking segment. Eight-byte packing halves that live array and keeps wide
    # rank code below the provider register limit without changing other
    # backends' shared-core specialization.
    template_arguments = dict(implementation.template_arguments)
    template_arguments["SMEM_CONFIG"] = _WIDE_RANK_SMEM_CONFIG
    implementation = implementation.algorithm.specialize(
        template_arguments,
        metadata=implementation.metadata,
    )
    return dataclasses.replace(plan, implementation=implementation)


def _validate_group(group: Any, *, primitive_name: str) -> None:
    if not isinstance(group, ThreadGroup):
        raise TypeError(f"{_SCOPE}.{primitive_name} group must be a ThreadGroup")
    if group.kind != "block":
        raise NotImplementedError(
            f"{_SCOPE}.{primitive_name} currently lowers only this_block groups"
        )


def _radix_sort_keys(
    group: ThreadGroup,
    keys: Any,
    /,
    *args: Any,
    begin_bit: Any = 0,
    end_bit: Any | None = None,
    descending: bool = False,
    temp_storage: Any = None,
    source: str = "cutlass_root",
    **kwargs: Any,
) -> Any:
    from ._dsl._launch import infer_launch_facts, pop_launch_metadata
    from ._dsl._scope import validate_no_extra_args

    launch_kwargs = pop_launch_metadata(kwargs)
    validate_no_extra_args(
        _SCOPE,
        "radix_sort_keys",
        args=args,
        kwargs=kwargs,
        expected="expects a ThreadGroup and one positional scalar or ThreadData key",
    )
    _validate_group(group, primitive_name="radix_sort_keys")
    keys = _coerce_thread_payload(
        keys,
        scope=_SCOPE,
        primitive_name="radix_sort_keys",
        arg_name="keys",
        common_root_payload_kind="thread_data",
    )
    descending = _normalize_order(descending, primitive_name="radix_sort_keys")
    launch = infer_launch_facts(
        launch_kwargs,
        scope=_SCOPE,
        primitive_name="radix_sort_keys",
    )
    group = _resolve_collective_group_from_launch(
        group,
        launch,
        feature="radix_sort_keys",
    )
    from ._dsl import _cub_radix_provider as _provider

    return _provider.provider_radix_sort_keys(
        group=group,
        launch=launch,
        keys=keys,
        begin_bit=begin_bit,
        end_bit=end_bit,
        descending=descending,
        source=source,
        temp_storage=temp_storage,
    )


def radix_sort_keys(
    group: ThreadGroup,
    keys: Any,
    /,
    *,
    begin_bit: Any = 0,
    end_bit: Any | None = None,
    descending: bool = False,
    temp_storage: Any = None,
) -> Any:
    """Sort scalar or fixed-size register keys across an explicit block group."""

    return _radix_sort_keys(
        group,
        keys,
        begin_bit=begin_bit,
        end_bit=end_bit,
        descending=descending,
        temp_storage=temp_storage,
        source="cutlass_root",
    )


def _radix_sort_pairs(
    group: ThreadGroup,
    keys: Any,
    values: Any,
    /,
    *args: Any,
    begin_bit: Any = 0,
    end_bit: Any | None = None,
    descending: bool = False,
    temp_storage: Any = None,
    source: str = "cutlass_root",
    **kwargs: Any,
) -> tuple[Any, Any]:
    from ._dsl._launch import infer_launch_facts, pop_launch_metadata
    from ._dsl._scope import validate_no_extra_args

    launch_kwargs = pop_launch_metadata(kwargs)
    validate_no_extra_args(
        _SCOPE,
        "radix_sort_pairs",
        args=args,
        kwargs=kwargs,
        expected=(
            "expects a ThreadGroup and positional scalar or ThreadData key/value "
            "operands"
        ),
    )
    _validate_group(group, primitive_name="radix_sort_pairs")
    keys = _coerce_thread_payload(
        keys,
        scope=_SCOPE,
        primitive_name="radix_sort_pairs",
        arg_name="keys",
        common_root_payload_kind="thread_data",
    )
    values = _coerce_thread_payload(
        values,
        scope=_SCOPE,
        primitive_name="radix_sort_pairs",
        arg_name="values",
        common_root_payload_kind="thread_data",
    )
    descending = _normalize_order(descending, primitive_name="radix_sort_pairs")
    launch = infer_launch_facts(
        launch_kwargs,
        scope=_SCOPE,
        primitive_name="radix_sort_pairs",
    )
    group = _resolve_collective_group_from_launch(
        group,
        launch,
        feature="radix_sort_pairs",
    )
    from ._dsl import _cub_radix_provider as _provider

    return _provider.provider_radix_sort_pairs(
        group=group,
        launch=launch,
        keys=keys,
        values=values,
        begin_bit=begin_bit,
        end_bit=end_bit,
        descending=descending,
        source=source,
        temp_storage=temp_storage,
    )


def radix_sort_pairs(
    group: ThreadGroup,
    keys: Any,
    values: Any,
    /,
    *,
    begin_bit: Any = 0,
    end_bit: Any | None = None,
    descending: bool = False,
    temp_storage: Any = None,
) -> tuple[Any, Any]:
    """Sort key/value pairs across an explicit block group."""

    return _radix_sort_pairs(
        group,
        keys,
        values,
        begin_bit=begin_bit,
        end_bit=end_bit,
        descending=descending,
        temp_storage=temp_storage,
        source="cutlass_root",
    )


def _radix_rank(
    group: ThreadGroup,
    keys: Any,
    /,
    *args: Any,
    begin_bit: Any = 0,
    end_bit: Any | None = None,
    radix_bits: Any | None = None,
    descending: bool = False,
    exclusive_digit_prefix: Any = None,
    source: str = "cutlass_root",
    **kwargs: Any,
) -> Any:
    from ._dsl._launch import infer_launch_facts, pop_launch_metadata
    from ._dsl._scope import validate_no_extra_args

    launch_kwargs = pop_launch_metadata(kwargs)
    validate_no_extra_args(
        _SCOPE,
        "radix_rank",
        args=args,
        kwargs=kwargs,
        expected="expects a ThreadGroup and one positional scalar or ThreadData key",
    )
    _validate_group(group, primitive_name="radix_rank")
    keys = _coerce_thread_payload(
        keys,
        scope=_SCOPE,
        primitive_name="radix_rank",
        arg_name="keys",
        common_root_payload_kind="thread_data",
    )
    descending = _normalize_order(descending, primitive_name="radix_rank")
    begin, end = _resolve_rank_bits(
        begin_bit=begin_bit,
        end_bit=end_bit,
        radix_bits=radix_bits,
    )
    launch = infer_launch_facts(
        launch_kwargs,
        scope=_SCOPE,
        primitive_name="radix_rank",
    )
    group = _resolve_collective_group_from_launch(
        group,
        launch,
        feature="radix_rank",
    )
    from ._dsl import _cub_radix_provider as _provider

    return _provider.provider_radix_rank(
        group=group,
        launch=launch,
        keys=keys,
        begin_bit=begin,
        end_bit=end,
        descending=descending,
        exclusive_digit_prefix=exclusive_digit_prefix,
        source=source,
    )


def radix_rank(
    group: ThreadGroup,
    keys: Any,
    /,
    *,
    begin_bit: Any = 0,
    end_bit: Any | None = None,
    radix_bits: Any | None = None,
    descending: bool = False,
    exclusive_digit_prefix: Any = None,
) -> Any:
    """Rank a trace-static radix digit across an explicit block group."""

    return _radix_rank(
        group,
        keys,
        begin_bit=begin_bit,
        end_bit=end_bit,
        radix_bits=radix_bits,
        descending=descending,
        exclusive_digit_prefix=exclusive_digit_prefix,
        source="cutlass_root",
    )


__all__ = ["radix_rank", "radix_sort_keys", "radix_sort_pairs"]
