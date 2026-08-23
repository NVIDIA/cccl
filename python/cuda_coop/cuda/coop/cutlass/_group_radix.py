# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""CUTLASS group-first block radix sort and rank entrypoints."""

from __future__ import annotations

import operator
from typing import Any

from cuda.coop._core import (
    LaunchFacts,
)

from ._thread_data import _coerce_thread_payload
from ._thread_group import ThreadGroup, _resolve_collective_group_from_launch

_SCOPE = __name__.rsplit(".", 1)[0]
_MAX_SAFE_RADIX_RANK_BITS = 8
_WIDE_RANK_SMEM_CONFIG = "cudaSharedMemBankSizeEightByte"


def _is_boolean_scalar(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    return any(
        (value_type.__module__ or "").split(".", 1)[0] == "numpy"
        and value_type.__name__ in {"bool", "bool_"}
        for value_type in type(value).__mro__
    )


def _normalize_order(descending: Any, *, primitive_name: str) -> bool:
    if not isinstance(descending, bool):
        raise TypeError(f"{_SCOPE}.{primitive_name} descending must be a bool")
    return descending


def _static_radix_int(value: Any, *, name: str) -> int:
    if _is_boolean_scalar(value):
        raise TypeError(
            f"{_SCOPE}.radix_rank {name} must be a trace-time static integer"
        )
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise TypeError(
            f"{_SCOPE}.radix_rank {name} must be a trace-time static integer"
        ) from exc
    if _is_boolean_scalar(value):
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


def _validate_group(group: Any, *, primitive_name: str) -> None:
    if not isinstance(group, ThreadGroup):
        raise TypeError(f"{_SCOPE}.{primitive_name} group must be a ThreadGroup")
    if group.kind != "block":
        raise NotImplementedError(
            f"{_SCOPE}.{primitive_name} currently lowers only this_block groups"
        )


def _resolve_block_launch(
    group: ThreadGroup,
    *,
    primitive_name: str,
) -> tuple[LaunchFacts, ThreadGroup]:
    from ._compiler._launch import infer_launch_facts

    launch = infer_launch_facts({}, scope=_SCOPE, primitive_name=primitive_name)
    if not launch.is_verified("exact_block_dim"):
        raise NotImplementedError(
            f"{_SCOPE}.{primitive_name} requires exact block dimensions from "
            "verified compiler launch facts"
        )
    return launch, _resolve_collective_group_from_launch(
        group,
        launch,
        feature=primitive_name,
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
    """Radix-sort scalar or register keys across one complete CUDA block.

    Signed and unsigned 32- or 64-bit keys are accepted. ``begin_bit`` and
    ``end_bit`` select a runtime half-open interval in CUB's bit-ordered key
    representation; omitting ``end_bit`` selects the key width. The result is
    a fresh payload with the same shape and type as ``keys``.
    """

    _validate_group(group, primitive_name="radix_sort_keys")
    keys = _coerce_thread_payload(
        keys,
        scope=_SCOPE,
        primitive_name="radix_sort_keys",
        arg_name="keys",
        common_root_payload_kind="thread_data",
    )
    descending = _normalize_order(descending, primitive_name="radix_sort_keys")
    launch, group = _resolve_block_launch(group, primitive_name="radix_sort_keys")

    from ._lowering import _radix as _provider

    return _provider.provider_radix_sort_keys(
        group=group,
        launch=launch,
        keys=keys,
        begin_bit=begin_bit,
        end_bit=end_bit,
        descending=descending,
        source="cutlass_root",
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
    """Radix-sort associated key/value payloads across one complete block.

    Keys use the integral radix profile while values may use any supported
    numeric CUTLASS payload type. The returned key and value payloads are new;
    neither input is mutated and key/value association is preserved.
    """

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
    launch, group = _resolve_block_launch(group, primitive_name="radix_sort_pairs")

    from ._lowering import _radix as _provider

    return _provider.provider_radix_sort_pairs(
        group=group,
        launch=launch,
        keys=keys,
        values=values,
        begin_bit=begin_bit,
        end_bit=end_bit,
        descending=descending,
        source="cutlass_root",
        temp_storage=temp_storage,
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
    """Rank one trace-static radix digit across a complete CUDA block.

    The selected interval defaults to four bits and may contain at most eight
    bits. Ranks are returned in a fresh signed 32-bit payload. The optional
    ``exclusive_digit_prefix`` output receives the per-digit prefixes that CUB
    distributes across block threads.
    """

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
    launch, group = _resolve_block_launch(group, primitive_name="radix_rank")

    from ._lowering import _radix as _provider

    return _provider.provider_radix_rank(
        group=group,
        launch=launch,
        keys=keys,
        begin_bit=begin,
        end_bit=end,
        descending=descending,
        exclusive_digit_prefix=exclusive_digit_prefix,
        source="cutlass_root",
    )


__all__ = ["radix_rank", "radix_sort_keys", "radix_sort_pairs"]
