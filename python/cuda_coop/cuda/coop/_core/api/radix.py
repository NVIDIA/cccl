# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Portable block radix ordering operations."""

from __future__ import annotations

from numbers import Integral
from typing import Any

from .._bindings import ArgumentBinding
from ..block.radix import make_radix_bit_range
from ..thread_group import ThreadGroup
from ._dispatch import (
    _backend_module_name,
    _group_primitive_marker,
    _portable_group_operation,
    _validate_portable_operation_group,
)
from ._payload import (
    _common_thread_data_extent,
    _validate_common_integer_value,
    _validate_common_numeric_value,
    _validate_common_temp_storage,
)


def _radix_bounds(operation, key_width, begin_bit, end_bit, radix_bits=None):
    for name, value in (
        ("begin_bit", begin_bit),
        ("end_bit", end_bit),
        ("radix_bits", radix_bits),
    ):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, Integral)
        ):
            raise TypeError(
                f"cuda.coop.{operation} {name} must be a compile-time integer"
            )
    if radix_bits is not None and radix_bits <= 0:
        raise ValueError("radix_bits must be positive")
    if end_bit is None:
        end_bit = (
            begin_bit + (4 if radix_bits is None else radix_bits)
            if operation == "radix_rank"
            else key_width
        )
    if radix_bits is not None and end_bit - begin_bit != radix_bits:
        raise ValueError("radix_bits must match end_bit - begin_bit")
    make_radix_bit_range(begin_bit=begin_bit, end_bit=end_bit, bit_width=key_width)
    if operation == "radix_rank" and end_bit - begin_bit > 8:
        raise ValueError("radix_rank bit width must be <= 8")
    return int(begin_bit), int(end_bit)


def _validate(
    operation,
    group,
    keys,
    values,
    begin_bit,
    end_bit,
    descending,
    temp_storage,
    radix_bits=None,
):
    if _backend_module_name() is None:
        return
    _validate_portable_operation_group(operation, group)
    name = _validate_common_numeric_value(
        operation,
        "keys",
        keys,
        require_thread_data=True,
        allow_readonly_thread_data=True,
    )
    if name not in {"int32", "uint32", "int64", "uint64"}:
        raise TypeError(
            f"cuda.coop.{operation} keys require int32, uint32, int64, or uint64"
        )
    if operation == "radix_sort_pairs":
        _validate_common_numeric_value(
            operation,
            "values",
            values,
            require_thread_data=True,
            allow_readonly_thread_data=True,
        )
        if _common_thread_data_extent(
            operation, "keys", keys
        ) != _common_thread_data_extent(operation, "values", values):
            raise ValueError("keys and values must have the same items_per_thread")
    if not isinstance(descending, bool):
        raise TypeError(f"cuda.coop.{operation} descending must be a compile-time bool")
    width = int(name[-2:])
    if operation == "radix_rank":
        _radix_bounds(operation, width, begin_bit, end_bit, radix_bits)
    else:
        begin = _validate_common_integer_value(operation, "begin_bit", begin_bit)
        end = (
            width
            if end_bit is None
            else _validate_common_integer_value(operation, "end_bit", end_bit)
        )
        make_radix_bit_range(
            begin_bit=ArgumentBinding.runtime() if begin is None else begin,
            end_bit=ArgumentBinding.runtime() if end is None else end,
            bit_width=width,
        )
    if temp_storage is not None:
        _validate_common_temp_storage(operation, temp_storage)


@_portable_group_operation("radix_sort_keys", group_kinds=("block",))
def radix_sort_keys(
    group: ThreadGroup,
    keys: Any,
    /,
    *,
    begin_bit: int = 0,
    end_bit: int | None = None,
    descending: bool = False,
    temp_storage: Any = None,
) -> Any:
    """Return stable, blocked radix-sorted integral keys without mutation.

    The complete block participates. Keys are fixed-size ThreadData with
    32- or 64-bit signed or unsigned integer elements. The half-open bit
    interval selects CUB's ordered representation, including the inverted
    sign bit for signed integers; omitted end_bit selects the key width.
    """
    _validate(
        "radix_sort_keys",
        group,
        keys,
        None,
        begin_bit,
        end_bit,
        descending,
        temp_storage,
    )
    return _group_primitive_marker(
        "radix_sort_keys",
        group,
        keys,
        begin_bit=begin_bit,
        end_bit=end_bit,
        descending=descending,
        temp_storage=temp_storage,
    )


@_portable_group_operation("radix_sort_pairs", group_kinds=("block",))
def radix_sort_pairs(
    group: ThreadGroup,
    keys: Any,
    values: Any,
    /,
    *,
    begin_bit: int = 0,
    end_bit: int | None = None,
    descending: bool = False,
    temp_storage: Any = None,
) -> tuple[Any, Any]:
    """Return stable sorted keys and associated numeric values without mutation.

    Both ThreadData payloads must have the same extent. Equal selected key
    digits retain flattened blocked input order, including descending sorts.
    """
    _validate(
        "radix_sort_pairs",
        group,
        keys,
        values,
        begin_bit,
        end_bit,
        descending,
        temp_storage,
    )
    return _group_primitive_marker(
        "radix_sort_pairs",
        group,
        keys,
        values,
        begin_bit=begin_bit,
        end_bit=end_bit,
        descending=descending,
        temp_storage=temp_storage,
    )


@_portable_group_operation("radix_rank", group_kinds=("block",))
def radix_rank(
    group: ThreadGroup,
    keys: Any,
    /,
    *,
    begin_bit: int = 0,
    end_bit: int | None = None,
    radix_bits: int | None = None,
    descending: bool = False,
) -> Any:
    """Return shape-preserving int32 ranks without mutating integral keys.

    Rank the selected digit stably across a complete block. The interval
    defaults to four bits starting at begin_bit and may contain at most eight
    bits. Signed keys use the same sign-bit transformation as radix sort.
    """
    _validate(
        "radix_rank",
        group,
        keys,
        None,
        begin_bit,
        end_bit,
        descending,
        None,
        radix_bits,
    )
    return _group_primitive_marker(
        "radix_rank",
        group,
        keys,
        begin_bit=begin_bit,
        end_bit=end_bit,
        radix_bits=radix_bits,
        descending=descending,
    )


__all__ = ["radix_rank", "radix_sort_keys", "radix_sort_pairs"]
