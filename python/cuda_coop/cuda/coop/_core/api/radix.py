# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Portable radix-sort and radix-rank entry points.

The related operations share integer-key, pair-payload, and bit-control
validation here before backend delegation. The semantic planner owns CUB tile
constraints and specialization; compiler lifecycle remains backend-specific.
"""

from __future__ import annotations

from typing import Any

from ..thread_group import ThreadGroup
from ._dispatch import (
    _backend_module_name,
    _group_primitive_marker,
    _validate_portable_operation_group,
)
from ._payload import (
    _validate_common_integer_key_dtype,
    _validate_common_pair_payloads,
    _validate_common_radix_rank_controls,
    _validate_common_radix_sort_controls,
    _validate_common_thread_data_payload,
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
    """Return radix-sorted integral keys through the compiler-selected backend.

    ``group`` must be a complete physical block and ``keys`` must be a
    fixed-size ``ThreadData`` payload of Python, NumPy, or compiler 32- or
    64-bit signed or unsigned integers. ``begin_bit`` and ``end_bit`` select a
    half-open interval in CUB's bit-ordered key representation. Omitting
    ``end_bit`` selects the key width, including when ``begin_bit`` is nonzero.
    The returned payload preserves the input item type and item count without
    mutating ``keys``.

    Use the qualified ``cuda.coop.<backend>`` API for scalar/register payloads,
    striped output, or other backend-specific behavior.
    """

    if _backend_module_name() is not None:
        _validate_portable_operation_group("radix_sort_keys", group)
        _validate_common_thread_data_payload("radix_sort_keys", "keys", keys)
        key_width = _validate_common_integer_key_dtype("radix_sort_keys", keys)
        _validate_common_radix_sort_controls(
            "radix_sort_keys",
            key_width=key_width,
            begin_bit=begin_bit,
            end_bit=end_bit,
            descending=descending,
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
    """Return radix-sorted pairs through the compiler-selected backend.

    ``group`` is a complete physical block. Matching ``ThreadData`` payloads
    preserve both item types and key/value association without mutation.
    ``begin_bit`` and ``end_bit`` select the same portable bit interval as
    :func:`radix_sort_keys`.

    Use the qualified ``cuda.coop.<backend>`` API for scalar/register payloads,
    striped output, launch controls, or other backend-specific behavior.
    """

    if _backend_module_name() is not None:
        _validate_portable_operation_group("radix_sort_pairs", group)
        key_width, _ = _validate_common_pair_payloads("radix_sort_pairs", keys, values)
        _validate_common_radix_sort_controls(
            "radix_sort_pairs",
            key_width=key_width,
            begin_bit=begin_bit,
            end_bit=end_bit,
            descending=descending,
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


def radix_rank(
    group: ThreadGroup,
    keys: Any,
    /,
    *,
    begin_bit: Any = 0,
    end_bit: Any | None = None,
    radix_bits: Any | None = None,
    descending: bool = False,
) -> Any:
    """Return stable ranks through the compiler-selected backend.

    ``group`` must be a complete physical block and ``keys`` must be a
    fixed-size ``ThreadData`` payload of Python, NumPy, or compiler 32- or
    64-bit signed or unsigned integers. The digit is extracted from CUB's
    bit-ordered key representation, so signed keys are ordered by value rather
    than by their raw two's-complement sign bit. The selected half-open
    interval defaults to four bits starting at ``begin_bit`` and may contain at
    most eight bits. Equal digits retain flattened blocked input order. The
    returned payload contains shape-preserving signed 32-bit ranks and does not
    mutate ``keys``.

    Use the qualified ``cuda.coop.<backend>`` API for scalar/register payloads
    or an exclusive digit-prefix side output.
    """

    if _backend_module_name() is not None:
        _validate_portable_operation_group("radix_rank", group)
        _validate_common_thread_data_payload("radix_rank", "keys", keys)
        key_width = _validate_common_integer_key_dtype("radix_rank", keys)
        _validate_common_radix_rank_controls(
            key_width=key_width,
            begin_bit=begin_bit,
            end_bit=end_bit,
            radix_bits=radix_bits,
            descending=descending,
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
