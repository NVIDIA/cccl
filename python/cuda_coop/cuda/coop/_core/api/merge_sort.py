# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Portable merge-sort entry points for keys and pairs.

These frontends enforce the common payload, dtype, and out-of-bounds contract
before backend delegation. Sort semantics, CUB selection, and compiler-owned
result allocation remain in lower layers.
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
    _validate_common_merge_sort_oob_default,
    _validate_common_pair_payloads,
    _validate_common_thread_data_payload,
)


def merge_sort_keys(
    group: ThreadGroup,
    keys: Any,
    /,
    *,
    descending: bool = False,
    valid_items: Any = None,
    oob_default: Any = None,
    temp_storage: Any = None,
) -> Any:
    """Return merge-sorted keys through the compiler-selected backend.

    Complete physical blocks and warps accept fixed-size integral ``ThreadData``
    and return a shape-preserving payload without mutating ``keys``. A block
    must contain a power-of-two number of threads. For a partial tile, provide
    ``valid_items`` and ``oob_default`` together. The sentinel must have the
    matching key dtype, or be a plain Python integer representable in that
    dtype, and sort after every valid key: greater for ascending order and less
    for descending order. Only the valid sorted prefix is defined.

    Use the qualified ``cuda.coop.<backend>`` API for custom comparators,
    backend-specific payloads, or other backend-specific behavior.
    """

    if _backend_module_name() is not None:
        _validate_portable_operation_group("merge_sort_keys", group)
        _validate_common_thread_data_payload("merge_sort_keys", "keys", keys)
        _validate_common_integer_key_dtype("merge_sort_keys", keys)
        if (valid_items is None) != (oob_default is None):
            raise ValueError(
                "cuda.coop.merge_sort_keys valid_items and oob_default must be "
                "provided together"
            )
        if oob_default is not None:
            _validate_common_merge_sort_oob_default(
                "merge_sort_keys", keys, oob_default
            )

    return _group_primitive_marker(
        "merge_sort_keys",
        group,
        keys,
        descending=descending,
        valid_items=valid_items,
        oob_default=oob_default,
        temp_storage=temp_storage,
    )


def merge_sort_pairs(
    group: ThreadGroup,
    keys: Any,
    values: Any,
    /,
    *,
    descending: bool = False,
    valid_items: Any = None,
    oob_default: Any = None,
    temp_storage: Any = None,
) -> tuple[Any, Any]:
    """Return merge-sorted pairs through the compiler-selected backend.

    Complete physical blocks and warps accept matching fixed-size ``ThreadData``
    payloads. Keys use the portable integral-key profile and values use the
    portable numeric profile. Values remain attached to their keys, equal-key
    ordering is unspecified, and neither input is mutated. For a partial tile,
    provide ``valid_items`` and ``oob_default`` together. The sentinel must
    have the matching key dtype or be a plain Python integer representable in
    that dtype; only the valid sorted prefix is defined.

    Use the qualified ``cuda.coop.<backend>`` API for custom comparators,
    scalar/register payloads, or backend-specific behavior.
    """

    if _backend_module_name() is not None:
        _validate_portable_operation_group("merge_sort_pairs", group)
        _validate_common_pair_payloads("merge_sort_pairs", keys, values)
        if (valid_items is None) != (oob_default is None):
            raise ValueError(
                "cuda.coop.merge_sort_pairs valid_items and oob_default must "
                "be provided together"
            )
        if oob_default is not None:
            _validate_common_merge_sort_oob_default(
                "merge_sort_pairs", keys, oob_default
            )

    return _group_primitive_marker(
        "merge_sort_pairs",
        group,
        keys,
        values,
        descending=descending,
        valid_items=valid_items,
        oob_default=oob_default,
        temp_storage=temp_storage,
    )


__all__ = ["merge_sort_keys", "merge_sort_pairs"]
