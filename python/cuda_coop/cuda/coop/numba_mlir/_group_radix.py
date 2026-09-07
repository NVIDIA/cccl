# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Group-first radix rank and sort markers for Numba-CUDA-MLIR.

This module owns radix API signatures.  Trace-static bit controls and launch
dimensions are resolved by compiler planning before provider construction.
"""

from __future__ import annotations

from typing import Any

from ._compiler._operations import group_operation
from ._group_marker import group_primitive_marker
from ._thread_group import ThreadGroup


@group_operation("radix_sort_keys")
def radix_sort_keys(
    group: ThreadGroup,
    keys: Any,
    /,
    *,
    begin_bit: Any = 0,
    end_bit: Any | None = None,
    descending: bool = False,
    temp_storage: Any = None,
    blocked_to_striped: bool = False,
) -> Any:
    """Return radix-sorted keys without mutating the input payload."""

    return group_primitive_marker(
        "radix_sort_keys",
        group,
        keys,
        begin_bit=begin_bit,
        end_bit=end_bit,
        descending=descending,
        temp_storage=temp_storage,
        blocked_to_striped=blocked_to_striped,
    )


@group_operation("radix_sort_pairs")
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
    blocked_to_striped: bool = False,
) -> tuple[Any, Any]:
    """Return radix-sorted key/value payloads without mutating inputs."""

    return group_primitive_marker(
        "radix_sort_pairs",
        group,
        keys,
        values,
        begin_bit=begin_bit,
        end_bit=end_bit,
        descending=descending,
        temp_storage=temp_storage,
        blocked_to_striped=blocked_to_striped,
    )


@group_operation("radix_rank")
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
    """Return block-wide ranks for one trace-static radix digit."""

    return group_primitive_marker(
        "radix_rank",
        group,
        keys,
        begin_bit=begin_bit,
        end_bit=end_bit,
        radix_bits=radix_bits,
        descending=descending,
        exclusive_digit_prefix=exclusive_digit_prefix,
    )


__all__ = ["radix_rank", "radix_sort_keys", "radix_sort_pairs"]
