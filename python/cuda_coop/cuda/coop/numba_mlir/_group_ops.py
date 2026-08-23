# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compile-time group-first primitive markers for Numba-CUDA-MLIR."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cuda.coop._core.block import BlockShuffleMode

from ._thread_group import ThreadGroup

if TYPE_CHECKING:
    from . import ThreadData

_ROOT_SCOPE = __name__.rsplit(".", 1)[0]


def _group_primitive_marker(operation: str, *args: Any, **kwargs: Any) -> Any:
    del args, kwargs
    raise RuntimeError(
        f"{_ROOT_SCOPE}.{operation} is a compile-time kernel construct and "
        "must be lowered by the whole-function planner"
    )


def load(
    group: ThreadGroup,
    source: Any,
    output: ThreadData,
    /,
    *,
    algorithm: Any = "direct",
    valid_items: Any = None,
    oob_default: Any = None,
    offset: Any = None,
    temp_storage: Any = None,
) -> ThreadData:
    """Load a per-thread tile through a block or warp group."""

    return _group_primitive_marker(
        "load",
        group,
        source,
        output,
        algorithm=algorithm,
        valid_items=valid_items,
        oob_default=oob_default,
        offset=offset,
        temp_storage=temp_storage,
    )


def store(
    group: ThreadGroup,
    destination: Any,
    value: Any,
    /,
    *,
    algorithm: Any = "direct",
    valid_items: Any = None,
    offset: Any = None,
    temp_storage: Any = None,
) -> None:
    """Store a per-thread tile through a block or warp group."""

    _group_primitive_marker(
        "store",
        group,
        destination,
        value,
        algorithm=algorithm,
        valid_items=valid_items,
        offset=offset,
        temp_storage=temp_storage,
    )


def reduce(
    group: ThreadGroup,
    value: Any,
    /,
    *,
    binary_op: Any = None,
    broadcast: bool = True,
    valid_items: Any = None,
    algorithm: Any = None,
) -> Any:
    """Reduce values across a group."""

    return _group_primitive_marker(
        "reduce",
        group,
        value,
        binary_op=binary_op,
        broadcast=broadcast,
        valid_items=valid_items,
        algorithm=algorithm,
    )


def sum(
    group: ThreadGroup,
    value: Any,
    /,
    *,
    broadcast: bool = True,
    valid_items: Any = None,
    algorithm: Any = None,
) -> Any:
    """Sum values across a group."""

    return _group_primitive_marker(
        "sum",
        group,
        value,
        broadcast=broadcast,
        valid_items=valid_items,
        algorithm=algorithm,
    )


def scan(
    group: ThreadGroup,
    value: Any,
    /,
    *,
    mode: str = "exclusive",
    scan_op: Any = None,
    initial_value: Any = None,
    algorithm: Any = None,
    temp_storage: Any = None,
    valid_items: Any = None,
    aggregate_output: Any = None,
) -> Any:
    """Scan values across a block or warp group."""

    return _group_primitive_marker(
        "scan",
        group,
        value,
        mode=mode,
        scan_op=scan_op,
        initial_value=initial_value,
        algorithm=algorithm,
        temp_storage=temp_storage,
        valid_items=valid_items,
        aggregate_output=aggregate_output,
    )


def exclusive_sum(
    group: ThreadGroup,
    value: Any,
    /,
    *,
    algorithm: Any = None,
    temp_storage: Any = None,
    valid_items: Any = None,
    aggregate_output: Any = None,
) -> Any:
    """Return an exclusive prefix sum across a block or warp group."""

    return _group_primitive_marker(
        "exclusive_sum",
        group,
        value,
        algorithm=algorithm,
        temp_storage=temp_storage,
        valid_items=valid_items,
        aggregate_output=aggregate_output,
    )


def inclusive_sum(
    group: ThreadGroup,
    value: Any,
    /,
    *,
    algorithm: Any = None,
    temp_storage: Any = None,
    valid_items: Any = None,
    aggregate_output: Any = None,
) -> Any:
    """Return an inclusive prefix sum across a block or warp group."""

    return _group_primitive_marker(
        "inclusive_sum",
        group,
        value,
        algorithm=algorithm,
        temp_storage=temp_storage,
        valid_items=valid_items,
        aggregate_output=aggregate_output,
    )


def exclusive_scan(
    group: ThreadGroup,
    value: Any,
    /,
    *,
    scan_op: Any = None,
    initial_value: Any = None,
    algorithm: Any = None,
    temp_storage: Any = None,
    valid_items: Any = None,
    aggregate_output: Any = None,
) -> Any:
    """Return an exclusive scan across a block or warp group."""

    return _group_primitive_marker(
        "exclusive_scan",
        group,
        value,
        scan_op=scan_op,
        initial_value=initial_value,
        algorithm=algorithm,
        temp_storage=temp_storage,
        valid_items=valid_items,
        aggregate_output=aggregate_output,
    )


def inclusive_scan(
    group: ThreadGroup,
    value: Any,
    /,
    *,
    scan_op: Any = None,
    algorithm: Any = None,
    temp_storage: Any = None,
    valid_items: Any = None,
    aggregate_output: Any = None,
) -> Any:
    """Return an inclusive scan across a block or warp group."""

    return _group_primitive_marker(
        "inclusive_scan",
        group,
        value,
        scan_op=scan_op,
        algorithm=algorithm,
        temp_storage=temp_storage,
        valid_items=valid_items,
        aggregate_output=aggregate_output,
    )


def exchange(
    group: ThreadGroup,
    value: Any,
    /,
    *,
    mode: str = "striped_to_blocked",
    ranks: Any = None,
    valid_flags: Any = None,
    warp_time_slicing: bool = False,
) -> Any:
    """Rearrange a fixed-size per-thread tile within a group."""

    return _group_primitive_marker(
        "exchange",
        group,
        value,
        mode=mode,
        ranks=ranks,
        valid_flags=valid_flags,
        warp_time_slicing=warp_time_slicing,
    )


def shuffle(
    group: ThreadGroup,
    value: Any,
    /,
    *,
    mode: Any = BlockShuffleMode.DOWN,
    distance: Any = 1,
    block_prefix: Any = None,
    block_suffix: Any = None,
) -> Any:
    """Shuffle scalar values or fixed-size per-thread tiles within a block."""

    return _group_primitive_marker(
        "shuffle",
        group,
        value,
        mode=mode,
        distance=distance,
        block_prefix=block_prefix,
        block_suffix=block_suffix,
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
    compare_op: Any = None,
) -> Any:
    """Merge-sort keys across a block or warp group."""

    return _group_primitive_marker(
        "merge_sort_keys",
        group,
        keys,
        descending=descending,
        valid_items=valid_items,
        oob_default=oob_default,
        temp_storage=temp_storage,
        compare_op=compare_op,
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
    compare_op: Any = None,
) -> tuple[Any, Any]:
    """Merge-sort keys and associated values across a block or warp group."""

    return _group_primitive_marker(
        "merge_sort_pairs",
        group,
        keys,
        values,
        descending=descending,
        valid_items=valid_items,
        oob_default=oob_default,
        temp_storage=temp_storage,
        compare_op=compare_op,
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
    blocked_to_striped: bool = False,
) -> Any:
    """Return radix-sorted keys without mutating the input payload."""

    return _group_primitive_marker(
        "radix_sort_keys",
        group,
        keys,
        begin_bit=begin_bit,
        end_bit=end_bit,
        descending=descending,
        temp_storage=temp_storage,
        blocked_to_striped=blocked_to_striped,
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
    blocked_to_striped: bool = False,
) -> tuple[Any, Any]:
    """Return radix-sorted key/value payloads without mutating inputs."""

    return _group_primitive_marker(
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

    return _group_primitive_marker(
        "radix_rank",
        group,
        keys,
        begin_bit=begin_bit,
        end_bit=end_bit,
        radix_bits=radix_bits,
        descending=descending,
        exclusive_digit_prefix=exclusive_digit_prefix,
    )


__all__ = [
    "exchange",
    "exclusive_scan",
    "exclusive_sum",
    "inclusive_scan",
    "inclusive_sum",
    "load",
    "merge_sort_keys",
    "merge_sort_pairs",
    "radix_rank",
    "radix_sort_keys",
    "radix_sort_pairs",
    "reduce",
    "scan",
    "shuffle",
    "store",
    "sum",
]
