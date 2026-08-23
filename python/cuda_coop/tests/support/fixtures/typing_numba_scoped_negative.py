# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Strict negative typing fixture for Numba scoped primitive boundaries."""

# pyright: strict, reportPrivateUsage=none, reportUnnecessaryTypeIgnoreComment=error

from typing import TYPE_CHECKING


def _add(left: int, right: int) -> int:
    return left + right


def _less(left: int, right: int) -> bool:
    return left < right


if TYPE_CHECKING:
    import cuda.coop.numba_mlir as coop

    class _PrefixOp:
        pass

    coop.StatefulFunction(_PrefixOp)  # pyright: ignore[reportCallIssue]
    invalid_stateful_op = coop.StatefulFunction(
        _PrefixOp,
        object(),
        name=1,  # pyright: ignore[reportArgumentType]
    )
    invalid_stateful_op.fn  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]

    values = coop.ThreadData(2, int)
    output = coop.ThreadData(2, int)

    # Scoped Load/Store retain ``num_valid_items`` rather than the normalized
    # group-first ``valid_items`` spelling.
    coop._block.make_load(int, 128, valid_items=64)  # pyright: ignore[reportCallIssue]
    coop._block.make_store(int, 128, valid_items=64)  # pyright: ignore[reportCallIssue]
    coop._warp.make_load(int, valid_items=16)  # pyright: ignore[reportCallIssue]
    coop._warp.make_store(int, valid_items=16)  # pyright: ignore[reportCallIssue]

    # Block reduction and TopK retain the established ``num_valid`` spelling;
    # warp reduction continues to use ``valid_items``.
    coop._block.make_reduce(int, 128, _add, valid_items=64)  # pyright: ignore[reportCallIssue]
    coop._block.make_sum(int, 128, valid_items=64)  # pyright: ignore[reportCallIssue]
    coop._block.make_topk_max_keys(int, 128, valid_items=64)  # pyright: ignore[reportCallIssue]
    coop._warp.make_reduce(int, _add, num_valid=16)  # pyright: ignore[reportCallIssue]
    coop._warp.make_sum(int, num_valid=16)  # pyright: ignore[reportCallIssue]

    # Factory-only controls stay family-specific; unrelated normalized or
    # backend-routing keywords are rejected instead of disappearing into Any.
    coop._block.make_scan(int, 128, valid_items=64)  # pyright: ignore[reportCallIssue]
    coop._block.make_exchange(int, threads_per_block=128, ranks=output)  # pyright: ignore[reportCallIssue]
    coop._block.make_adjacent_difference(int, threads_per_block=128, descending=True)  # pyright: ignore[reportCallIssue]
    coop._block.make_discontinuity(int, 128, valid_items=64)  # pyright: ignore[reportCallIssue]
    coop._block.make_shuffle(int, threads_per_block=128, valid_items=64)  # pyright: ignore[reportCallIssue]
    coop._block.make_merge_sort_keys(int, 128, 2, _less, descending=True)  # pyright: ignore[reportCallIssue]
    coop._block.make_radix_sort_keys(int, 128, valid_items=64)  # pyright: ignore[reportCallIssue]
    coop._block.make_radix_rank(int, 128, valid_items=64)  # pyright: ignore[reportCallIssue]
    coop._block.make_histogram(int, int, threads_per_block=128, valid_items=64)  # pyright: ignore[reportCallIssue]
    coop._block.make_run_length(int, threads_per_block=128, valid_items=64)  # pyright: ignore[reportCallIssue]
    coop._warp.make_exclusive_scan(int, _add, num_valid=16)  # pyright: ignore[reportCallIssue]
    coop._warp.make_exchange(int, ranks=output)  # pyright: ignore[reportCallIssue]
    coop._warp.make_merge_sort_keys(int, 2, _less, descending=True)  # pyright: ignore[reportCallIssue]
    coop._warp.make_merge_sort_pairs(int, int, 2, _less, key_dtype=int)  # pyright: ignore[reportCallIssue]

    # The compiler-call overloads enforce the same scoped names and do not
    # advertise the removed warp merge-sort controls.
    coop._block.load(object(), values, valid_items=64)  # pyright: ignore[reportCallIssue]
    coop._block.topk_min_keys(values, 1, valid_items=64)  # pyright: ignore[reportCallIssue]
    coop._warp.load(object(), values, valid_items=16)  # pyright: ignore[reportCallIssue]
    coop._warp.merge_sort_keys(values, compare_op=_less, descending=True)  # pyright: ignore[reportCallIssue]
    coop._warp.merge_sort_pairs(values, output, compare_op=_less, value_dtype=int)  # pyright: ignore[reportCallIssue]
    coop._block.merge_sort_keys(values, compare_op=_less, valid_items=64)  # pyright: ignore
    coop._block.merge_sort_keys(values, compare_op=_less, oob_default=0)  # pyright: ignore
    coop._block.merge_sort_pairs(values, output, compare_op=_less, valid_items=64)  # pyright: ignore
    coop._block.merge_sort_pairs(values, output, compare_op=_less, oob_default=0)  # pyright: ignore
    coop._warp.merge_sort_keys(values, compare_op=_less, valid_items=53)  # pyright: ignore[reportArgumentType]
    coop._warp.merge_sort_keys(values, compare_op=_less, oob_default=0)  # pyright: ignore[reportArgumentType]
    coop._warp.merge_sort_pairs(values, output, compare_op=_less, valid_items=53)  # pyright: ignore[reportArgumentType]
    coop._warp.merge_sort_pairs(values, output, compare_op=_less, oob_default=0)  # pyright: ignore[reportArgumentType]

    # Excess positional arguments are rejected on unambiguous factory names.
    coop._block.make_load(int, 128, 2, "direct", 64, 0, None, object())  # pyright: ignore[reportCallIssue]
    coop._warp.make_exchange(
        int,
        2,
        32,
        coop._warp.WarpExchangeType.StripedToBlocked,
        None,
        None,
        None,
        None,
        object(),
    )  # pyright: ignore[reportCallIssue]
