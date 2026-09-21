# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Strict consumer of the qualified Numba-CUDA-MLIR surface."""

from __future__ import annotations

import operator
from typing import Any, Generic, Literal, Protocol, TypeVar

import numpy as np
from typing_extensions import assert_type

import cuda.coop.numba_mlir as numba_coop
from cuda import coop as common_coop

_ItemT = TypeVar("_ItemT")


class _ReadOnlyThreadData(Generic[_ItemT]):
    """Structural readable payload without mutable item access."""

    items_per_thread: int
    dtype: object | None

    def __init__(self, value: _ItemT) -> None:
        self.items_per_thread = 1
        self.dtype = type(value)
        self._value = value

    def __len__(self) -> int:
        return self.items_per_thread

    def __getitem__(self, index: int, /) -> _ItemT:
        if index != 0:
            raise IndexError(index)
        return self._value


class _ReadonlyUInt16Payload(Protocol):
    items_per_thread: int
    dtype: object | None

    def __len__(self) -> int: ...

    def __getitem__(self, index: int, /) -> np.uint16: ...


def _select_left_int32(left: np.int32, right: np.int32) -> np.int32:
    del right
    return left


def _select_left_uint16(left: np.uint16, right: np.uint16) -> np.uint16:
    del right
    return left


def _prefix_from_int32_aggregate(block_aggregate: np.int32) -> np.int32:
    return block_aggregate


def _prefix_from_uint16_aggregate(block_aggregate: np.uint16) -> np.uint16:
    return block_aggregate


def _carry_int32_prefix(
    state: numba_coop.ThreadDataLike[np.int32],
    block_aggregate: np.int32,
) -> np.int32:
    previous = state[0]
    state[0] = block_aggregate
    return previous


def _carry_uint16_prefix(
    state: numba_coop.ThreadDataLike[np.uint16],
    block_aggregate: np.uint16,
) -> np.uint16:
    previous = state[0]
    state[0] = block_aggregate
    return previous


def _carry_int64_for_int32(
    state: numba_coop.ThreadDataLike[np.int64],
    block_aggregate: np.int32,
) -> np.int32:
    previous = np.int32(state[0])
    state[0] += np.int64(block_aggregate)
    return previous


class _Int32PrefixFunctor:
    def __call__(self, block_aggregate: np.int32) -> np.int32:
        return block_aggregate


_INT32_RUNNING_PREFIX = numba_coop.StatefulFunction(
    _carry_int32_prefix,
    np.int32,
    name="typing_int32_running_prefix",
)
_UINT16_RUNNING_PREFIX = numba_coop.StatefulFunction(
    _carry_uint16_prefix,
    np.uint16,
    name="typing_uint16_running_prefix",
)
_INT64_STATE_INT32_PREFIX = numba_coop.StatefulFunction(
    _carry_int64_for_int32,
    np.int64,
)
_INT32_PREFIX_FUNCTOR: numba_coop.StatefulFunction[np.int64, np.int32] = (
    numba_coop.StatefulFunction(
        _Int32PrefixFunctor,
        np.int64,
    )
)

assert_type(
    _INT32_RUNNING_PREFIX,
    numba_coop.StatefulFunction[np.int32, np.int32],
)
assert_type(
    _INT64_STATE_INT32_PREFIX,
    numba_coop.StatefulFunction[np.int64, np.int32],
)


def check_numba_surface(
    source: object,
    destination: object,
    readonly_values: _ReadonlyUInt16Payload,
    compiler_integer_dtype: Any,
) -> None:
    """Exercise Numba declarations through their public package."""

    block = numba_coop.this_block()
    warp = numba_coop.this_warp()
    logical_warp = warp.group_by(8)
    mapped_warps = block.group_by(2)
    cluster = numba_coop.this_cluster()
    byte_values = numba_coop.ThreadData(1, np.int8)
    values = numba_coop.ThreadData(2, np.uint16, alignment=16)
    ranks = numba_coop.ThreadData(2, np.int32)
    flags = numba_coop.ThreadData(2, np.uint8)
    read_only_values = _ReadOnlyThreadData(np.uint16(1))
    read_only_ranks = _ReadOnlyThreadData(np.int32(0))
    read_only_flags = _ReadOnlyThreadData(np.uint8(1))
    int32_aggregate = numba_coop.ThreadData(1, np.int32)
    uint16_aggregate = numba_coop.ThreadData(1, np.uint16)
    int32_prefix_state = numba_coop.ThreadData(1, np.int32)
    uint16_prefix_state = numba_coop.ThreadData(1, np.uint16)
    int64_prefix_state = numba_coop.ThreadData(1, np.int64)
    storage = numba_coop.TempStorage(alignment=16, sharing="shared")
    common_storage = common_coop.TempStorage(sharing="shared")

    assert_type(block, numba_coop.ThreadGroup[Literal["block"]])
    assert_type(warp, numba_coop.ThreadGroup[Literal["warp"]])
    assert_type(
        logical_warp,
        numba_coop.ThreadGroup[Literal["threads_within_warp"]],
    )
    generic_group: numba_coop.ThreadGroup = block
    generic_group.rank()
    generic_group.count("warp")
    assert_type(generic_group.rank_as(np.uint32), np.uint32)
    assert_type(generic_group.count_as(int, "warp"), int)
    assert_type(block.rank(), np.uint32 | np.uint64)
    assert_type(block.count("warp"), np.uint32 | np.uint64)
    assert_type(block.rank_as(np.uint16), np.uint16)
    assert_type(block.rank_as(int), int)
    assert_type(block.count_as(np.int64, "grid"), np.int64)
    block.count_as(compiler_integer_dtype)
    assert_type(block.is_member(), np.uint8)
    assert_type(block.sync(), None)
    assert_type(block.sync_aligned(), None)
    assert_type(logical_warp.sync(), None)
    assert_type(mapped_warps.rank("warp"), np.uint32 | np.uint64)
    assert_type(mapped_warps.count("block"), np.uint32 | np.uint64)
    assert_type(mapped_warps.is_member(), np.uint8)
    assert_type(numba_coop.this_grid().rank(), np.uint32 | np.uint64)
    assert_type(byte_values, numba_coop.ThreadDataLike[np.int8])
    assert_type(values, numba_coop.ThreadDataLike[np.uint16])
    assert_type(storage, numba_coop.TempStorage)
    qualified_storage: numba_coop.TempStorageLike = storage
    assert_type(qualified_storage, numba_coop.TempStorageLike)
    assert_type(
        numba_coop.load(
            block,
            source,
            values,
            algorithm="direct",
            temp_storage=storage,
        ),
        None,
    )
    assert_type(
        numba_coop.load(block, source, byte_values),
        None,
    )
    assert_type(
        numba_coop.load(
            block,
            source,
            byte_values,
            valid_items=1,
            oob_default=0,
        ),
        None,
    )
    assert_type(
        numba_coop.load(block, source, values, algorithm="vectorize"),
        None,
    )
    assert_type(
        numba_coop.load(
            block,
            source,
            values,
            algorithm="warp_transpose_timesliced",
        ),
        None,
    )
    assert_type(
        numba_coop.store(
            block,
            destination,
            values,
            algorithm="direct",
            temp_storage=storage,
        ),
        None,
    )
    assert_type(
        numba_coop.exchange(block, values, mode="blocked_to_warp_striped"),
        numba_coop.ThreadDataLike[np.uint16],
    )
    assert_type(
        numba_coop.exchange(
            block,
            values,
            mode="scatter_to_striped_flagged",
            ranks=ranks,
            valid_flags=flags,
        ),
        numba_coop.ThreadDataLike[np.uint16],
    )
    assert_type(
        numba_coop.exchange(
            logical_warp,
            values,
            mode="blocked_to_striped",
        ),
        numba_coop.ThreadDataLike[np.uint16],
    )
    assert_type(
        numba_coop.shuffle(block, values, mode="down"),
        numba_coop.ThreadDataLike[np.uint16],
    )
    assert_type(
        numba_coop.exchange(block, read_only_values, mode="blocked_to_striped"),
        numba_coop.ThreadDataLike[np.uint16],
    )
    assert_type(
        numba_coop.exchange(
            block,
            read_only_values,
            mode="scatter_to_striped_flagged",
            ranks=read_only_ranks,
            valid_flags=read_only_flags,
        ),
        numba_coop.ThreadDataLike[np.uint16],
    )
    assert_type(
        numba_coop.shuffle(block, read_only_values, mode="up"),
        numba_coop.ThreadDataLike[np.uint16],
    )
    assert_type(numba_coop.shuffle(block, np.int32(4), mode="rotate"), np.int32)
    assert_type(numba_coop.store(block, destination, byte_values), None)
    assert_type(
        numba_coop.load(
            warp,
            source,
            values,
            algorithm="transpose",
        ),
        None,
    )
    assert_type(
        numba_coop.store(
            warp,
            destination,
            values,
            algorithm="striped",
        ),
        None,
    )
    assert_type(
        numba_coop.load(
            logical_warp,
            source,
            values,
            algorithm="transpose",
        ),
        None,
    )
    assert_type(
        numba_coop.store(
            logical_warp,
            destination,
            values,
            algorithm="striped",
        ),
        None,
    )
    assert_type(numba_coop.sum(block, np.int32(4)), np.int32)
    assert_type(
        numba_coop.reduce(logical_warp, np.float32(4), binary_op="max"),
        np.float32,
    )
    assert_type(numba_coop.sum(mapped_warps, np.uint32(4)), np.uint32)
    assert_type(numba_coop.reduce(cluster, values, binary_op="min"), np.uint16)
    assert_type(
        numba_coop.reduce(cluster, readonly_values, binary_op="min"),
        np.uint16,
    )
    assert_type(numba_coop.sum(cluster, readonly_values), np.uint16)
    assert_type(numba_coop.reduce(cluster, values, binary_op=np.maximum), np.uint16)
    assert_type(
        numba_coop.reduce(mapped_warps, np.int32(4), binary_op=operator.add),
        np.int32,
    )
    assert_type(
        numba_coop.reduce(
            block,
            np.int32(4),
            binary_op="max",
            broadcast=False,
            algorithm="raking_commutative_only",
        ),
        np.int32,
    )
    assert_type(
        numba_coop.scan(block, values, scan_op=np.add),
        numba_coop.ThreadDataLike[np.uint16],
    )
    assert_type(numba_coop.scan(block, np.int32(4), scan_op=np.add), np.int32)
    assert_type(numba_coop.scan(warp, np.int32(4), scan_op=np.add), np.int32)
    assert_type(
        numba_coop.exclusive_scan(block, values, scan_op=np.add),
        numba_coop.ThreadDataLike[np.uint16],
    )
    assert_type(numba_coop.exclusive_scan(block, np.int32(4), scan_op=np.add), np.int32)
    assert_type(numba_coop.exclusive_scan(warp, np.int32(4), scan_op=np.add), np.int32)
    assert_type(
        numba_coop.scan(
            block,
            np.int32(4),
            mode="inclusive",
            scan_op=np.maximum,
            algorithm="raking_memoize",
            aggregate_output=int32_aggregate,
        ),
        np.int32,
    )
    assert_type(
        numba_coop.inclusive_sum(
            block,
            np.int32(4),
            None,
            prefix_op=None,
        ),
        np.int32,
    )
    assert_type(
        numba_coop.exclusive_scan(
            warp,
            np.int32(4),
            scan_op=operator.mul,
            initial_value=np.int32(1),
            valid_items=np.int32(7),
            aggregate_output=int32_aggregate,
        ),
        np.int32,
    )
    assert_type(
        numba_coop.exclusive_scan(
            block,
            np.int32(4),
            scan_op="max",
            initial_value=-17,
        ),
        np.int32,
    )
    assert_type(
        numba_coop.inclusive_scan(
            logical_warp,
            np.int32(4),
            scan_op=_select_left_int32,
        ),
        np.int32,
    )
    assert_type(
        numba_coop.exclusive_sum(
            block,
            values,
            algorithm="warp_scans",
            aggregate_output=uint16_aggregate,
        ),
        numba_coop.ThreadDataLike[np.uint16],
    )
    assert_type(
        numba_coop.inclusive_sum(
            block,
            readonly_values,
            algorithm="raking",
            temp_storage=common_storage,
        ),
        numba_coop.ThreadDataLike[np.uint16],
    )
    assert_type(
        numba_coop.inclusive_sum(
            logical_warp,
            np.int32(4),
            valid_items=np.int32(7),
            aggregate_output=int32_aggregate,
        ),
        np.int32,
    )
    assert_type(
        numba_coop.inclusive_sum(
            block,
            np.int32(4),
            int64_prefix_state,
            prefix_op=_INT32_PREFIX_FUNCTOR,
        ),
        np.int32,
    )
    assert_type(
        numba_coop.exclusive_scan(
            block,
            np.int32(4),
            scan_op=_select_left_int32,
            prefix_op=_prefix_from_int32_aggregate,
        ),
        np.int32,
    )
    assert_type(
        numba_coop.scan(
            block,
            values,
            mode="inclusive",
            prefix_op=_prefix_from_uint16_aggregate,
        ),
        numba_coop.ThreadDataLike[np.uint16],
    )
    assert_type(
        numba_coop.exclusive_sum(
            block,
            np.int32(4),
            int64_prefix_state,
            prefix_op=_INT64_STATE_INT32_PREFIX,
        ),
        np.int32,
    )
    assert_type(
        numba_coop.exclusive_sum(
            block,
            np.int32(4),
            int32_prefix_state,
            prefix_op=_INT32_RUNNING_PREFIX,
        ),
        np.int32,
    )
    assert_type(
        numba_coop.inclusive_sum(
            block,
            values,
            uint16_prefix_state,
            prefix_op=_UINT16_RUNNING_PREFIX,
        ),
        numba_coop.ThreadDataLike[np.uint16],
    )
    assert_type(
        numba_coop.reduce(
            block,
            np.int32(4),
            binary_op=np.maximum,
            broadcast=False,
            algorithm="raking_commutative_only",
        ),
        np.int32,
    )
    assert_type(
        numba_coop.reduce(
            block,
            np.int32(4),
            binary_op=operator.add,
            broadcast=False,
            algorithm="raking_commutative_only",
        ),
        np.int32,
    )
    assert_type(
        numba_coop.sum(warp, np.int32(4), broadcast=False, valid_items=np.int32(7)),
        np.int32,
    )
    assert_type(
        numba_coop.sum(block, values, broadcast=False, algorithm="raking"),
        np.uint16,
    )
    assert_type(
        numba_coop.sum(block, readonly_values, broadcast=False, algorithm="raking"),
        np.uint16,
    )
    assert_type(
        numba_coop.reduce(
            warp,
            np.int32(4),
            binary_op=_select_left_int32,
            broadcast=False,
        ),
        np.int32,
    )
    assert_type(
        numba_coop.reduce(
            block,
            values,
            binary_op=_select_left_uint16,
            broadcast=False,
            algorithm="warp_reductions",
        ),
        np.uint16,
    )
    assert_type(
        numba_coop.reduce(
            block,
            readonly_values,
            binary_op=_select_left_uint16,
            broadcast=False,
            algorithm="warp_reductions",
        ),
        np.uint16,
    )
    assert_type(
        numba_coop.reduce(
            block,
            np.int32(4),
            binary_op=_select_left_int32,
            broadcast=False,
            algorithm="raking",
        ),
        np.int32,
    )


def check_merge_sort_surface() -> None:
    keys = numba_coop.ThreadData(3, np.int32)
    values = numba_coop.ThreadData(3, np.float64)

    def compare(left: np.int32, right: np.int32) -> np.bool_:
        return left > right

    assert_type(
        numba_coop.merge_sort_keys(numba_coop.this_block(), keys, compare_op=compare),
        numba_coop.ThreadDataLike[np.int32],
    )
    assert_type(
        numba_coop.merge_sort_pairs(
            numba_coop.this_warp().group_by(8),
            keys,
            values,
            valid_items=23,
            oob_default=-1000,
            compare_op=compare,
        ),
        tuple[
            numba_coop.ThreadDataLike[np.int32], numba_coop.ThreadDataLike[np.float64]
        ],
    )


def check_radix_surface() -> None:
    block = numba_coop.this_block()
    keys = numba_coop.ThreadData(3, np.int32)
    values = numba_coop.ThreadData(3, np.float64)
    assert_type(
        numba_coop.radix_sort_keys(block, keys), numba_coop.ThreadDataLike[np.int32]
    )
    assert_type(
        numba_coop.radix_rank(block, keys, radix_bits=4),
        numba_coop.ThreadDataLike[np.int32],
    )
    assert_type(
        numba_coop.radix_sort_pairs(block, keys, values),
        tuple[
            numba_coop.ThreadDataLike[np.int32], numba_coop.ThreadDataLike[np.float64]
        ],
    )
    prefix = numba_coop.ThreadData(1, np.int32)
    assert_type(
        numba_coop.radix_rank(block, keys, exclusive_digit_prefix=prefix),
        numba_coop.ThreadDataLike[np.int32],
    )
    assert_type(
        numba_coop.radix_sort_keys(block, np.float64(1.5), blocked_to_striped=True),
        np.float64,
    )


def check_topk_surface() -> None:
    block = numba_coop.this_block()
    keys = numba_coop.ThreadData(3, np.int16)
    values = numba_coop.ThreadData(3, np.float64)
    assert_type(
        numba_coop.topk_min_keys(block, keys, k=7), numba_coop.ThreadDataLike[np.int16]
    )
    assert_type(
        numba_coop.topk_max_keys(block, keys, k=np.int64(7), valid_items=31),
        numba_coop.ThreadDataLike[np.int16],
    )
    assert_type(
        numba_coop.topk_min_pairs(block, keys, values, k=7),
        tuple[
            numba_coop.ThreadDataLike[np.int16], numba_coop.ThreadDataLike[np.float64]
        ],
    )
    assert_type(
        numba_coop.topk_max_pairs(
            block, keys, values, k=7, temp_storage=numba_coop.TempStorage()
        ),
        tuple[
            numba_coop.ThreadDataLike[np.int16], numba_coop.ThreadDataLike[np.float64]
        ],
    )


def check_neighbor_results() -> None:
    block = numba_coop.this_block()
    values = numba_coop.ThreadData(3, np.float64)
    assert_type(
        numba_coop.adjacent_difference(block, values),
        numba_coop.ThreadDataLike[np.float64],
    )
    assert_type(
        numba_coop.discontinuity(block, values), numba_coop.ThreadDataLike[np.int32]
    )
    assert_type(
        numba_coop.discontinuity(block, values, mode="heads_and_tails"),
        tuple[numba_coop.ThreadDataLike[np.int32], numba_coop.ThreadDataLike[np.int32]],
    )
    assert_type(
        numba_coop.adjacent_difference(
            block, values, difference_op=lambda current, neighbor: current - neighbor
        ),
        numba_coop.ThreadDataLike[np.float64],
    )
    assert_type(
        numba_coop.discontinuity(
            block, values, flag_op=lambda previous, current: previous < current
        ),
        numba_coop.ThreadDataLike[np.int32],
    )


def check_histogram_surface() -> None:
    block = numba_coop.this_block()
    samples = numba_coop.ThreadData(3, np.uint8)
    assert_type(
        numba_coop.histogram(block, samples, bins=33),
        numba_coop.ThreadDataLike[np.int32],
    )
    assert_type(
        numba_coop.histogram(
            block,
            samples,
            bins=65,
            bins_per_thread=2,
            counter_dtype=np.uint64,
            algorithm="sort",
        ),
        numba_coop.ThreadDataLike[np.uint64],
    )
    assert_type(
        numba_coop.histogram(block, samples, bins=33, counter_dtype=int),
        numba_coop.ThreadDataLike[np.int32],
    )
    assert_type(
        numba_coop.histogram(block, np.int64(3), bins=33),
        numba_coop.ThreadDataLike[np.int32],
    )


def check_run_length_surface(destination: object, offsets: object) -> None:
    block = numba_coop.this_block()
    values = numba_coop.ThreadData(2, np.float32)
    lengths = numba_coop.ThreadData(2, np.uint64)
    total = numba_coop.ThreadData(1, np.uint64)
    relative = numba_coop.ThreadData(4, np.uint64)
    assert_type(
        numba_coop.run_length_decode(
            block,
            values,
            lengths,
            decoded_items_per_thread=4,
            decoded_window_offset=np.uint64(2**32),
            decoded_offset_dtype=np.uint64,
            total_decoded_size=total,
            relative_offsets=relative,
        ),
        numba_coop.ThreadDataLike[np.float32],
    )
    assert_type(
        numba_coop.run_length_decode_into(
            block,
            values,
            lengths,
            destination,
            decoded_items_per_thread=4,
            relative_offsets=offsets,
            decoded_offset_dtype=np.uint64,
        ),
        np.uint32 | np.uint64,
    )


def check_batched_reduction_typing() -> None:
    warp = numba_coop.this_warp()
    values = numba_coop.ThreadData(3, np.float32)
    assert_type(
        numba_coop.reduce_batched(warp, values), numba_coop.ThreadDataLike[np.float32]
    )
