# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Strict consumer of the qualified Numba-CUDA-MLIR surface."""

from __future__ import annotations

import operator
from typing import Generic, Literal, Protocol, TypeVar

import numpy as np
from typing_extensions import assert_type

import cuda.coop.numba_mlir as coop
from cuda import coop as portable_coop

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


def check_numba_surface(
    source: object,
    destination: object,
    readonly_values: _ReadonlyUInt16Payload,
) -> None:
    """Exercise Numba declarations through their public package."""

    block = coop.this_block()
    warp = coop.this_warp()
    logical_warp = warp.group_by(8)
    mapped_warps = block.group_by(2)
    cluster = coop.this_cluster()
    byte_values = coop.ThreadData(1, np.int8)
    values = coop.ThreadData(2, np.uint16, alignas=16)
    ranks = coop.ThreadData(2, np.int32)
    flags = coop.ThreadData(2, np.uint8)
    read_only_values = _ReadOnlyThreadData(np.uint16(1))
    read_only_ranks = _ReadOnlyThreadData(np.int32(0))
    read_only_flags = _ReadOnlyThreadData(np.uint8(1))
    int32_aggregate = coop.ThreadData(1, np.int32)
    uint16_aggregate = coop.ThreadData(1, np.uint16)
    storage = coop.TempStorage(alignment=16, sharing="shared")
    portable_storage = portable_coop.TempStorage(sharing="shared")

    assert_type(block, coop.ThreadGroup[Literal["block"]])
    assert_type(warp, coop.ThreadGroup[Literal["warp"]])
    assert_type(
        logical_warp,
        coop.ThreadGroup[Literal["threads_within_warp"]],
    )
    assert_type(byte_values, coop.ThreadDataLike[np.int8])
    assert_type(values, coop.ThreadDataLike[np.uint16])
    assert_type(storage, coop.TempStorage)
    qualified_storage: coop.TempStorageLike = storage
    assert_type(qualified_storage, coop.TempStorageLike)
    assert_type(
        coop.load(
            block,
            source,
            values,
            algorithm="direct",
            temp_storage=storage,
        ),
        coop.ThreadDataLike[np.uint16],
    )
    assert_type(
        coop.load(block, source, byte_values),
        coop.ThreadDataLike[np.int8],
    )
    assert_type(
        coop.load(
            block,
            source,
            byte_values,
            valid_items=1,
            oob_default=0,
        ),
        coop.ThreadDataLike[np.int8],
    )
    assert_type(
        coop.load(block, source, values, algorithm="vectorize"),
        coop.ThreadDataLike[np.uint16],
    )
    assert_type(
        coop.load(
            block,
            source,
            values,
            algorithm="warp_transpose_timesliced",
        ),
        coop.ThreadDataLike[np.uint16],
    )
    assert_type(
        coop.store(
            block,
            destination,
            values,
            algorithm="direct",
            temp_storage=storage,
        ),
        None,
    )
    assert_type(
        coop.exchange(block, values, mode="blocked_to_warp_striped"),
        coop.ThreadDataLike[np.uint16],
    )
    assert_type(
        coop.exchange(
            block,
            values,
            mode="scatter_to_striped_flagged",
            ranks=ranks,
            valid_flags=flags,
        ),
        coop.ThreadDataLike[np.uint16],
    )
    assert_type(
        coop.exchange(
            logical_warp,
            values,
            mode="blocked_to_striped",
        ),
        coop.ThreadDataLike[np.uint16],
    )
    assert_type(
        coop.shuffle(block, values, mode="down"),
        coop.ThreadDataLike[np.uint16],
    )
    assert_type(
        coop.exchange(block, read_only_values, mode="blocked_to_striped"),
        coop.ThreadDataLike[np.uint16],
    )
    assert_type(
        coop.exchange(
            block,
            read_only_values,
            mode="scatter_to_striped_flagged",
            ranks=read_only_ranks,
            valid_flags=read_only_flags,
        ),
        coop.ThreadDataLike[np.uint16],
    )
    assert_type(
        coop.shuffle(block, read_only_values, mode="up"),
        coop.ThreadDataLike[np.uint16],
    )
    assert_type(coop.shuffle(block, np.int32(4), mode="rotate"), np.int32)
    assert_type(coop.store(block, destination, byte_values), None)
    assert_type(
        coop.load(
            warp,
            source,
            values,
            algorithm="transpose",
        ),
        coop.ThreadDataLike[np.uint16],
    )
    assert_type(
        coop.store(
            warp,
            destination,
            values,
            algorithm="striped",
        ),
        None,
    )
    assert_type(
        coop.load(
            logical_warp,
            source,
            values,
            algorithm="transpose",
        ),
        coop.ThreadDataLike[np.uint16],
    )
    assert_type(
        coop.store(
            logical_warp,
            destination,
            values,
            algorithm="striped",
        ),
        None,
    )
    assert_type(coop.sum(block, np.int32(4)), np.int32)
    assert_type(
        coop.reduce(logical_warp, np.float32(4), binary_op="max"),
        np.float32,
    )
    assert_type(coop.sum(mapped_warps, np.uint32(4)), np.uint32)
    assert_type(coop.reduce(cluster, values, binary_op="min"), np.uint16)
    assert_type(
        coop.reduce(cluster, readonly_values, binary_op="min"),
        np.uint16,
    )
    assert_type(coop.sum(cluster, readonly_values), np.uint16)
    assert_type(coop.reduce(cluster, values, binary_op=np.maximum), np.uint16)
    assert_type(
        coop.reduce(mapped_warps, np.int32(4), binary_op=operator.add),
        np.int32,
    )
    assert_type(
        coop.reduce(
            block,
            np.int32(4),
            binary_op="max",
            broadcast=False,
            algorithm="raking_commutative_only",
        ),
        np.int32,
    )
    assert_type(
        coop.scan(
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
        coop.exclusive_scan(
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
        coop.exclusive_scan(
            block,
            np.int32(4),
            scan_op="max",
            initial_value=-17,
        ),
        np.int32,
    )
    assert_type(
        coop.inclusive_scan(
            logical_warp,
            np.int32(4),
            scan_op=_select_left_int32,
        ),
        np.int32,
    )
    assert_type(
        coop.exclusive_sum(
            block,
            values,
            algorithm="warp_scans",
            aggregate_output=uint16_aggregate,
        ),
        coop.ThreadDataLike[np.uint16],
    )
    assert_type(
        coop.inclusive_sum(
            block,
            readonly_values,
            algorithm="raking",
            temp_storage=portable_storage,
        ),
        coop.ThreadDataLike[np.uint16],
    )
    assert_type(
        coop.inclusive_sum(
            logical_warp,
            np.int32(4),
            valid_items=np.int32(7),
            aggregate_output=int32_aggregate,
        ),
        np.int32,
    )
    assert_type(
        coop.reduce(
            block,
            np.int32(4),
            binary_op=np.maximum,
            broadcast=False,
            algorithm="raking_commutative_only",
        ),
        np.int32,
    )
    assert_type(
        coop.reduce(
            block,
            np.int32(4),
            binary_op=operator.add,
            broadcast=False,
            algorithm="raking_commutative_only",
        ),
        np.int32,
    )
    assert_type(
        coop.sum(warp, np.int32(4), broadcast=False, valid_items=np.int32(7)),
        np.int32,
    )
    assert_type(
        coop.sum(block, values, broadcast=False, algorithm="raking"),
        np.uint16,
    )
    assert_type(
        coop.sum(block, readonly_values, broadcast=False, algorithm="raking"),
        np.uint16,
    )
    assert_type(
        coop.reduce(
            warp,
            np.int32(4),
            binary_op=_select_left_int32,
            broadcast=False,
        ),
        np.int32,
    )
    assert_type(
        coop.reduce(
            block,
            values,
            binary_op=_select_left_uint16,
            broadcast=False,
            algorithm="warp_reductions",
        ),
        np.uint16,
    )
    assert_type(
        coop.reduce(
            block,
            readonly_values,
            binary_op=_select_left_uint16,
            broadcast=False,
            algorithm="warp_reductions",
        ),
        np.uint16,
    )
    assert_type(
        coop.reduce(
            block,
            np.int32(4),
            binary_op=_select_left_int32,
            broadcast=False,
            algorithm="raking",
        ),
        np.int32,
    )
