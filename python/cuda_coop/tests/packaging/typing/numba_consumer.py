# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import operator
from typing import Generic, Literal, TypeVar

import numpy as np
from typing_extensions import assert_type

import cuda.coop.numba_mlir as coop

_ItemT = TypeVar("_ItemT")


class _ReadOnlyThreadData(Generic[_ItemT]):
    """Check readable-input protocols without requiring mutable item access.

    These consumers check static declarations; they do not compile kernels
    with this custom Python payload.
    """

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


def _select_left_int32(left: np.int32, right: np.int32) -> np.int32:
    del right
    return left


def _select_left_uint16(left: np.uint16, right: np.uint16) -> np.uint16:
    del right
    return left


def check_numba_surface(source: object, destination: object) -> None:
    """Exercise Numba declarations through their public package."""

    block = coop.this_block()
    warp = coop.this_warp()
    logical_warp = warp.group_by(8)
    mapped_warps = block.group_by(2)
    cluster = coop.this_cluster()
    byte_values = coop.ThreadData(1, np.int8)
    values = coop.ThreadData(2, np.uint16, alignment=16)
    storage = coop.TempStorage(alignment=16, sharing="shared")

    assert_type(block, coop.ThreadGroup[Literal["block"]])
    assert_type(warp, coop.ThreadGroup[Literal["warp"]])
    assert_type(
        logical_warp,
        coop.ThreadGroup[Literal["threads_within_warp"]],
    )
    assert_type(byte_values, coop.ThreadDataLike[np.int8])
    assert_type(values, coop.ThreadDataLike[np.uint16])
    assert_type(storage, coop.TempStorage)
    common_storage: coop.TempStorageLike = storage
    assert_type(common_storage, coop.TempStorageLike)
    assert_type(
        coop.load(
            block,
            source,
            values,
            algorithm="direct",
            temp_storage=storage,
        ),
        None,
    )
    assert_type(
        coop.load(block, source, byte_values),
        None,
    )
    assert_type(
        coop.load(
            block,
            source,
            byte_values,
            valid_items=1,
            oob_default=0,
        ),
        None,
    )
    assert_type(
        coop.load(block, source, values, algorithm="vectorize"),
        None,
    )
    assert_type(
        coop.load(
            block,
            source,
            values,
            algorithm="warp_transpose_timesliced",
        ),
        None,
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
    assert_type(coop.store(block, destination, byte_values), None)
    assert_type(
        coop.load(
            warp,
            source,
            values,
            algorithm="transpose",
        ),
        None,
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
        None,
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
            np.int32(4),
            binary_op=_select_left_int32,
            broadcast=False,
            algorithm="raking",
        ),
        np.int32,
    )
