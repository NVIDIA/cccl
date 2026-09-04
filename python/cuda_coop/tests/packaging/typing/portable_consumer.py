# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Strict consumer of the portable cooperative data-movement surface."""

from __future__ import annotations

from typing import Generic, Literal, TypeVar

import numpy as np
from typing_extensions import assert_type

import cuda.coop as coop

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


def check_portable_surface(source: object, destination: object) -> None:
    """Exercise public declarations without importing package internals."""

    block = coop.this_block()
    warp = coop.this_warp()
    logical_warp = warp.group_by(8)
    values = coop.ThreadData(2, np.int16)
    read_only_values = _ReadOnlyThreadData(np.int16(1))
    storage = coop.TempStorage(sharing="shared")

    assert_type(block, coop.ThreadGroup[Literal["block"]])
    assert_type(warp, coop.ThreadGroup[Literal["warp"]])
    assert_type(
        logical_warp,
        coop.ThreadGroup[Literal["threads_within_warp"]],
    )
    assert_type(values, coop.ThreadDataLike[np.int16])
    assert_type(storage, coop.TempStorageLike)
    assert_type(
        coop.load(
            block,
            source,
            values,
            algorithm="direct",
            valid_items=15,
            oob_default=np.int16(0),
            offset=1,
            temp_storage=storage,
        ),
        coop.ThreadDataLike[np.int16],
    )
    assert_type(
        coop.load(block, source, values, algorithm="warp_transpose_timesliced"),
        coop.ThreadDataLike[np.int16],
    )
    assert_type(
        coop.store(
            block,
            destination,
            values,
            algorithm="direct",
            valid_items=15,
            offset=1,
            temp_storage=storage,
        ),
        None,
    )
    assert_type(
        coop.load(warp, source, values, algorithm="transpose"),
        coop.ThreadDataLike[np.int16],
    )
    assert_type(
        coop.store(warp, destination, values, algorithm="vectorize"),
        None,
    )
    assert_type(
        coop.load(logical_warp, source, values, algorithm="transpose"),
        coop.ThreadDataLike[np.int16],
    )
    assert_type(
        coop.store(logical_warp, destination, values, algorithm="striped"),
        None,
    )
    assert_type(
        coop.exchange(block, values, mode="blocked_to_striped"),
        coop.ThreadDataLike[np.int16],
    )
    assert_type(
        coop.exchange(logical_warp, values, mode="striped_to_blocked"),
        coop.ThreadDataLike[np.int16],
    )
    assert_type(
        coop.shuffle(block, values, mode="up", distance=1),
        coop.ThreadDataLike[np.int16],
    )
    assert_type(
        coop.exchange(block, read_only_values, mode="blocked_to_striped"),
        coop.ThreadDataLike[np.int16],
    )
    assert_type(
        coop.shuffle(block, read_only_values, mode="down"),
        coop.ThreadDataLike[np.int16],
    )
