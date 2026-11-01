# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Strict typing for payload construction, adapters, hierarchy, and storage."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from cutlass import Int16, Int32, Uint8, Uint16, Uint32, Uint64
from typing_extensions import assert_type

import cuda.coop.cutlass as coop


def check_payloads() -> None:
    inferred = coop.ThreadData(2)
    assert_type(inferred, coop.ThreadData[Any])
    values = coop.ThreadData(items_per_thread=2, dtype=np.int32, alignment=16)
    assert_type(values, coop.ThreadData[np.int32])
    assert_type(values[0], np.int32)
    assert_type(coop.ThreadData.from_payload(values), coop.ThreadData[np.int32])
    assert_type(
        coop.ThreadData.from_fn(2, lambda index: np.int32(index)),
        coop.ThreadData[np.int32],
    )
    assert_type(
        coop.ThreadData.from_values(np.int32(1), np.int32(2)),
        coop.ThreadData[np.int32],
    )
    fragment = values.to_register_tensor()
    vector = values.to_tensor_ssa()
    assert_type(
        coop.ThreadData.from_register_tensor(fragment, dtype=np.int32),
        coop.ThreadData[np.int32],
    )
    assert_type(
        coop.ThreadData.from_vector(vector, dtype=np.int32),
        coop.ThreadData[np.int32],
    )


def check_storage() -> None:
    storage = coop.TempStorage(alignment=32, sharing="exclusive")
    assert_type(storage, coop.TempStorage)
    assert_type(storage.auto_sync, bool)
    assert_type(storage.size_in_bytes, int | None)
    assert_type(storage.alignment, int | None)
    storage.reserve(32, Int32, alignment=16)
    assert_type(storage.sync(), None)
    storage.release(Int32)


def check_cutlass_hierarchy_surface() -> None:
    """Check group kinds, query dtypes and synchronization."""
    thread = coop.this_thread()
    block = coop.this_block()
    cluster = coop.this_cluster()
    grid = coop.this_grid()
    assert_type(thread, coop.ThreadGroup[Literal["thread"]])
    assert_type(cluster, coop.ThreadGroup[Literal["cluster"]])
    assert_type(grid, coop.ThreadGroup[Literal["grid"]])
    assert_type(block.rank(), Uint32 | Uint64)
    assert_type(block.count("warp"), Uint32 | Uint64)
    assert_type(grid.rank(), Uint32 | Uint64)
    assert_type(block.rank_as(np.int64), np.int64)
    assert_type(block.count_as(int), int)
    assert_type(block.rank_as(Int16), Int16)
    assert_type(block.count_as(Uint16), Uint16)
    assert_type(block.rank_as(), Uint32 | Uint64)
    assert_type(block.is_member(), Uint8)
    assert_type(thread.sync(), None)
    assert_type(block.sync_aligned(), None)
    logical = coop.this_warp().group_by(8)
    assert_type(logical.rank("warp"), Uint32 | Uint64)
    assert_type(logical.count_as(Uint32, "thread"), Uint32)
    assert_type(logical.sync(), None)
    mapped = block.group_by(2, exhaustive=False)
    assert_type(mapped.rank_as(Int32, "block"), Int32)
    assert_type(mapped.is_member(), Uint8)
