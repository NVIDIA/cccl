# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Strict consumer of the qualified CUTLASS primitive API."""

from __future__ import annotations

import operator
from typing import Any, Literal

import numpy as np
from cutlass import Float32, Float64, Int16, Int32, Int64, Uint8, Uint16, Uint32, Uint64
from typing_extensions import assert_type

import cuda.coop.cutlass as cutlass_coop
from cuda import coop as common_coop


def register_cutlass() -> None:
    assert_type(common_coop.register("cutlass"), None)


def check_cutlass_surface(source: object, destination: object) -> None:
    block = cutlass_coop.this_block()
    values = cutlass_coop.ThreadData(2, np.int32, alignment=16)
    assert_type(block, cutlass_coop.ThreadGroup[Literal["block"]])
    assert_type(values, cutlass_coop.ThreadData[np.int32])
    assert_type(values[0], np.int32)
    assert_type(cutlass_coop.load(block, source, values), None)
    assert_type(cutlass_coop.store(block, destination, values), None)
    cutlass_coop.load(
        block,
        source,
        values,
        algorithm="direct",
        valid_items=31,
        oob_default=-1,
        offset=4,
    )
    cutlass_coop.store(block, destination, values, valid_items=31, offset=4)
    common_coop.load(common_coop.this_block(), source, values)
    common_coop.store(common_coop.this_block(), destination, values)

    storage = cutlass_coop.TempStorage(alignment=1, sharing="exclusive")
    assert_type(storage, cutlass_coop.TempStorage)
    assert_type(storage.auto_sync, bool)
    assert_type(storage.size_in_bytes, int | None)
    assert_type(storage.alignment, int | None)
    cutlass_coop.load(
        block, source, values, algorithm="transpose", temp_storage=storage
    )
    cutlass_coop.store(
        block, destination, values, algorithm="transpose", temp_storage=storage
    )
    manual = cutlass_coop.TempStorage(
        16384, alignment=32, auto_sync=False, sharing="shared"
    )
    common_coop.load(
        common_coop.this_block(),
        source,
        values,
        algorithm="transpose",
        temp_storage=manual,
    )
    assert_type(manual.sync(), None)
    common_coop.store(
        common_coop.this_block(),
        destination,
        values,
        algorithm="transpose",
        temp_storage=manual,
    )
    manual.sync()

    common_storage = common_coop.TempStorage(alignment=32)
    cutlass_coop.store(
        block, destination, values, algorithm="transpose", temp_storage=common_storage
    )

    copied = cutlass_coop.ThreadData.from_payload(values)
    assert_type(copied, cutlass_coop.ThreadData[np.int32])
    generated = cutlass_coop.ThreadData.from_fn(2, lambda index: np.int32(index))
    assert_type(generated, cutlass_coop.ThreadData[np.int32])
    initialized = cutlass_coop.ThreadData.from_values(np.int32(1), np.int32(2))
    assert_type(initialized, cutlass_coop.ThreadData[np.int32])
    fragment = values.to_register_tensor()
    restored = cutlass_coop.ThreadData.from_register_tensor(fragment, dtype=np.int32)
    assert_type(restored, cutlass_coop.ThreadData[np.int32])
    vector = values.to_tensor_ssa()
    restored_vector = cutlass_coop.ThreadData.from_vector(vector, dtype=np.int32)
    assert_type(restored_vector, cutlass_coop.ThreadData[np.int32])


def check_cutlass_warp_surface(source: object, destination: object) -> None:
    warp = cutlass_coop.this_warp()
    values = cutlass_coop.ThreadData(2, np.int32)
    assert_type(warp, cutlass_coop.ThreadGroup[Literal["warp"]])
    assert_type(cutlass_coop.load(warp, source, values), None)
    assert_type(cutlass_coop.store(warp, destination, values), None)
    for algorithm in ("direct", "striped", "vectorize", "transpose"):
        cutlass_coop.load(
            warp,
            source,
            values,
            algorithm=algorithm,
            valid_items=61,
            oob_default=-1,
            offset=4,
        )
        cutlass_coop.store(
            warp, destination, values, algorithm=algorithm, valid_items=61
        )
    common_coop.load(warp, source, values)
    common_coop.store(warp, destination, values)
    cutlass_coop.load(common_coop.this_warp(), source, values)
    cutlass_coop.store(common_coop.this_warp(), destination, values)


def check_cutlass_logical_warp_surface(source: object, destination: object) -> None:
    values = cutlass_coop.ThreadData(2, np.int32)
    for width in (1, 2, 4, 8, 16, 32):
        group = cutlass_coop.this_warp().group_by(width)
        assert_type(group, cutlass_coop.ThreadGroup[Literal["threads_within_warp"]])
        for algorithm in ("direct", "striped", "vectorize", "transpose"):
            assert_type(
                cutlass_coop.load(group, source, values, algorithm=algorithm), None
            )
            assert_type(
                cutlass_coop.store(group, destination, values, algorithm=algorithm),
                None,
            )
        common_coop.load(group, source, values)
        common_coop.store(group, destination, values)
    group = cutlass_coop.this_warp().group_by(8, exhaustive=False)
    assert_type(group, cutlass_coop.ThreadGroup[Literal["threads_within_warp"]])
    cutlass_coop.load(common_coop.this_warp().group_by(8), source, values)
    cutlass_coop.store(common_coop.this_warp().group_by(8), destination, values)

    mapped = cutlass_coop.this_block().group_by(2, exhaustive=False)
    assert_type(mapped, cutlass_coop.ThreadGroup[Literal["warps_within_block"]])


def check_cutlass_hierarchy_surface() -> None:
    thread = cutlass_coop.this_thread()
    block = cutlass_coop.this_block()
    cluster = cutlass_coop.this_cluster()
    grid = cutlass_coop.this_grid()
    assert_type(thread, cutlass_coop.ThreadGroup[Literal["thread"]])
    assert_type(cluster, cutlass_coop.ThreadGroup[Literal["cluster"]])
    assert_type(grid, cutlass_coop.ThreadGroup[Literal["grid"]])
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
    logical = cutlass_coop.this_warp().group_by(8)
    assert_type(logical.rank("warp"), Uint32 | Uint64)
    assert_type(logical.count_as(Uint32, "thread"), Uint32)
    assert_type(logical.sync(), None)
    mapped = block.group_by(2, exhaustive=False)
    assert_type(mapped.rank_as(Int32, "block"), Int32)
    assert_type(mapped.is_member(), Uint8)


def check_cutlass_reduce_surface(scalar: Uint32) -> None:
    block = cutlass_coop.this_block()
    values = cutlass_coop.ThreadData(2, np.int32)
    assert_type(cutlass_coop.reduce(block, values), np.int32)
    assert_type(cutlass_coop.sum(block, values), np.int32)
    assert_type(cutlass_coop.reduce(block, scalar, binary_op="max"), Uint32)
    assert_type(cutlass_coop.sum(block, scalar), Uint32)
    assert_type(common_coop.sum(block, scalar), Uint32)
    assert_type(cutlass_coop.sum(common_coop.this_block(), scalar), Uint32)
    assert_type(cutlass_coop.reduce(block, scalar, binary_op=operator.add), Uint32)
    assert_type(cutlass_coop.reduce(block, values, binary_op=np.maximum), np.int32)
    assert_type(
        cutlass_coop.sum(block, scalar, broadcast=False, valid_items=17), Uint32
    )
    assert_type(
        cutlass_coop.reduce(block, values, broadcast=False, algorithm="raking"),
        np.int32,
    )
    assert_type(
        cutlass_coop.sum(
            cutlass_coop.this_warp().group_by(8), scalar, broadcast=False, valid_items=7
        ),
        Uint32,
    )
    assert_type(cutlass_coop.sum(cutlass_coop.this_thread(), scalar), Uint32)
    assert_type(cutlass_coop.sum(block.group_by(2), values), np.int32)
    assert_type(
        cutlass_coop.sum(cutlass_coop.this_cluster(), values, broadcast=False), np.int32
    )


def check_cutlass_scan_surface(scalar: Uint32) -> None:
    block = cutlass_coop.this_block()
    logical = cutlass_coop.this_warp().group_by(8)
    values = cutlass_coop.ThreadData(2, np.int32)
    aggregate = cutlass_coop.ThreadData(1, Uint32)
    assert_type(cutlass_coop.scan(block, values), cutlass_coop.ThreadData[np.int32])
    assert_type(
        cutlass_coop.exclusive_scan(block, values), cutlass_coop.ThreadData[np.int32]
    )
    assert_type(
        cutlass_coop.inclusive_scan(block, values), cutlass_coop.ThreadData[np.int32]
    )
    assert_type(
        cutlass_coop.exclusive_sum(block, values), cutlass_coop.ThreadData[np.int32]
    )
    assert_type(
        cutlass_coop.inclusive_sum(block, values), cutlass_coop.ThreadData[np.int32]
    )
    assert_type(
        cutlass_coop.scan(block, values, scan_op="max", initial_value=np.int32(-8)),
        cutlass_coop.ThreadData[np.int32],
    )
    assert_type(
        cutlass_coop.scan(block, values, scan_op=np.add),
        cutlass_coop.ThreadData[np.int32],
    )
    assert_type(
        cutlass_coop.exclusive_scan(
            block, scalar, scan_op=operator.mul, initial_value=1
        ),
        Uint32,
    )
    assert_type(
        cutlass_coop.inclusive_sum(block, values, algorithm="raking_memoize"),
        cutlass_coop.ThreadData[np.int32],
    )
    storage = cutlass_coop.TempStorage(sharing="exclusive")
    assert_type(
        cutlass_coop.scan(block, values, algorithm="warp_scans", temp_storage=storage),
        cutlass_coop.ThreadData[np.int32],
    )
    assert_type(
        cutlass_coop.scan(logical, scalar, valid_items=7, aggregate_output=aggregate),
        Uint32,
    )
    assert_type(
        cutlass_coop.exclusive_scan(
            logical, scalar, valid_items=7, aggregate_output=aggregate
        ),
        Uint32,
    )
    assert_type(
        cutlass_coop.inclusive_scan(
            logical, scalar, valid_items=7, aggregate_output=aggregate
        ),
        Uint32,
    )
    assert_type(
        cutlass_coop.exclusive_sum(
            logical, scalar, valid_items=7, aggregate_output=aggregate
        ),
        Uint32,
    )
    assert_type(
        cutlass_coop.inclusive_sum(
            logical, scalar, valid_items=7, aggregate_output=aggregate
        ),
        Uint32,
    )
    assert_type(common_coop.inclusive_sum(logical, scalar), Uint32)
    assert_type(cutlass_coop.inclusive_sum(common_coop.this_warp(), scalar), Uint32)
    assert_type(
        cutlass_coop.inclusive_sum(block, values.to_register_tensor()),
        cutlass_coop.ThreadData[Any],
    )
    assert_type(
        cutlass_coop.exclusive_sum(block, values.to_tensor_ssa()),
        cutlass_coop.ThreadData[Any],
    )


def check_cutlass_exchange_surface() -> None:
    block = cutlass_coop.this_block()
    values = cutlass_coop.ThreadData(3, np.float32)
    ranks = cutlass_coop.ThreadData(3, Int16)
    flags = cutlass_coop.ThreadData(3, Uint64)
    for mode in (
        "striped_to_blocked",
        "blocked_to_striped",
        "warp_striped_to_blocked",
        "blocked_to_warp_striped",
    ):
        assert_type(
            cutlass_coop.exchange(block, values, mode=mode),
            cutlass_coop.ThreadData[np.float32],
        )
        assert_type(
            cutlass_coop.exchange(block, values, mode=mode, warp_time_slicing=True),
            cutlass_coop.ThreadData[np.float32],
        )
    for scatter_mode in ("scatter_to_blocked", "scatter_to_striped"):
        assert_type(
            cutlass_coop.exchange(
                block, values, mode=scatter_mode, ranks=ranks, warp_time_slicing=True
            ),
            cutlass_coop.ThreadData[np.float32],
        )
    assert_type(
        cutlass_coop.exchange(
            block, values, mode="scatter_to_striped_guarded", ranks=ranks
        ),
        cutlass_coop.ThreadData[np.float32],
    )
    assert_type(
        cutlass_coop.exchange(
            block,
            values,
            mode="scatter_to_striped_flagged",
            ranks=ranks,
            valid_flags=flags,
        ),
        cutlass_coop.ThreadData[np.float32],
    )
    assert_type(
        cutlass_coop.exchange(cutlass_coop.this_warp(), values),
        cutlass_coop.ThreadData[np.float32],
    )
    for width in (1, 2, 4, 8, 16, 32):
        logical = cutlass_coop.this_warp().group_by(width)
        assert_type(
            cutlass_coop.exchange(logical, values), cutlass_coop.ThreadData[np.float32]
        )
    assert_type(
        cutlass_coop.exchange(common_coop.this_block(), values),
        cutlass_coop.ThreadData[np.float32],
    )
    common_coop.exchange(block, values)
    assert_type(
        cutlass_coop.exchange(block, values.to_tensor_ssa()),
        cutlass_coop.ThreadData[Any],
    )
    assert_type(
        cutlass_coop.exchange(
            block,
            values.to_register_tensor(),
            mode="scatter_to_striped_flagged",
            ranks=ranks.to_tensor_ssa(),
            valid_flags=flags.to_register_tensor(),
        ),
        cutlass_coop.ThreadData[Any],
    )
    assert_type(
        cutlass_coop.exchange(
            block,
            values,
            mode="scatter_to_blocked",
            ranks=cutlass_coop.ThreadData(3, np.int8),
        ),
        cutlass_coop.ThreadData[np.float32],
    )


def check_cutlass_shuffle_surface(scalar: Uint32) -> None:
    block = cutlass_coop.this_block()
    values = cutlass_coop.ThreadData(3, np.int32)
    assert_type(cutlass_coop.shuffle(block, values), cutlass_coop.ThreadData[np.int32])
    assert_type(
        cutlass_coop.shuffle(block, values, mode="up"),
        cutlass_coop.ThreadData[np.int32],
    )
    assert_type(cutlass_coop.shuffle(block, scalar, mode="offset", distance=-2), Uint32)
    assert_type(
        cutlass_coop.shuffle(block, scalar, mode="rotate", distance=Int16(2)), Uint32
    )
    assert_type(
        cutlass_coop.shuffle(block, scalar, mode="rotate", distance=np.uint32(2)),
        Uint32,
    )
    assert_type(
        cutlass_coop.shuffle(common_coop.this_block(), values),
        cutlass_coop.ThreadData[np.int32],
    )
    common_coop.shuffle(block, values)
    assert_type(
        cutlass_coop.shuffle(block, values.to_register_tensor()),
        cutlass_coop.ThreadData[Any],
    )
    assert_type(
        cutlass_coop.shuffle(block, values.to_tensor_ssa(), mode="up"),
        cutlass_coop.ThreadData[Any],
    )


class _ReadOnlyKeys:
    items_per_thread: int = 2
    dtype: object | None = np.int32

    def __len__(self) -> int:
        return self.items_per_thread

    def __getitem__(self, index: int, /) -> np.int32:
        return np.int32(index)


def check_cutlass_merge_sort_surface() -> None:
    block = cutlass_coop.this_block()
    warp = cutlass_coop.this_warp()
    logical = warp.group_by(8)
    keys = cutlass_coop.ThreadData(3, np.int32)
    values = cutlass_coop.ThreadData(3, np.float64)
    storage = cutlass_coop.TempStorage(alignment=32)
    assert_type(
        cutlass_coop.merge_sort_keys(block, keys), cutlass_coop.ThreadData[np.int32]
    )
    assert_type(
        cutlass_coop.merge_sort_pairs(block, keys, values, temp_storage=storage),
        tuple[cutlass_coop.ThreadData[np.int32], cutlass_coop.ThreadData[np.float64]],
    )
    assert_type(
        cutlass_coop.merge_sort_keys(
            block,
            keys,
            valid_items=Uint32(63),
            oob_default=1000,
            temp_storage=storage,
        ),
        cutlass_coop.ThreadData[np.int32],
    )
    assert_type(
        cutlass_coop.merge_sort_pairs(
            warp,
            keys,
            values,
            descending=True,
            valid_items=np.int64(31),
            oob_default=-1000,
        ),
        tuple[cutlass_coop.ThreadData[np.int32], cutlass_coop.ThreadData[np.float64]],
    )
    assert_type(
        cutlass_coop.merge_sort_keys(
            logical, keys, valid_items=23, oob_default=np.int32(1000)
        ),
        cutlass_coop.ThreadData[np.int32],
    )
    assert_type(
        cutlass_coop.merge_sort_keys(warp, _ReadOnlyKeys()),
        cutlass_coop.ThreadData[np.int32],
    )
    assert_type(
        cutlass_coop.merge_sort_pairs(logical, _ReadOnlyKeys(), _ReadOnlyKeys()),
        tuple[cutlass_coop.ThreadData[np.int32], cutlass_coop.ThreadData[np.int32]],
    )
    assert_type(
        cutlass_coop.merge_sort_keys(common_coop.this_block(), keys),
        cutlass_coop.ThreadData[np.int32],
    )
    assert_type(
        common_coop.merge_sort_pairs(block, keys, values),
        tuple[
            common_coop.ThreadDataLike[np.int32], common_coop.ThreadDataLike[np.float64]
        ],
    )
    assert_type(
        cutlass_coop.merge_sort_keys(block, keys.to_register_tensor()),
        cutlass_coop.ThreadData[Any],
    )
    assert_type(
        cutlass_coop.merge_sort_pairs(warp, keys.to_tensor_ssa(), values),
        tuple[cutlass_coop.ThreadData[Any], cutlass_coop.ThreadData[np.float64]],
    )
    assert_type(
        cutlass_coop.merge_sort_pairs(logical, keys, values.to_register_tensor()),
        tuple[cutlass_coop.ThreadData[np.int32], cutlass_coop.ThreadData[Any]],
    )
    assert_type(
        cutlass_coop.merge_sort_pairs(
            block,
            keys.to_register_tensor(),
            values.to_tensor_ssa(),
            valid_items=95,
            oob_default=np.int32(1000),
        ),
        tuple[cutlass_coop.ThreadData[Any], cutlass_coop.ThreadData[Any]],
    )


def check_cutlass_radix_surface(scalar: Float32, value: Int16) -> None:
    block = cutlass_coop.this_block()
    keys = cutlass_coop.ThreadData(3, np.int32)
    values = cutlass_coop.ThreadData(3, np.float64)
    prefixes = cutlass_coop.ThreadData(1, Int32)
    storage = cutlass_coop.TempStorage(alignment=32)
    assert_type(
        cutlass_coop.radix_sort_keys(block, keys), cutlass_coop.ThreadData[np.int32]
    )
    assert_type(
        cutlass_coop.radix_sort_pairs(
            block,
            keys,
            values,
            begin_bit=Int64(3),
            end_bit=Uint32(8),
            descending=True,
            temp_storage=storage,
            blocked_to_striped=True,
        ),
        tuple[cutlass_coop.ThreadData[np.int32], cutlass_coop.ThreadData[np.float64]],
    )
    assert_type(cutlass_coop.radix_sort_keys(block, scalar), Float32)
    assert_type(
        cutlass_coop.radix_sort_pairs(block, scalar, value), tuple[Float32, Int16]
    )
    assert_type(
        cutlass_coop.radix_sort_keys(block, cutlass_coop.ThreadData(2, Float64)),
        cutlass_coop.ThreadData[Float64],
    )
    assert_type(
        cutlass_coop.radix_sort_pairs(block, Uint64(1), np.uint8(2)),
        tuple[Uint64, np.uint8],
    )
    assert_type(
        cutlass_coop.radix_sort_keys(block, _ReadOnlyKeys()),
        cutlass_coop.ThreadData[np.int32],
    )
    assert_type(
        cutlass_coop.radix_sort_pairs(block, _ReadOnlyKeys(), _ReadOnlyKeys()),
        tuple[cutlass_coop.ThreadData[np.int32], cutlass_coop.ThreadData[np.int32]],
    )
    assert_type(
        cutlass_coop.radix_sort_keys(block, keys.to_tensor_ssa()),
        cutlass_coop.ThreadData[Any],
    )
    assert_type(
        cutlass_coop.radix_sort_pairs(block, keys.to_register_tensor(), values),
        tuple[cutlass_coop.ThreadData[Any], cutlass_coop.ThreadData[np.float64]],
    )
    assert_type(
        cutlass_coop.radix_sort_pairs(block, keys, values.to_tensor_ssa()),
        tuple[cutlass_coop.ThreadData[np.int32], cutlass_coop.ThreadData[Any]],
    )
    assert_type(
        cutlass_coop.radix_sort_pairs(
            block,
            keys.to_tensor_ssa(),
            values.to_register_tensor(),
        ),
        tuple[cutlass_coop.ThreadData[Any], cutlass_coop.ThreadData[Any]],
    )
    assert_type(
        cutlass_coop.radix_rank(
            block,
            keys,
            begin_bit=4,
            radix_bits=4,
            descending=True,
            exclusive_digit_prefix=prefixes,
        ),
        cutlass_coop.ThreadData[Int32],
    )
    assert_type(
        cutlass_coop.radix_rank(block, _ReadOnlyKeys()), cutlass_coop.ThreadData[Int32]
    )
    assert_type(
        cutlass_coop.radix_rank(block, keys.to_register_tensor()),
        cutlass_coop.ThreadData[Int32],
    )
    assert_type(
        cutlass_coop.radix_rank(block, keys.to_tensor_ssa()),
        cutlass_coop.ThreadData[Int32],
    )
    assert_type(
        cutlass_coop.radix_rank(
            block,
            Int64(1),
            end_bit=8,
            exclusive_digit_prefix=cutlass_coop.ThreadData(4, np.int32),
        ),
        Int32,
    )
    assert_type(
        common_coop.radix_sort_keys(block, keys), common_coop.ThreadDataLike[np.int32]
    )
    assert_type(
        common_coop.radix_rank(block, keys), common_coop.ThreadDataLike[np.int32]
    )
    assert_type(
        cutlass_coop.radix_sort_keys(common_coop.this_block(), keys),
        cutlass_coop.ThreadData[np.int32],
    )


def check_cutlass_topk_surface() -> None:
    block = cutlass_coop.this_block()
    keys = cutlass_coop.ThreadData(3, np.float32)
    values = cutlass_coop.ThreadData(3, Int16)
    storage = cutlass_coop.TempStorage(alignment=32)
    assert_type(
        cutlass_coop.topk_min_keys(block, keys, k=7),
        cutlass_coop.ThreadData[np.float32],
    )
    assert_type(
        cutlass_coop.topk_max_keys(block, keys, k=Int64(7), valid_items=Uint32(63)),
        cutlass_coop.ThreadData[np.float32],
    )
    assert_type(
        cutlass_coop.topk_min_pairs(block, keys, values, k=7, temp_storage=storage),
        tuple[cutlass_coop.ThreadData[np.float32], cutlass_coop.ThreadData[Int16]],
    )
    assert_type(
        cutlass_coop.topk_max_pairs(
            block, keys, values, k=np.uint16(7), valid_items=31, temp_storage=storage
        ),
        tuple[cutlass_coop.ThreadData[np.float32], cutlass_coop.ThreadData[Int16]],
    )
    assert_type(
        cutlass_coop.topk_min_keys(block, _ReadOnlyKeys(), k=0),
        cutlass_coop.ThreadData[np.int32],
    )
    assert_type(
        cutlass_coop.topk_max_pairs(block, _ReadOnlyKeys(), _ReadOnlyKeys(), k=2),
        tuple[cutlass_coop.ThreadData[np.int32], cutlass_coop.ThreadData[np.int32]],
    )
    assert_type(
        cutlass_coop.topk_min_keys(block, keys.to_register_tensor(), k=7),
        cutlass_coop.ThreadData[Any],
    )
    assert_type(
        cutlass_coop.topk_max_keys(block, keys.to_tensor_ssa(), k=7),
        cutlass_coop.ThreadData[Any],
    )
    assert_type(
        cutlass_coop.topk_min_pairs(block, keys.to_tensor_ssa(), values, k=7),
        tuple[cutlass_coop.ThreadData[Any], cutlass_coop.ThreadData[Int16]],
    )
    assert_type(
        cutlass_coop.topk_max_pairs(block, keys, values.to_register_tensor(), k=7),
        tuple[cutlass_coop.ThreadData[np.float32], cutlass_coop.ThreadData[Any]],
    )
    assert_type(
        cutlass_coop.topk_min_pairs(
            block, keys.to_register_tensor(), values.to_tensor_ssa(), k=7
        ),
        tuple[cutlass_coop.ThreadData[Any], cutlass_coop.ThreadData[Any]],
    )
    assert_type(
        common_coop.topk_min_keys(block, keys, k=7),
        common_coop.ThreadDataLike[np.float32],
    )
    assert_type(
        common_coop.topk_max_pairs(block, keys, values, k=7),
        tuple[
            common_coop.ThreadDataLike[np.float32], common_coop.ThreadDataLike[Int16]
        ],
    )
    assert_type(
        cutlass_coop.topk_max_keys(common_coop.this_block(), keys, k=7),
        cutlass_coop.ThreadData[np.float32],
    )
