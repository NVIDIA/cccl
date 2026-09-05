# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Strict consumer of every qualified Numba-CUDA-MLIR family."""

from __future__ import annotations

from typing import Literal

from typing_extensions import assert_type

import cuda.coop.numba_mlir as coop
from cuda.coop import ThreadDataLike


def prefix_from_aggregate(aggregate: int) -> int:
    """Return a stateless prefix for one block aggregate."""

    return aggregate


class RunningPrefix:
    """Model a stateful device callback used across block tiles."""

    def __call__(self, aggregate: int) -> int:
        """Return the prior prefix represented by ``state``."""

        return aggregate


running_prefix = coop.StatefulFunction(
    RunningPrefix,
    int,
    name="running_prefix",
)
assert_type(running_prefix, coop.StatefulFunction[type[RunningPrefix]])


def check_numba_surface(source: object, destination: object) -> None:
    """Exercise Numba declarations through their public package."""

    block = coop.this_block()
    warp = coop.this_warp()
    keys = coop.ThreadData(2, int)
    values = coop.ThreadData(2, float)
    lengths = coop.ThreadData(2, int)
    prefix_state = coop.ThreadData(1, int)
    storage = coop.TempStorage()

    assert_type(block, coop.ThreadGroup[Literal["block"]])
    assert_type(warp, coop.ThreadGroup[Literal["warp"]])
    assert_type(keys, ThreadDataLike[int])
    assert_type(values, ThreadDataLike[float])
    assert_type(storage, coop.TempStorage)

    assert_type(
        coop.load(block, source, values, temp_storage=storage),
        ThreadDataLike[float],
    )
    coop.store(block, destination, values, temp_storage=storage)
    assert_type(coop.exchange(warp, values), ThreadDataLike[float])

    assert_type(coop.reduce(block, values), float)
    assert_type(coop.sum(warp, 1), int)
    assert_type(
        coop.scan(block, values, temp_storage=storage),
        ThreadDataLike[float],
    )
    assert_type(
        coop.scan(block, keys, prefix_op=prefix_from_aggregate),
        ThreadDataLike[int],
    )
    assert_type(
        coop.scan(
            block,
            keys,
            prefix_state,
            prefix_op=running_prefix,
        ),
        ThreadDataLike[int],
    )
    assert_type(
        coop.scan(
            block,
            keys,
            block_prefix_callback_op=prefix_from_aggregate,
        ),
        ThreadDataLike[int],
    )
    assert_type(
        coop.exclusive_sum(block, values, temp_storage=storage),
        ThreadDataLike[float],
    )
    assert_type(
        coop.inclusive_sum(block, values, temp_storage=storage),
        ThreadDataLike[float],
    )
    assert_type(
        coop.exclusive_scan(block, values, temp_storage=storage),
        ThreadDataLike[float],
    )
    assert_type(
        coop.inclusive_scan(block, values, temp_storage=storage),
        ThreadDataLike[float],
    )

    assert_type(
        coop.adjacent_difference(block, values, temp_storage=storage),
        ThreadDataLike[float],
    )
    assert_type(
        coop.discontinuity(block, values, temp_storage=storage),
        ThreadDataLike[int],
    )
    assert_type(coop.shuffle(block, values), ThreadDataLike[float])

    assert_type(
        coop.histogram(block, keys, bins=4, bins_per_thread=1),
        ThreadDataLike[int],
    )
    assert_type(
        coop.merge_sort_keys(block, keys, temp_storage=storage),
        ThreadDataLike[int],
    )
    assert_type(
        coop.merge_sort_pairs(block, keys, values, temp_storage=storage),
        tuple[ThreadDataLike[int], ThreadDataLike[float]],
    )
    assert_type(
        coop.radix_sort_keys(block, keys, temp_storage=storage),
        ThreadDataLike[int],
    )
    assert_type(
        coop.radix_sort_pairs(block, keys, values, temp_storage=storage),
        tuple[ThreadDataLike[int], ThreadDataLike[float]],
    )
    assert_type(coop.radix_rank(block, keys), ThreadDataLike[int])
    assert_type(
        coop.run_length_decode(
            block,
            keys,
            lengths,
            decoded_items_per_thread=2,
        ),
        ThreadDataLike[int],
    )
    assert_type(
        coop.topk_max_keys(block, keys, 1, temp_storage=storage),
        ThreadDataLike[int],
    )
    assert_type(
        coop.topk_min_keys(block, keys, 1, temp_storage=storage),
        ThreadDataLike[int],
    )
    assert_type(
        coop.topk_max_pairs(block, keys, values, 1, temp_storage=storage),
        tuple[ThreadDataLike[int], ThreadDataLike[float]],
    )
    assert_type(
        coop.topk_min_pairs(block, keys, values, 1, temp_storage=storage),
        tuple[ThreadDataLike[int], ThreadDataLike[float]],
    )

    block_load: coop.BlockLoadAlgorithm = coop.BlockLoadAlgorithm.DIRECT
    block_store: coop.BlockStoreAlgorithm = coop.BlockStoreAlgorithm.DIRECT
    histogram: coop.BlockHistogramAlgorithm = coop.BlockHistogramAlgorithm.SORT
    block_scan: coop.BlockScanAlgorithm = coop.BlockScanAlgorithm.RAKING
    warp_load: coop.WarpLoadAlgorithm = coop.WarpLoadAlgorithm.DIRECT
    warp_store: coop.WarpStoreAlgorithm = coop.WarpStoreAlgorithm.DIRECT
    assert_type(block_load, coop.BlockLoadAlgorithm)
    assert_type(block_store, coop.BlockStoreAlgorithm)
    assert_type(histogram, coop.BlockHistogramAlgorithm)
    assert_type(block_scan, coop.BlockScanAlgorithm)
    assert_type(warp_load, coop.WarpLoadAlgorithm)
    assert_type(warp_store, coop.WarpStoreAlgorithm)
