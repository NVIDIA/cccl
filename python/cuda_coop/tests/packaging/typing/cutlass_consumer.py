# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Strict consumer of every qualified CUTLASS primitive family."""

from __future__ import annotations

from typing import Literal

from typing_extensions import assert_type

import cuda.coop.cutlass as coop


class StringSource:
    """Produce a string payload through the public load-source protocol."""

    def __cuda_coop_thread_data_load__(self) -> coop.ThreadData[str]:
        """Provide a typed payload to ``ThreadData.load``."""

        raise NotImplementedError


string_source: coop.ThreadDataLoadSource[str] = StringSource()
covariant_source: coop.ThreadDataLoadSource[object] = string_source
assert_type(coop.ThreadData.load(string_source), coop.ThreadData[str])


def check_cutlass_surface(source: object, destination: object) -> None:
    """Exercise CUTLASS declarations through their public package."""

    block = coop.this_block()
    warp = coop.this_warp()
    keys = coop.ThreadData(2, int)
    values = coop.ThreadData(2, float)
    lengths = coop.ThreadData(2, int)
    storage = coop.TempStorage()

    assert_type(block, coop.ThreadGroup[Literal["block"]])
    assert_type(warp, coop.ThreadGroup[Literal["warp"]])
    assert_type(keys, coop.ThreadData[int])
    assert_type(values, coop.ThreadData[float])
    assert_type(storage, coop.TempStorage)

    assert_type(
        coop.load(block, source, values, temp_storage=storage),
        coop.ThreadData[float],
    )
    coop.store(block, destination, values, temp_storage=storage)
    assert_type(coop.exchange(warp, values), coop.ThreadData[float])

    assert_type(coop.reduce(block, values), float)
    assert_type(coop.sum(warp, 1), int)
    assert_type(
        coop.scan(block, values, temp_storage=storage),
        coop.ThreadData[float],
    )
    assert_type(
        coop.exclusive_sum(block, values, temp_storage=storage),
        coop.ThreadData[float],
    )
    assert_type(
        coop.inclusive_sum(block, values, temp_storage=storage),
        coop.ThreadData[float],
    )
    assert_type(
        coop.exclusive_scan(block, values, temp_storage=storage),
        coop.ThreadData[float],
    )
    assert_type(
        coop.inclusive_scan(block, values, temp_storage=storage),
        coop.ThreadData[float],
    )

    assert_type(
        coop.adjacent_difference(block, values, temp_storage=storage),
        coop.ThreadData[float],
    )
    assert_type(
        coop.discontinuity(block, values, temp_storage=storage),
        coop.ThreadData[int],
    )
    assert_type(coop.shuffle(block, values), coop.ThreadData[float])

    assert_type(
        coop.histogram(block, 1, bins=4, bins_per_thread=1),
        coop.ThreadData[int],
    )
    assert_type(
        coop.merge_sort_keys(block, keys, temp_storage=storage),
        coop.ThreadData[int],
    )
    assert_type(
        coop.merge_sort_pairs(block, keys, values, temp_storage=storage),
        tuple[coop.ThreadData[int], coop.ThreadData[float]],
    )
    assert_type(
        coop.radix_sort_keys(block, keys, temp_storage=storage),
        coop.ThreadData[int],
    )
    assert_type(
        coop.radix_sort_pairs(block, keys, values, temp_storage=storage),
        tuple[coop.ThreadData[int], coop.ThreadData[float]],
    )
    assert_type(coop.radix_rank(block, keys), coop.ThreadData[int])
    assert_type(
        coop.run_length_decode(
            block,
            keys,
            lengths,
            decoded_items_per_thread=2,
        ),
        coop.ThreadData[int],
    )
    assert_type(
        coop.topk_max_keys(block, keys, 1, temp_storage=storage),
        coop.ThreadData[int],
    )
    assert_type(
        coop.topk_min_keys(block, keys, 1, temp_storage=storage),
        coop.ThreadData[int],
    )
    assert_type(
        coop.topk_max_pairs(block, keys, values, 1, temp_storage=storage),
        tuple[coop.ThreadData[int], coop.ThreadData[float]],
    )
    assert_type(
        coop.topk_min_pairs(block, keys, values, 1, temp_storage=storage),
        tuple[coop.ThreadData[int], coop.ThreadData[float]],
    )
