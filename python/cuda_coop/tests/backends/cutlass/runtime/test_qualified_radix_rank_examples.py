# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check qualified radix rank examples against host results."""

import numpy as np
import pytest

pytest.importorskip("cutlass")

from tests.backends.cutlass.support import device_array

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.runtime, pytest.mark.gpu]


@pytest.mark.parametrize("items_per_thread", [1, 4])
def test_radix_rank_example(items_per_thread):
    # qualified-radix-rank-example-begin
    import cutlass
    from cutlass import cute

    import cuda.coop.cutlass as coop

    @cute.kernel
    def rank_digits(
        source: cute.Pointer,
        ranks_output: cute.Pointer,
        prefixes_output: cute.Pointer,
        items_per_thread: cutlass.Constexpr,
    ):
        block = coop.this_block()
        keys = coop.ThreadData(items_per_thread)
        coop.load(block, source, keys)
        prefixes = coop.ThreadData(items_per_thread=1)
        ranks = coop.radix_rank_keys(
            block, keys, radix_bits=4, exclusive_digit_prefix=prefixes
        )
        coop.store(block, ranks_output, ranks)
        coop.store(block, prefixes_output, prefixes, valid_items=16)

    @cute.jit
    def launch(
        source: cute.Pointer,
        ranks_output: cute.Pointer,
        prefixes_output: cute.Pointer,
        items_per_thread: cutlass.Constexpr,
    ):
        rank_digits(
            source, ranks_output, prefixes_output, items_per_thread
        ).launch(grid=1, block=64)

    # qualified-radix-rank-example-end
    values = (np.arange(64 * items_per_thread, dtype=np.int32) * 13) % 127
    ranks = np.zeros_like(values)
    prefixes = np.zeros(16, dtype=np.int32)
    with (
        device_array(values) as src,
        device_array(ranks) as rank,
        device_array(prefixes) as prefix,
    ):
        launch(src, rank, prefix, items_per_thread)
    order = np.argsort(values & 15, kind="stable")
    np.testing.assert_array_equal(ranks[order], np.arange(len(values)))
    counts = np.bincount(values & 15, minlength=16)
    np.testing.assert_array_equal(prefixes, counts.cumsum() - counts)
