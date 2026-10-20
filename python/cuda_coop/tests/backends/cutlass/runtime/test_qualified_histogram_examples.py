# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check qualified histogram examples against host results."""

import numpy as np
import pytest

pytest.importorskip("cutlass")

from tests.backends.cutlass.support import device_array

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.runtime, pytest.mark.gpu]


@pytest.mark.parametrize("items_per_thread", [1, 4])
def test_histogram_example(items_per_thread):
    # qualified-histogram-example-begin
    import cutlass
    from cutlass import cute

    import cuda.coop.cutlass as coop

    @cute.kernel
    def count_samples(
        source: cute.Pointer,
        atomic_output: cute.Pointer,
        sort_output: cute.Pointer,
        items_per_thread: cutlass.Constexpr,
    ):
        block = coop.this_block()
        samples = coop.ThreadData(items_per_thread)
        coop.load(block, source, samples)
        atomic_counts = coop.histogram(
            block, samples, bins=70, bins_per_thread=2
        )
        sort_counts = coop.histogram(
            block, samples, bins=70, bins_per_thread=2, algorithm="sort"
        )
        coop.store(block, atomic_output, atomic_counts, algorithm="striped")
        coop.store(block, sort_output, sort_counts, algorithm="striped")

    @cute.jit
    def launch(
        source: cute.Pointer,
        atomic_output: cute.Pointer,
        sort_output: cute.Pointer,
        items_per_thread: cutlass.Constexpr,
    ):
        count_samples(
            source, atomic_output, sort_output, items_per_thread
        ).launch(grid=1, block=64)

    # qualified-histogram-example-end
    values = (np.arange(64 * items_per_thread, dtype=np.int32) * 13) % 70
    atomic, sorted_counts = (np.full(128, -1, dtype=np.int32) for _ in range(2))
    with (
        device_array(values) as src,
        device_array(atomic) as first,
        device_array(sorted_counts) as second,
    ):
        launch(src, first, second, items_per_thread)
    expected = np.bincount(values, minlength=128)
    np.testing.assert_array_equal(atomic, expected)
    np.testing.assert_array_equal(sorted_counts, expected)
