# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Run the qualified block and partial-warp scan documentation examples.

Host checks distinguish seeded prefixes from the unseeded warp aggregate,
check block sums and running maxima, and preserve undefined-lane sentinels.
"""

import numpy as np
import pytest

pytest.importorskip("cutlass")

from tests.backends.cutlass.support import device_array

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.runtime, pytest.mark.gpu]


def test_partial_warp_scan_example():
    """Check a seeded five-lane prefix within complete eight-lane groups.

    All lanes participate and record the aggregate of the five valid inputs.
    Only the first five use their primary Scan result. The aggregate excludes
    the seed, although the exclusive results start from seven.
    """

    # qualified-exclusive-scan-example-begin
    import operator

    from cutlass import cute

    import cuda.coop.cutlass as cutlass_coop

    @cute.kernel
    def scan_prefix(
        source: cute.Pointer,
        destination: cute.Pointer,
        aggregates: cute.Pointer,
    ):
        thread = cutlass_coop.this_block().rank()
        group = cutlass_coop.this_warp().group_by(8)
        inputs = cute.make_tensor(source, cute.make_layout(64))
        outputs = cute.make_tensor(destination, cute.make_layout(64))
        totals = cute.make_tensor(aggregates, cute.make_layout(64))
        aggregate = cutlass_coop.ThreadData(items_per_thread=1)
        result = cutlass_coop.exclusive_scan(
            group,
            inputs[thread],
            scan_op=operator.add,
            initial_value=7,
            valid_items=5,
            aggregate_output=aggregate,
        )
        if group.rank() < 5:
            outputs[thread] = result
        totals[thread] = aggregate[0]

    @cute.jit
    def launch(
        source: cute.Pointer,
        destination: cute.Pointer,
        aggregates: cute.Pointer,
    ):
        scan_prefix(source, destination, aggregates).launch(grid=1, block=64)

    # qualified-exclusive-scan-example-end

    values = np.arange(64, dtype=np.int32)
    observed = np.full_like(values, -101)
    aggregates = np.zeros_like(values)
    expected = observed.copy().reshape(-1, 8)
    valid_values = values.reshape(-1, 8)[:, :5]
    expected[:, 0] = 7
    expected[:, 1:5] = 7 + np.cumsum(
        valid_values[:, :-1], axis=1, dtype=np.int32
    )
    expected_aggregates = np.repeat(valid_values.sum(axis=1, dtype=np.int32), 8)
    with (
        device_array(values) as source,
        device_array(observed) as destination,
        device_array(aggregates) as totals,
    ):
        launch(source, destination, totals)
    np.testing.assert_array_equal(observed, expected.reshape(-1))
    np.testing.assert_array_equal(aggregates, expected_aggregates)


@pytest.mark.parametrize("items_per_thread", [1, 4])
def test_scan_example(items_per_thread):
    # qualified-scan-example-begin
    import cutlass
    from cutlass import cute

    import cuda.coop.cutlass as coop

    @cute.kernel
    def prefixes(
        source: cute.Pointer,
        seeded: cute.Pointer,
        maxima: cute.Pointer,
        exclusive: cute.Pointer,
        inclusive: cute.Pointer,
        items_per_thread: cutlass.Constexpr,
    ):
        block = coop.this_block()
        values = coop.ThreadData(items_per_thread)
        coop.load(block, source, values)
        scratch = coop.TempStorage(auto_sync=True)
        seeded_sums = coop.scan(
            block, values, initial_value=7, temp_storage=scratch
        )
        running_maxima = coop.inclusive_scan(
            block, values, scan_op="max", temp_storage=scratch
        )
        before = coop.exclusive_sum(block, values, temp_storage=scratch)
        through = coop.inclusive_sum(block, values, temp_storage=scratch)
        coop.store(block, seeded, seeded_sums)
        coop.store(block, maxima, running_maxima)
        coop.store(block, exclusive, before)
        coop.store(block, inclusive, through)

    @cute.jit
    def launch(
        source: cute.Pointer,
        seeded: cute.Pointer,
        maxima: cute.Pointer,
        exclusive: cute.Pointer,
        inclusive: cute.Pointer,
        items_per_thread: cutlass.Constexpr,
    ):
        prefixes(
            source, seeded, maxima, exclusive, inclusive, items_per_thread
        ).launch(grid=1, block=64)

    # qualified-scan-example-end
    values = (np.arange(64 * items_per_thread, dtype=np.int32) * 13) % 31 - 15
    seeded, maxima, exclusive, inclusive = (
        np.zeros_like(values) for _ in range(4)
    )
    with (
        device_array(values) as src,
        device_array(seeded) as seed,
        device_array(maxima) as maximum,
        device_array(exclusive) as before,
        device_array(inclusive) as through,
    ):
        launch(src, seed, maximum, before, through, items_per_thread)
    expected = values.cumsum(dtype=np.int32)
    np.testing.assert_array_equal(seeded, expected - values + 7)
    np.testing.assert_array_equal(maxima, np.maximum.accumulate(values))
    np.testing.assert_array_equal(exclusive, expected - values)
    np.testing.assert_array_equal(inclusive, expected)
