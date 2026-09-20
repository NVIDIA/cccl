# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

"""Executable examples included in the qualified CUTLASS API reference."""

import numpy as np
import pytest

pytest.importorskip("cutlass")

from tests.backends.cutlass.support import device_array

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.runtime, pytest.mark.gpu]


def test_partial_warp_scan_example():
    # qualified-exclusive-scan-example-begin
    import operator

    from cutlass import cute

    import cuda.coop.cutlass as cutlass_coop

    @cute.kernel
    def scan_prefix(
        source: cute.Pointer, destination: cute.Pointer, aggregates: cute.Pointer
    ):
        thread = cutlass_coop.this_block().rank()
        group = cutlass_coop.this_warp().group_by(8)
        inputs = cute.make_tensor(source, cute.make_layout(64))
        outputs = cute.make_tensor(destination, cute.make_layout(64))
        totals = cute.make_tensor(aggregates, cute.make_layout(64))
        aggregate = cutlass_coop.ThreadData(1)
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
        source: cute.Pointer, destination: cute.Pointer, aggregates: cute.Pointer
    ):
        scan_prefix(source, destination, aggregates).launch(grid=1, block=64)

    # qualified-exclusive-scan-example-end

    values = np.arange(64, dtype=np.int32)
    observed = np.full_like(values, -101)
    aggregates = np.zeros_like(values)
    expected = observed.copy().reshape(-1, 8)
    valid_values = values.reshape(-1, 8)[:, :5]
    expected[:, 0] = 7
    expected[:, 1:5] = 7 + np.cumsum(valid_values[:, :-1], axis=1, dtype=np.int32)
    expected_aggregates = np.repeat(valid_values.sum(axis=1, dtype=np.int32), 8)
    with (
        device_array(values) as source,
        device_array(observed) as destination,
        device_array(aggregates) as totals,
    ):
        launch(source, destination, totals)
    np.testing.assert_array_equal(observed, expected.reshape(-1))
    np.testing.assert_array_equal(aggregates, expected_aggregates)


def test_scatter_example():
    # qualified-scatter-example-begin
    import cutlass
    from cutlass import cute

    import cuda.coop.cutlass as cutlass_coop

    @cute.kernel
    def reverse_tile(source: cute.Pointer, destination: cute.Pointer):
        block = cutlass_coop.this_block()
        thread = block.rank()
        items = cutlass_coop.ThreadData(2, dtype=cutlass.Int32)
        ranks = cutlass_coop.ThreadData(2, dtype=cutlass.Int32)
        cutlass_coop.load(block, source, items)
        for item in cutlass.range_constexpr(2):
            ranks[item] = cutlass.Int32(127 - (thread * 2 + item))
        reversed_items = cutlass_coop.exchange(
            block, items, mode="scatter_to_blocked", ranks=ranks
        )
        cutlass_coop.store(block, destination, reversed_items)

    @cute.jit
    def launch(source: cute.Pointer, destination: cute.Pointer):
        reverse_tile(source, destination).launch(grid=1, block=64)

    # qualified-scatter-example-end

    values = np.arange(128, dtype=np.int32) * 3 - 200
    observed = np.zeros_like(values)
    with device_array(values) as source, device_array(observed) as destination:
        launch(source, destination)
    np.testing.assert_array_equal(observed, values[::-1])


def test_rotate_example():
    # qualified-rotate-example-begin
    from cutlass import cute

    import cuda.coop.cutlass as cutlass_coop

    @cute.kernel
    def rotate_tile(source: cute.Pointer, destination: cute.Pointer):
        block = cutlass_coop.this_block()
        thread = block.rank()
        inputs = cute.make_tensor(source, cute.make_layout(64))
        outputs = cute.make_tensor(destination, cute.make_layout(64))
        outputs[thread] = cutlass_coop.shuffle(
            block, inputs[thread], mode="rotate", distance=7
        )

    @cute.jit
    def launch(source: cute.Pointer, destination: cute.Pointer):
        rotate_tile(source, destination).launch(grid=1, block=64)

    # qualified-rotate-example-end

    values = np.arange(64, dtype=np.int32) * 3 - 200
    observed = np.zeros_like(values)
    with device_array(values) as source, device_array(observed) as destination:
        launch(source, destination)
    np.testing.assert_array_equal(observed, np.roll(values, -7))
