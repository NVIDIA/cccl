# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Executable examples included in the qualified CUTLASS API reference."""

import numpy as np
import pytest

pytest.importorskip("cutlass")

from tests.backends.cutlass.support import device_array

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.runtime, pytest.mark.gpu]


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
