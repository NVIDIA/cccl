# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Run the qualified scatter and rotate examples used in the documentation.

Marked regions remain suitable for direct inclusion in the public guide.
Host oracles outside those regions compare the complete reversed tile or
rotation, covering the backend-specific modes shown in each example.
"""

import numpy as np
import pytest

pytest.importorskip("cutlass")

from tests.backends.cutlass.support import device_array

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.runtime, pytest.mark.gpu]


@pytest.mark.parametrize("items_per_thread", (1, 4))
def test_scatter_example(items_per_thread):
    """Reverse a tile with a unique destination rank for every input element.

    Host reversal checks the rank permutation across all threads and items,
    including the one-item case. It does not rely on an inverse Exchange call.
    """

    # qualified-scatter-example-begin
    import cutlass
    from cutlass import cute

    import cuda.coop.cutlass as cutlass_coop

    @cute.kernel
    def reverse_tile(
        source: cute.Pointer,
        destination: cute.Pointer,
        items_per_thread: cutlass.Constexpr,
    ):
        block = cutlass_coop.this_block()
        thread = block.rank()
        items = cutlass_coop.ThreadData(items_per_thread)
        ranks = cutlass_coop.ThreadData(items_per_thread)
        cutlass_coop.load(block, source, items)
        for item in cutlass.range_constexpr(items_per_thread):
            ranks[item] = cutlass.Int32(
                64 * items_per_thread - 1 - (thread * items_per_thread + item)
            )
        reversed_items = cutlass_coop.exchange(
            block, items, mode="scatter_to_blocked", ranks=ranks
        )
        cutlass_coop.store(block, destination, reversed_items)

    @cute.jit
    def launch(
        source: cute.Pointer,
        destination: cute.Pointer,
        items_per_thread: cutlass.Constexpr,
    ):
        reverse_tile(source, destination, items_per_thread).launch(
            grid=1, block=64
        )

    # qualified-scatter-example-end

    values = np.arange(64 * items_per_thread, dtype=np.int32) * 3 - 200
    observed = np.zeros_like(values)
    with device_array(values) as source, device_array(observed) as destination:
        launch(source, destination, items_per_thread)
    np.testing.assert_array_equal(observed, values[::-1])


def test_rotate_example():
    """Rotate scalar values across the block and compare with a host rotation.

    Each output reads seven threads ahead with wraparound. This checks the
    direction for a positive distance and the wraparound at the tile boundary.
    """

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
