# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check load store examples against host results."""

import numpy as np
import pytest

from tests.runtime_support import device_array

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.runtime, pytest.mark.gpu]


@pytest.mark.parametrize("items_per_thread", [1, 4])
def test_load_store_example(items_per_thread):
    # cutlass-load-store-example-begin
    import cutlass
    from cutlass import cute

    import cuda.coop.cutlass as coop

    @cute.kernel
    def copy_prefix(
        source: cute.Pointer,
        destination: cute.Pointer,
        items_per_thread: cutlass.Constexpr,
    ):
        block = coop.this_block()
        valid_items = 64 * items_per_thread - 3
        values = coop.load(
            block,
            source,
            items_per_thread=items_per_thread,
            valid_items=valid_items,
            oob_default=0,
            offset=5,
        )
        coop.store(
            block,
            destination,
            values,
            valid_items=valid_items,
            offset=2,
        )

    @cute.jit
    def launch(
        source: cute.Pointer,
        destination: cute.Pointer,
        items_per_thread: cutlass.Constexpr,
    ):
        copy_prefix(source, destination, items_per_thread).launch(
            grid=1, block=64
        )

    # cutlass-load-store-example-end
    count = 64 * items_per_thread - 3
    values = np.arange(count + 5, dtype=np.int32) * 3
    observed = np.full(count + 7, -101, dtype=np.int32)
    with device_array(values) as src, device_array(observed) as out:
        launch(src, out, items_per_thread)
    np.testing.assert_array_equal(observed[2 : 2 + count], values[5:])
    np.testing.assert_array_equal(observed[:2], -101)
    np.testing.assert_array_equal(observed[2 + count :], -101)


def test_unsigned_pointer_example():
    # cutlass-load-store-unsigned-example-begin
    import cutlass
    from cutlass import cute

    import cuda.coop.cutlass as coop

    @cute.kernel
    def sort_unsigned(source: cute.Pointer, destination: cute.Pointer):
        block = coop.this_block()
        keys = coop.load(
            block, source, items_per_thread=2, dtype=cutlass.Uint32
        )
        ordered = coop.radix_sort_keys(block, keys)
        coop.store(block, destination, ordered, dtype=cutlass.Uint32)

    @cute.jit
    def launch(source: cute.Pointer, destination: cute.Pointer):
        sort_unsigned(source, destination).launch(grid=1, block=64)

    # cutlass-load-store-unsigned-example-end
    keys = np.arange(128, dtype=np.uint32) * np.uint32(2654435761)
    assert (keys >> 31).min() == 0 and (keys >> 31).max() == 1
    observed = np.zeros_like(keys)
    with device_array(keys) as src, device_array(observed) as out:
        launch(src, out)
    np.testing.assert_array_equal(observed, np.sort(keys))
