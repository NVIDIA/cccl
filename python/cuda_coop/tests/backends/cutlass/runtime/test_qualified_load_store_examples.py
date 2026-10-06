# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check qualified load store examples against host results."""

import numpy as np
import pytest

pytest.importorskip("cutlass")

from tests.backends.cutlass.support import device_array

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.runtime, pytest.mark.gpu]


@pytest.mark.parametrize("items_per_thread", [1, 4])
def test_load_store_example(items_per_thread):
    # qualified-load-store-example-begin
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
        values = coop.ThreadData(items_per_thread)
        coop.load(
            block,
            source,
            values,
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

    # qualified-load-store-example-end
    count = 64 * items_per_thread - 3
    values = np.arange(count + 5, dtype=np.int32) * 3
    observed = np.full(count + 7, -101, dtype=np.int32)
    with device_array(values) as src, device_array(observed) as out:
        launch(src, out, items_per_thread)
    np.testing.assert_array_equal(observed[2 : 2 + count], values[5:])
    np.testing.assert_array_equal(observed[:2], -101)
    np.testing.assert_array_equal(observed[2 + count :], -101)
