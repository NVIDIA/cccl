# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Run the documented key and pair sorts against independent host results.

The marked examples are shared by the API reference and guides.
Keep its key order and key/index association checks together.
"""

import pytest

cuda = pytest.importorskip("numba_cuda_mlir.cuda")
if not cuda.is_available():
    pytest.skip("requires a CUDA-capable runtime", allow_module_level=True)

pytestmark = [
    pytest.mark.backend_numba_mlir,
    pytest.mark.runtime,
    pytest.mark.gpu,
    pytest.mark.filterwarnings(
        "ignore::numba_cuda_mlir.numba_cuda.core.errors.NumbaPerformanceWarning"
    ),
]


def test_merge_sort_pairs_example():
    # merge-sort-example-begin
    import numpy as np
    from numba_cuda_mlir import cuda, types

    from cuda import coop

    @cuda.jit
    def order_tile(source, sorted_keys, original_positions, items_per_thread):
        block = coop.this_block()
        keys = coop.ThreadData(items_per_thread)
        positions = coop.ThreadData(items_per_thread)
        coop.load(block, source, keys)
        for item in range(items_per_thread):
            positions[item] = types.int32(
                cuda.threadIdx.x * items_per_thread + item
            )
        ordered_keys, ordered_positions = coop.merge_sort_pairs(
            block, keys, positions
        )
        coop.store(block, sorted_keys, ordered_keys)
        coop.store(block, original_positions, ordered_positions)

    for items_per_thread in (1, 4):
        values = (
            np.random.default_rng(42)
            .permutation(64 * items_per_thread)
            .astype(np.int32)
            - 64
        )
        source = cuda.to_device(values)
        sorted_keys = cuda.device_array_like(source)
        original_positions = cuda.device_array_like(source)
        order_tile[1, 64](
            source, sorted_keys, original_positions, items_per_thread
        )
        actual = sorted_keys.copy_to_host()
        indices = original_positions.copy_to_host()
        np.testing.assert_array_equal(actual, np.sort(values))
        np.testing.assert_array_equal(actual, values[indices])
    # merge-sort-example-end


def test_merge_sort_keys_example():
    # merge-sort-keys-example-begin
    import numpy as np
    from numba_cuda_mlir import cuda

    from cuda import coop

    @cuda.jit
    def order_partial_tile(source, count, destination, items_per_thread):
        block = coop.this_block()
        keys = coop.ThreadData(items_per_thread)
        coop.load(block, source, keys, valid_items=count, oob_default=-1)
        ordered = coop.merge_sort_keys(
            block,
            keys,
            descending=True,
            valid_items=count,
            oob_default=-1,
        )
        coop.store(block, destination, ordered, valid_items=count)

    for items_per_thread in (1, 4):
        values = (
            np.random.default_rng(42)
            .permutation(64 * items_per_thread - 7)
            .astype(np.int32)
        )
        source = cuda.to_device(values)
        destination = cuda.device_array_like(source)
        order_partial_tile[1, 64](
            source, len(values), destination, items_per_thread
        )
        np.testing.assert_array_equal(
            destination.copy_to_host(), np.sort(values)[::-1]
        )
    # merge-sort-keys-example-end
