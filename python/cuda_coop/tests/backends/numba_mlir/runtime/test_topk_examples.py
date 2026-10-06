# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Execute the common TopK key and pair examples shown in the reference.

The programming guide and TopK visualization also include the pair example.
Keep each region self-contained for Sphinx. Negative input keys and zero-filled
out-of-bounds slots make an ignored valid count observable: those filler
zeros would otherwise enter a maximum selection.
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


def test_topk_pairs_example():
    """Check the example's partial-tile selection and original positions."""

    # topk-example-begin
    import numpy as np
    from numba_cuda_mlir import cuda, types

    from cuda import coop

    @cuda.jit
    def select_tile(
        source, count, selected_keys, original_positions, items_per_thread
    ):
        block = coop.this_block()
        keys = coop.ThreadData(items_per_thread)
        positions = coop.ThreadData(items_per_thread)
        coop.load(block, source, keys, valid_items=count, oob_default=0)
        for item in range(items_per_thread):
            positions[item] = types.int32(
                cuda.threadIdx.x * items_per_thread + item
            )
        chosen_keys, chosen_positions = coop.topk_max_pairs(
            block, keys, positions, k=8, valid_items=count
        )
        coop.store(block, selected_keys, chosen_keys, valid_items=8)
        coop.store(block, original_positions, chosen_positions, valid_items=8)

    for items_per_thread in (1, 4):
        values = (
            np.random.default_rng(42)
            .permutation(64 * items_per_thread)
            .astype(np.int32)
            - 256
        )
        values = values[
            : 64 * items_per_thread - 7
        ]  # A partial tile; all input keys are negative.
        source = cuda.to_device(values)
        selected_keys = cuda.device_array(8, dtype=np.int32)
        original_positions = cuda.device_array(8, dtype=np.int32)
        select_tile[1, 64](
            source,
            len(values),
            selected_keys,
            original_positions,
            items_per_thread,
        )
        actual = selected_keys.copy_to_host()
        indices = original_positions.copy_to_host()
        # TopK selects an unordered set. Sort only for this test's comparison.
        np.testing.assert_array_equal(np.sort(actual), np.sort(values)[-8:])
        assert np.all((indices >= 0) & (indices < len(values)))
        np.testing.assert_array_equal(actual, values[indices])
    # topk-example-end


def test_topk_keys_example():
    # topk-keys-example-begin
    import numpy as np
    from numba_cuda_mlir import cuda

    from cuda import coop

    @cuda.jit
    def select_extremes(source, count, smallest, largest, k, items_per_thread):
        block = coop.this_block()
        keys = coop.ThreadData(items_per_thread)
        coop.load(block, source, keys, valid_items=count, oob_default=0)
        low = coop.topk_min_keys(block, keys, k=k, valid_items=count)
        high = coop.topk_max_keys(block, keys, k=k, valid_items=count)
        selected_count = min(k, count)
        coop.store(block, smallest, low, valid_items=selected_count)
        coop.store(block, largest, high, valid_items=selected_count)

    for items_per_thread in (1, 4):
        for count in (5, 64 * items_per_thread - 7):
            values = (
                np.random.default_rng(42).permutation(count).astype(np.int32)
                - 256
            )
            source = cuda.to_device(values)
            selected_count = min(8, count)
            smallest = cuda.device_array(selected_count, dtype=np.int32)
            largest = cuda.device_array(selected_count, dtype=np.int32)
            select_extremes[1, 64](
                source, count, smallest, largest, 8, items_per_thread
            )
            # Selected keys are unordered; compare sets by sorting on the host.
            np.testing.assert_array_equal(
                np.sort(smallest.copy_to_host()), np.sort(values)[:8]
            )
            np.testing.assert_array_equal(
                np.sort(largest.copy_to_host()), np.sort(values)[-8:]
            )
    # topk-keys-example-end


def test_topk_min_pairs_example():
    # topk-min-pairs-example-begin
    import numpy as np
    from numba_cuda_mlir import cuda, types

    from cuda import coop

    @cuda.jit
    def select_smallest(
        source, count, selected_keys, positions, items_per_thread
    ):
        block = coop.this_block()
        keys = coop.ThreadData(items_per_thread)
        indices = coop.ThreadData(items_per_thread)
        coop.load(block, source, keys, valid_items=count, oob_default=0)
        for item in range(items_per_thread):
            indices[item] = types.int32(
                cuda.threadIdx.x * items_per_thread + item
            )
        chosen_keys, chosen_indices = coop.topk_min_pairs(
            block, keys, indices, k=8, valid_items=count
        )
        coop.store(block, selected_keys, chosen_keys, valid_items=8)
        coop.store(block, positions, chosen_indices, valid_items=8)

    for items_per_thread in (1, 4):
        # Positive keys make accidental selection of padded zeros observable.
        values = (
            np.random.default_rng(42)
            .permutation(64 * items_per_thread - 7)
            .astype(np.int32)
            + 1
        )
        source = cuda.to_device(values)
        selected_keys = cuda.device_array(8, dtype=np.int32)
        positions = cuda.device_array(8, dtype=np.int32)
        select_smallest[1, 64](
            source, len(values), selected_keys, positions, items_per_thread
        )
        actual = selected_keys.copy_to_host()
        indices = positions.copy_to_host()
        np.testing.assert_array_equal(np.sort(actual), np.sort(values)[:8])
        assert np.all((indices >= 0) & (indices < len(values)))
        np.testing.assert_array_equal(actual, values[indices])
    # topk-min-pairs-example-end
