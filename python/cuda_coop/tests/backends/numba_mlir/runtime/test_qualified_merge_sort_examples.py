# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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


def test_qualified_sort_examples():
    # qualified-sort-example-begin
    import numpy as np
    from numba_cuda_mlir import cuda, types

    import cuda.coop.numba_mlir as coop

    @cuda.jit(device=True)
    def greater(left, right):
        return left > right

    @cuda.jit
    def order_tile(
        source, key_output, pair_output, positions, items_per_thread
    ):
        block = coop.this_block()
        keys = coop.ThreadData(items_per_thread)
        indices = coop.ThreadData(items_per_thread)
        coop.load(block, source, keys)
        for item in range(items_per_thread):
            indices[item] = types.int32(block.rank() * items_per_thread + item)
        ordered = coop.merge_sort_keys(block, keys, compare_op=greater)
        pair_keys, pair_indices = coop.merge_sort_pairs(
            block, keys, indices, compare_op=greater
        )
        coop.store(block, key_output, ordered)
        coop.store(block, pair_output, pair_keys)
        coop.store(block, positions, pair_indices)

    for items_per_thread in (1, 4):
        values = (
            np.random.default_rng(42)
            .permutation(64 * items_per_thread)
            .astype(np.int32)
        )
        key_output = cuda.device_array_like(values)
        pair_output = cuda.device_array_like(values)
        positions = cuda.device_array_like(values)
        order_tile[1, 64](
            cuda.to_device(values),
            key_output,
            pair_output,
            positions,
            items_per_thread,
        )
        expected = np.sort(values)[::-1]
        np.testing.assert_array_equal(key_output.copy_to_host(), expected)
        np.testing.assert_array_equal(pair_output.copy_to_host(), expected)
        np.testing.assert_array_equal(
            pair_output.copy_to_host(), values[positions.copy_to_host()]
        )
    # qualified-sort-example-end
