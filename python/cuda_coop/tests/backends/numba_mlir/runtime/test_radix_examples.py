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


def test_radix_sort_pairs_example():
    # radix-sort-example-begin
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
        ordered_keys, ordered_positions = coop.radix_sort_pairs(
            block, keys, positions
        )
        coop.store(block, sorted_keys, ordered_keys)
        coop.store(block, original_positions, ordered_positions)

    for items_per_thread in (1, 4):
        values = np.random.default_rng(42).integers(
            -8, 9, size=64 * items_per_thread, dtype=np.int32
        )
        source = cuda.to_device(values)
        sorted_keys = cuda.device_array_like(source)
        original_positions = cuda.device_array_like(source)
        order_tile[1, 64](
            source, sorted_keys, original_positions, items_per_thread
        )
        expected_positions = np.argsort(values, kind="stable")
        np.testing.assert_array_equal(
            sorted_keys.copy_to_host(), values[expected_positions]
        )
        np.testing.assert_array_equal(
            original_positions.copy_to_host(), expected_positions
        )
    # radix-sort-example-end


def test_radix_rank_example():
    # radix-rank-example-begin
    import numpy as np
    from numba_cuda_mlir import cuda

    from cuda import coop

    @cuda.jit
    def rank_low_digit(source, destination, items_per_thread):
        block = coop.this_block()
        keys = coop.ThreadData(items_per_thread)
        coop.load(block, source, keys)
        ranks = coop.radix_rank(block, keys, begin_bit=0, end_bit=4)
        coop.store(block, destination, ranks)

    for items_per_thread in (1, 4):
        values = np.random.default_rng(42).integers(
            0, 256, size=64 * items_per_thread, dtype=np.uint32
        )
        source = cuda.to_device(values)
        destination = cuda.device_array(64 * items_per_thread, dtype=np.int32)
        rank_low_digit[1, 64](source, destination, items_per_thread)
        order = np.argsort(values & np.uint32(15), kind="stable")
        expected = np.empty(64 * items_per_thread, dtype=np.int32)
        expected[order] = np.arange(64 * items_per_thread, dtype=np.int32)
        np.testing.assert_array_equal(destination.copy_to_host(), expected)
    # radix-rank-example-end
