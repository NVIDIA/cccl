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


def test_qualified_radix_examples():
    # qualified-radix-example-begin
    import numpy as np
    from numba_cuda_mlir import cuda, types

    import cuda.coop.numba_mlir as coop

    @cuda.jit
    def order_digit(
        source, key_output, pair_output, positions, ranks_out, items_per_thread
    ):
        block = coop.this_block()
        keys = coop.ThreadData(items_per_thread)
        indices = coop.ThreadData(items_per_thread)
        coop.load(block, source, keys)
        for item in range(items_per_thread):
            indices[item] = types.int32(block.rank() * items_per_thread + item)
        ordered = coop.radix_sort_keys(block, keys, begin_bit=0, end_bit=4)
        pair_keys, pair_indices = coop.radix_sort_pairs(
            block, keys, indices, begin_bit=0, end_bit=4
        )
        ranks = coop.radix_rank_keys(block, keys, begin_bit=0, end_bit=4)
        coop.store(block, key_output, ordered)
        coop.store(block, pair_output, pair_keys)
        coop.store(block, positions, pair_indices)
        coop.store(block, ranks_out, ranks)

    for items_per_thread in (1, 4):
        values = np.random.default_rng(42).integers(
            0, 256, size=64 * items_per_thread, dtype=np.uint32
        )
        key_output = cuda.device_array_like(values)
        pair_output = cuda.device_array_like(values)
        positions = cuda.device_array(values.size, dtype=np.int32)
        ranks = cuda.device_array(values.size, dtype=np.int32)
        order_digit[1, 64](
            cuda.to_device(values),
            key_output,
            pair_output,
            positions,
            ranks,
            items_per_thread,
        )
        order = np.argsort(values & np.uint32(15), kind="stable")
        expected_ranks = np.empty(values.size, dtype=np.int32)
        expected_ranks[order] = np.arange(values.size)
        np.testing.assert_array_equal(key_output.copy_to_host(), values[order])
        np.testing.assert_array_equal(pair_output.copy_to_host(), values[order])
        np.testing.assert_array_equal(positions.copy_to_host(), order)
        np.testing.assert_array_equal(ranks.copy_to_host(), expected_ranks)
    # qualified-radix-example-end
