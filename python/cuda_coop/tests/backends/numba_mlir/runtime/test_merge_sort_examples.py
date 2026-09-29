# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Executable merge-sort examples included by the programming guide."""

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

    coop.register("numba-cuda-mlir")

    @cuda.jit
    def order_tile(source, sorted_keys, original_positions):
        block = coop.this_block()
        keys = coop.ThreadData(2, dtype=np.int32)
        positions = coop.ThreadData(2, dtype=np.int32)
        coop.load(block, source, keys)
        for item in range(2):
            positions[item] = types.int32(cuda.threadIdx.x * 2 + item)
        ordered_keys, ordered_positions = coop.merge_sort_pairs(block, keys, positions)
        coop.store(block, sorted_keys, ordered_keys)
        coop.store(block, original_positions, ordered_positions)

    values = np.random.default_rng(42).permutation(128).astype(np.int32) - 64
    source = cuda.to_device(values)
    sorted_keys = cuda.device_array_like(source)
    original_positions = cuda.device_array_like(source)
    order_tile[1, 64](source, sorted_keys, original_positions)
    actual = sorted_keys.copy_to_host()
    indices = original_positions.copy_to_host()
    np.testing.assert_array_equal(actual, np.sort(values))
    np.testing.assert_array_equal(actual, values[indices])
    # merge-sort-example-end
