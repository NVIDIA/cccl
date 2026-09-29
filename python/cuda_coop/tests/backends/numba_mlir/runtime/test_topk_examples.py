# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Executable selection example included by the programming guide."""

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
    # topk-example-begin
    import numpy as np
    from numba_cuda_mlir import cuda, types

    from cuda import coop

    coop.register("numba-cuda-mlir")

    @cuda.jit
    def select_tile(source, count, selected_keys, original_positions):
        block = coop.this_block()
        keys = coop.ThreadData(2, dtype=np.int32)
        positions = coop.ThreadData(2, dtype=np.int32)
        coop.load(block, source, keys, valid_items=count, oob_default=0)
        for item in range(2):
            positions[item] = types.int32(cuda.threadIdx.x * 2 + item)
        chosen_keys, chosen_positions = coop.topk_max_pairs(
            block, keys, positions, k=8, valid_items=count
        )
        coop.store(block, selected_keys, chosen_keys, valid_items=8)
        coop.store(block, original_positions, chosen_positions, valid_items=8)

    values = np.random.default_rng(42).permutation(128).astype(np.int32) - 256
    values = values[:93]  # A partial tile; all input keys are negative.
    source = cuda.to_device(values)
    selected_keys = cuda.device_array(8, dtype=np.int32)
    original_positions = cuda.device_array(8, dtype=np.int32)
    select_tile[1, 64](source, len(values), selected_keys, original_positions)
    actual = selected_keys.copy_to_host()
    indices = original_positions.copy_to_host()
    # TopK selects an unordered set. Sort only for this test's comparison.
    np.testing.assert_array_equal(np.sort(actual), np.sort(values)[-8:])
    assert np.all((indices >= 0) & (indices < len(values)))
    np.testing.assert_array_equal(actual, values[indices])
    # topk-example-end
