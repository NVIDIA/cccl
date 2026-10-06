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


def test_qualified_neighbor_examples():
    # qualified-neighbor-example-begin
    import numpy as np
    from numba_cuda_mlir import cuda

    import cuda.coop.numba_mlir as coop

    @cuda.jit(device=True)
    def absolute_gap(current, neighbor):
        return abs(current - neighbor)

    @cuda.jit(device=True)
    def large_jump(previous, current):
        return abs(current - previous) > 2

    @cuda.jit
    def compare_neighbors(
        source, deltas, heads_out, tails_out, items_per_thread
    ):
        block = coop.this_block()
        items = coop.ThreadData(items_per_thread)
        coop.load(block, source, items)
        differences = coop.adjacent_difference(
            block,
            items,
            direction="right",
            tile_successor_item=0,
            difference_op=absolute_gap,
        )
        heads, tails = coop.discontinuity(
            block, items, mode="heads_and_tails", flag_op=large_jump
        )
        coop.store(block, deltas, differences)
        coop.store(block, heads_out, heads)
        coop.store(block, tails_out, tails)

    for items_per_thread in (1, 4):
        values = np.arange(64 * items_per_thread, dtype=np.int32) // 5 % 7
        outputs = [cuda.device_array_like(values) for _ in range(3)]
        compare_neighbors[1, 64](
            cuda.to_device(values), *outputs, items_per_thread
        )
        deltas, heads, tails = [output.copy_to_host() for output in outputs]
        expected_deltas = np.abs(np.diff(np.r_[values, np.int32(0)]))
        boundaries = np.abs(values[1:] - values[:-1]) > 2
        np.testing.assert_array_equal(deltas, expected_deltas)
        np.testing.assert_array_equal(heads, np.r_[True, boundaries])
        np.testing.assert_array_equal(tails, np.r_[boundaries, True])
    # qualified-neighbor-example-end
