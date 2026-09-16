# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Executable examples included in the qualified scan API docstrings."""

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


def test_qualified_inclusive_scan_example():
    # qualified-inclusive-scan-example-begin
    import numpy as np
    from numba_cuda_mlir import cuda, types

    import cuda.coop.numba_mlir as coop

    @cuda.jit(device=True)
    def maximum(left, right):
        if left > right:
            return left
        return right

    @cuda.jit
    def running_maximum(source, destination):
        block = coop.this_block()
        items = cuda.local.array(2, dtype=types.int32)
        for item in range(2):
            items[item] = source[2 * cuda.threadIdx.x + item]
        prefixes = coop.inclusive_scan(block, items, scan_op=maximum)
        for item in range(2):
            destination[2 * cuda.threadIdx.x + item] = prefixes[item]

    values = ((np.arange(128, dtype=np.int32) * 17) % 113) - 51
    source = cuda.to_device(values)
    destination = cuda.device_array_like(source)
    running_maximum[1, 64](source, destination)
    expected = np.maximum.accumulate(values)
    np.testing.assert_array_equal(destination.copy_to_host(), expected)
    # qualified-inclusive-scan-example-end


def test_qualified_exclusive_scan_example():
    # qualified-exclusive-scan-example-begin
    import numpy as np
    from numba_cuda_mlir import cuda

    import cuda.coop.numba_mlir as coop

    @cuda.jit
    def partial_prefixes(source, destination, totals, initial, valid):
        thread = cuda.threadIdx.x
        group = coop.this_warp().group_by(8)
        aggregate = coop.ThreadData(1, dtype=np.int32)
        prefix = coop.exclusive_scan(
            group,
            source[thread],
            initial_value=initial,
            valid_items=valid,
            aggregate_output=aggregate,
        )
        if group.rank() < valid:
            destination[thread] = prefix
        totals[thread] = aggregate[0]

    values = (np.arange(64, dtype=np.int32) % 13) + 1
    source = cuda.to_device(values)
    destination = cuda.to_device(np.full_like(values, -1))
    totals = cuda.device_array_like(source)
    initial = np.int32(11)
    valid = np.int64(5)
    partial_prefixes[1, 64](source, destination, totals, initial, valid)
    expected = np.full_like(values, -1).reshape(8, 8)
    valid_values = values.reshape(8, 8)[:, :valid]
    expected[:, 0] = initial
    expected[:, 1:valid] = initial + np.cumsum(valid_values[:, :-1], axis=1)
    expected_totals = np.repeat(valid_values.sum(axis=1), 8)
    np.testing.assert_array_equal(destination.copy_to_host(), expected.ravel())
    np.testing.assert_array_equal(totals.copy_to_host(), expected_totals)
    # qualified-exclusive-scan-example-end


def test_qualified_exclusive_sum_example():
    # qualified-exclusive-sum-example-begin
    import numpy as np
    from numba_cuda_mlir import cuda, types

    import cuda.coop.numba_mlir as coop

    @cuda.jit(device=True)
    def carry_total(state, tile_total):
        previous = state[0]
        state[0] = previous + tile_total
        return previous

    running_prefix = coop.StatefulFunction(carry_total, types.int64)

    @cuda.jit
    def scan_successive_tiles(source, destination, final_total):
        block = coop.this_block()
        state = coop.ThreadData(1, dtype=types.int64)
        state[0] = types.int64(0)
        scratch = coop.TempStorage()
        for tile in range(3):
            index = tile * cuda.blockDim.x + cuda.threadIdx.x
            destination[index] = coop.exclusive_sum(
                block,
                source[index],
                state,
                prefix_op=running_prefix,
                temp_storage=scratch,
            )
        if block.rank() == 0:
            final_total[0] = state[0]

    values = np.arange(192, dtype=np.int32) % 9
    source = cuda.to_device(values)
    destination = cuda.device_array_like(source)
    final_total = cuda.device_array(1, dtype=np.int64)
    scan_successive_tiles[1, 64](source, destination, final_total)
    expected = np.zeros_like(values)
    expected[1:] = np.cumsum(values[:-1], dtype=np.int32)
    np.testing.assert_array_equal(destination.copy_to_host(), expected)
    np.testing.assert_array_equal(final_total.copy_to_host(), [values.sum()])
    # qualified-exclusive-sum-example-end
