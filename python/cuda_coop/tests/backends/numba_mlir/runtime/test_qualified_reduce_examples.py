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


def test_qualified_reduce_example():
    # qualified-reduce-example-begin
    import numpy as np
    from numba_cuda_mlir import cuda

    import cuda.coop.numba_mlir as coop

    @cuda.jit(device=True)
    def maximum(left, right):
        return max(left, right)

    @cuda.jit
    def tile_maximum(source, output, items_per_thread):
        block = coop.this_block()
        items = coop.ThreadData(items_per_thread)
        coop.load(block, source, items)
        result = coop.reduce(block, items, binary_op=maximum, broadcast=False)
        # Custom reductions define the result at group rank zero.
        if block.rank() == 0:
            output[0] = result

    for items_per_thread in (1, 4):
        values = (
            np.arange(64 * items_per_thread, dtype=np.int32) * 7
        ) % 31 - 15
        output = cuda.device_array(1, dtype=np.int32)
        tile_maximum[1, 64](cuda.to_device(values), output, items_per_thread)
        np.testing.assert_array_equal(output.copy_to_host(), [values.max()])
    # qualified-reduce-example-end


def test_qualified_sum_example():
    # qualified-sum-example-begin
    import numpy as np
    from numba_cuda_mlir import cuda

    import cuda.coop.numba_mlir as coop

    @cuda.jit
    def warp_totals(source, destination):
        group = coop.this_warp().group_by(8)
        thread = cuda.threadIdx.x
        # The default broadcasts each logical warp's total to all its lanes.
        destination[thread] = coop.sum(group, source[thread])

    values = np.arange(64, dtype=np.int32) % 7
    destination = cuda.device_array_like(values)
    warp_totals[1, 64](cuda.to_device(values), destination)
    expected = np.repeat(values.reshape(8, 8).sum(axis=1), 8)
    np.testing.assert_array_equal(destination.copy_to_host(), expected)
    # qualified-sum-example-end
