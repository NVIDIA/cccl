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


def test_qualified_batched_reduce_example():
    # qualified-batched-reduce-example-begin
    import numpy as np
    from numba_cuda_mlir import cuda

    import cuda.coop.numba_mlir as coop

    @cuda.jit(device=True)
    def maximum(left, right):
        return max(left, right)

    @cuda.jit
    def feature_maxima(source, destination, items_per_thread):
        warp = coop.this_warp()
        features = coop.ThreadData(items_per_thread)
        coop.load(coop.this_block(), source, features)
        result = coop.reduce_batched(warp, features, binary_op=maximum)
        # Each input slot is a separate batch reduced across the warp.
        if warp.rank() < items_per_thread:
            destination[warp.rank()] = result[0]

    for items_per_thread in (1, 4):
        values = (
            np.arange(32 * items_per_thread, dtype=np.int32) * 7
        ) % 31 - 15
        destination = cuda.device_array(items_per_thread, dtype=np.int32)
        feature_maxima[1, 32](
            cuda.to_device(values), destination, items_per_thread
        )
        np.testing.assert_array_equal(
            destination.copy_to_host(),
            values.reshape(32, items_per_thread).max(axis=0),
        )
    # qualified-batched-reduce-example-end
