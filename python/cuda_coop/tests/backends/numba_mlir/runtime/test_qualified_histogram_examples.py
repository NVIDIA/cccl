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


def test_qualified_histogram_example():
    # qualified-histogram-example-begin
    import numpy as np
    from numba_cuda_mlir import cuda, types

    import cuda.coop.numba_mlir as coop

    @cuda.jit
    def count_bins(source, destination, items_per_thread, use_sort):
        block = coop.this_block()
        samples = cuda.local.array(items_per_thread, dtype=types.int32)
        coop.load(block, source, samples)
        # bins is a count. Intermediate counters are allocated automatically.
        if use_sort:
            counts = coop.histogram(
                block, samples, bins=80, bins_per_thread=2, algorithm="sort"
            )
        else:
            counts = coop.histogram(block, samples, bins=80, bins_per_thread=2)
        coop.store(
            block, destination, counts, algorithm="striped", valid_items=80
        )

    for items_per_thread in (1, 4):
        values = (np.arange(64 * items_per_thread, dtype=np.int32) * 7) % 80
        for use_sort in (False, True):
            destination = cuda.device_array(80, dtype=np.int32)
            count_bins[1, 64](
                cuda.to_device(values), destination, items_per_thread, use_sort
            )
            np.testing.assert_array_equal(
                destination.copy_to_host(), np.bincount(values, minlength=80)
            )
    # qualified-histogram-example-end
