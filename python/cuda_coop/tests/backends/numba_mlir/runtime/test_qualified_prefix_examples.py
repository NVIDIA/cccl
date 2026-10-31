# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


"""Execute qualified built-in Scan examples against NumPy results."""

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


def test_qualified_prefix_examples():
    # qualified-prefix-example-begin
    import numpy as np
    from numba_cuda_mlir import cuda

    import cuda.coop.numba_mlir as coop

    @cuda.jit
    def prefixes(source, maxima, sums, items_per_thread):
        block = coop.this_block()
        items = coop.ThreadData(items_per_thread)
        coop.load(block, source, items)
        running_max = coop.scan(block, items, mode="inclusive", scan_op="max")
        running_sum = coop.inclusive_sum(block, items)
        coop.store(block, maxima, running_max)
        coop.store(block, sums, running_sum)

    for items_per_thread in (1, 4):
        values = (
            np.arange(64 * items_per_thread, dtype=np.int32) * 7
        ) % 31 - 15
        maxima = cuda.device_array_like(values)
        sums = cuda.device_array_like(values)
        prefixes[1, 64](cuda.to_device(values), maxima, sums, items_per_thread)
        np.testing.assert_array_equal(
            maxima.copy_to_host(), np.maximum.accumulate(values)
        )
        np.testing.assert_array_equal(
            sums.copy_to_host(), np.cumsum(values, dtype=np.int32)
        )
    # qualified-prefix-example-end
