# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Executable examples included in the common scan API docstrings."""

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


def test_scan_example():
    # scan-example-begin
    import numpy as np
    from numba_cuda_mlir import cuda

    import cuda.coop.numba_mlir as numba_coop  # noqa: F401 -- activate the backend
    from cuda import coop

    @cuda.jit
    def xor_prefixes(source, destination):
        thread = cuda.threadIdx.x
        destination[thread] = coop.scan(
            coop.this_block(),
            source[thread],
            mode="inclusive",
            scan_op="bit_xor",
        )

    values = np.arange(64, dtype=np.int32) % 7
    source = cuda.to_device(values)
    destination = cuda.device_array_like(source)
    xor_prefixes[1, 64](source, destination)
    expected = np.bitwise_xor.accumulate(values)
    np.testing.assert_array_equal(destination.copy_to_host(), expected)
    # scan-example-end


def test_exclusive_sum_example():
    # exclusive-sum-example-begin
    import numpy as np
    from numba_cuda_mlir import cuda

    import cuda.coop.numba_mlir as numba_coop  # noqa: F401 -- activate the backend
    from cuda import coop

    @cuda.jit
    def count_offsets(counts, offsets):
        thread = cuda.threadIdx.x
        offsets[thread] = coop.exclusive_sum(coop.this_block(), counts[thread])

    counts = np.arange(64, dtype=np.int32) % 4
    source = cuda.to_device(counts)
    offsets = cuda.device_array_like(source)
    count_offsets[1, 64](source, offsets)
    expected = np.zeros_like(counts)
    expected[1:] = np.cumsum(counts[:-1], dtype=np.int32)
    np.testing.assert_array_equal(offsets.copy_to_host(), expected)
    # exclusive-sum-example-end


def test_inclusive_sum_example():
    # inclusive-sum-example-begin
    import numpy as np
    from numba_cuda_mlir import cuda

    import cuda.coop.numba_mlir as numba_coop  # noqa: F401 -- activate the backend
    from cuda import coop

    @cuda.jit
    def sum_tile(source, destination, original):
        block = coop.this_block()
        items = coop.ThreadData(2, dtype=np.int32)
        coop.load(block, source, items)
        prefixes = coop.inclusive_sum(block, items, algorithm="raking_memoize")
        coop.store(block, destination, prefixes)
        coop.store(block, original, items)

    values = (np.arange(128, dtype=np.int32) % 11) - 5
    source = cuda.to_device(values)
    destination = cuda.device_array_like(source)
    original = cuda.device_array_like(source)
    sum_tile[1, 64](source, destination, original)
    expected = np.cumsum(values, dtype=np.int32)
    np.testing.assert_array_equal(destination.copy_to_host(), expected)
    np.testing.assert_array_equal(original.copy_to_host(), values)
    # inclusive-sum-example-end


def test_exclusive_scan_example():
    # exclusive-scan-example-begin
    import numpy as np
    from numba_cuda_mlir import cuda

    import cuda.coop.numba_mlir as numba_coop  # noqa: F401 -- activate the backend
    from cuda import coop

    @cuda.jit
    def minimum_prefixes(source, destination, initial):
        thread = cuda.threadIdx.x
        destination[thread] = coop.exclusive_scan(
            coop.this_block(),
            source[thread],
            scan_op="min",
            initial_value=initial,
        )

    values = ((np.arange(64, dtype=np.int32) * 7 + 29) % 31) - 15
    initial = np.int32(10)
    source = cuda.to_device(values)
    destination = cuda.device_array_like(source)
    minimum_prefixes[1, 64](source, destination, initial)
    expected = np.minimum.accumulate(np.concatenate(([initial], values[:-1])))
    np.testing.assert_array_equal(destination.copy_to_host(), expected)
    # exclusive-scan-example-end


def test_inclusive_scan_example():
    # inclusive-scan-example-begin
    import numpy as np
    from numba_cuda_mlir import cuda

    import cuda.coop.numba_mlir as numba_coop  # noqa: F401 -- activate the backend
    from cuda import coop

    @cuda.jit
    def warp_maxima(source, destination):
        thread = cuda.threadIdx.x
        group = coop.this_warp().group_by(8)
        destination[thread] = coop.inclusive_scan(group, source[thread], scan_op="max")

    values = ((np.arange(64, dtype=np.int32) * 7) % 31) - 15
    source = cuda.to_device(values)
    destination = cuda.device_array_like(source)
    warp_maxima[1, 64](source, destination)
    expected = np.maximum.accumulate(values.reshape(8, 8), axis=1).ravel()
    np.testing.assert_array_equal(destination.copy_to_host(), expected)
    # inclusive-scan-example-end
