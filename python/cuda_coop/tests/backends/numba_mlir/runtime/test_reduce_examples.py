# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Executable examples included in the portable API docstrings."""

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


def test_reduce_example():
    # reduce-example-begin
    import numpy as np
    from numba_cuda_mlir import cuda

    import cuda.coop.numba_mlir as numba_coop  # noqa: F401 -- activate the backend
    from cuda import coop

    @cuda.jit
    def extrema(source, maxima, prefix_minimum, valid):
        block = coop.this_block()
        value = source[cuda.threadIdx.x]
        maxima[cuda.threadIdx.x] = coop.reduce(block, value, binary_op="max")
        minimum = coop.reduce(
            block,
            value,
            binary_op="min",
            broadcast=False,
            valid_items=valid,
            algorithm="warp_reductions",
        )
        if cuda.threadIdx.x == 0:
            prefix_minimum[0] = minimum

    values = np.arange(128, 0, -1, dtype=np.int32)
    source = cuda.to_device(values)
    maxima = cuda.device_array_like(source)
    prefix_minimum = cuda.device_array(1, dtype=np.int32)
    extrema[1, 128](source, maxima, prefix_minimum, 93)
    np.testing.assert_array_equal(
        maxima.copy_to_host(), np.full_like(values, values.max())
    )
    assert prefix_minimum.copy_to_host()[0] == values[:93].min()
    # reduce-example-end


def test_sum_example():
    # sum-example-begin
    import numpy as np
    from numba_cuda_mlir import cuda

    import cuda.coop.numba_mlir as numba_coop  # noqa: F401 -- activate the backend
    from cuda import coop

    @cuda.jit
    def sum_tiles(source, totals):
        block = coop.this_block()
        items = coop.ThreadData(2, dtype=np.int32)
        tile_size = cuda.blockDim.x * 2
        offset = cuda.blockIdx.x * tile_size
        valid = min(max(source.size - offset, 0), tile_size)
        coop.load(block, source, items, offset=offset, valid_items=valid, oob_default=0)
        total = coop.sum(block, items, broadcast=False, algorithm="raking")
        if cuda.threadIdx.x == 0:
            totals[cuda.blockIdx.x] = total

    values = np.arange(300, dtype=np.int32) - 150
    source = cuda.to_device(values)
    totals = cuda.device_array(2, dtype=np.int32)
    sum_tiles[2, 128](source, totals)
    expected = np.array([values[:256].sum(), values[256:].sum()], dtype=np.int32)
    np.testing.assert_array_equal(totals.copy_to_host(), expected)
    # sum-example-end
