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


def test_scatter_example():
    # scatter-example-begin
    import numpy as np
    from numba_cuda_mlir import cuda, types

    import cuda.coop.numba_mlir as numba_coop

    @cuda.jit
    def reverse_tile(source, destination, items_per_thread):
        block = numba_coop.this_block()
        items = numba_coop.ThreadData(items_per_thread)
        ranks = numba_coop.ThreadData(items_per_thread)
        numba_coop.load(block, source, items)
        for item in range(items_per_thread):
            index = cuda.threadIdx.x * items_per_thread + item
            ranks[item] = types.int32(
                cuda.blockDim.x * items_per_thread - 1 - index
            )
        reversed_items = numba_coop.exchange(
            block, items, mode="scatter_to_blocked", ranks=ranks
        )
        numba_coop.store(block, destination, reversed_items)

    for items_per_thread in (1, 4):
        values = np.arange(128 * items_per_thread, dtype=np.int32) * 3 - 200
        source = cuda.to_device(values)
        destination = cuda.device_array_like(source)
        reverse_tile[1, 128](source, destination, items_per_thread)
        np.testing.assert_array_equal(destination.copy_to_host(), values[::-1])
    # scatter-example-end


def test_rotate_example():
    # rotate-example-begin
    import numpy as np
    from numba_cuda_mlir import cuda

    import cuda.coop.numba_mlir as numba_coop

    @cuda.jit
    def rotate_tile(source, destination):
        thread = cuda.threadIdx.x
        destination[thread] = numba_coop.shuffle(
            numba_coop.this_block(), source[thread], mode="rotate", distance=7
        )

    values = np.arange(128, dtype=np.int32) * 3 - 200
    source = cuda.to_device(values)
    destination = cuda.device_array_like(source)
    rotate_tile[1, 128](source, destination)
    np.testing.assert_array_equal(
        destination.copy_to_host(), np.roll(values, -7)
    )
    # rotate-example-end
