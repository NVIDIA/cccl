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


def test_store_example():
    # example-begin
    import numpy as np
    from numba_cuda_mlir import cuda, types

    import cuda.coop.numba_mlir as numba_coop  # noqa: F401 -- activate the backend
    from cuda import coop

    @cuda.jit
    def store_prefix(destination, items_per_thread):
        block = coop.this_block()
        rank = cuda.threadIdx.x
        items = coop.ThreadData(items_per_thread)
        for item in range(items_per_thread):
            items[item] = types.int32(items_per_thread * rank + item)
        coop.store(
            block,
            destination,
            items,
            algorithm="transpose",
            valid_items=cuda.blockDim.x * items_per_thread - 5,
            offset=5,
        )

    for items_per_thread in (1, 4):
        destination = cuda.to_device(
            np.full(128 * items_per_thread + 7, -1, dtype=np.int32)
        )
        store_prefix[1, 128](destination, items_per_thread)
        expected = np.full(128 * items_per_thread + 7, -1, dtype=np.int32)
        expected[5 : 128 * items_per_thread] = np.arange(
            128 * items_per_thread - 5, dtype=np.int32
        )
        np.testing.assert_array_equal(destination.copy_to_host(), expected)
    # example-end
