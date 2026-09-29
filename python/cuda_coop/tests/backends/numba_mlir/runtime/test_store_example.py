# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Executable example included in the ``cuda.coop.store`` docstring."""

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
    def store_prefix(destination, preserved):
        block = coop.this_block()
        rank = cuda.threadIdx.x
        items = coop.ThreadData(2, dtype=np.int32)
        items[0] = types.int32(2 * rank)
        items[1] = types.int32(2 * rank + 1)
        coop.store(
            block,
            destination,
            items,
            algorithm="transpose",
            valid_items=251,
            offset=5,
        )
        preserved[2 * rank] = items[0]
        preserved[2 * rank + 1] = items[1]

    destination = cuda.to_device(np.full(263, -1, dtype=np.int32))
    preserved = cuda.device_array(256, dtype=np.int32)
    store_prefix[1, 128](destination, preserved)
    expected = np.full(263, -1, dtype=np.int32)
    expected[5:256] = np.arange(251, dtype=np.int32)
    np.testing.assert_array_equal(destination.copy_to_host(), expected)
    np.testing.assert_array_equal(preserved.copy_to_host(), np.arange(256))
    # example-end
