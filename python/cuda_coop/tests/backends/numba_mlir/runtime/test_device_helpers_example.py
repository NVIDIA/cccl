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


def test_device_helpers_example():
    # coop-pg-device-helpers-begin
    import numpy as np
    from numba_cuda_mlir import cuda

    from cuda import coop

    @cuda.jit(device=True, inline="always")
    def load_into(group, source, items):
        coop.load(group, source, items)

    @cuda.jit(device=True)
    def load_tile(group, source, items_per_thread):
        items = coop.ThreadData(items_per_thread)
        load_into(group, source, items)
        return items

    @cuda.jit
    def copy_tile(source, destination, items_per_thread):
        block = coop.this_block()
        items = load_tile(block, source, items_per_thread)
        coop.store(block, destination, items)

    for items_per_thread in (1, 4):
        expected = np.arange(128 * items_per_thread, dtype=np.int32)
        source = cuda.to_device(expected)
        destination = cuda.device_array_like(source)
        copy_tile[1, 128](source, destination, items_per_thread)
        np.testing.assert_array_equal(destination.copy_to_host(), expected)
    # coop-pg-device-helpers-end
