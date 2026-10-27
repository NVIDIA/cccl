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


@pytest.mark.parametrize("items_per_thread", [1, 4])
def test_shared_load_store_example(items_per_thread):
    # shared-load-store-example-begin
    import numpy as np
    from numba_cuda_mlir import cuda, types

    from cuda import coop

    # Capture the static shared-array extent before compiling the kernel.
    threads = 32
    tile_items = threads * items_per_thread

    @cuda.jit
    def stage_and_reload(source, destination, items_per_thread):
        block = coop.this_block()
        shared = cuda.shared.array(tile_items, dtype=types.int32)

        values = coop.load(block, source, items_per_thread=items_per_thread)
        coop.store(block, shared, values)
        cuda.syncthreads()

        # Striped ownership reads values staged by other threads.
        reloaded = coop.load(
            block,
            shared,
            items_per_thread=items_per_thread,
            algorithm="striped",
        )
        coop.store(block, destination, reloaded)

    values = np.arange(tile_items, dtype=np.int32)
    source = cuda.to_device(values)
    destination = cuda.device_array_like(source)
    stage_and_reload[1, threads](source, destination, items_per_thread)
    expected = values.reshape(items_per_thread, threads).T.ravel()
    np.testing.assert_array_equal(destination.copy_to_host(), expected)
    # shared-load-store-example-end
