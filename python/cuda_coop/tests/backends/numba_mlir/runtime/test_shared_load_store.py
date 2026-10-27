# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Round-trip group tiles through explicitly synchronized shared memory."""

import numpy as np
import pytest

cuda = pytest.importorskip("numba_cuda_mlir.cuda")
if not cuda.is_available():
    pytest.skip("requires a CUDA-capable runtime", allow_module_level=True)

from numba_cuda_mlir import types

import cuda.coop.numba_mlir as numba_coop
from cuda import coop

pytestmark = [
    pytest.mark.backend_numba_mlir,
    pytest.mark.runtime,
    pytest.mark.gpu,
    pytest.mark.filterwarnings(
        "ignore::numba_cuda_mlir.numba_cuda.core.errors.NumbaPerformanceWarning"
    ),
]


@pytest.mark.parametrize("api", [coop, numba_coop], ids=["common", "qualified"])
@pytest.mark.parametrize("items_per_thread", [1, 4])
@pytest.mark.parametrize(
    "returning", [False, True], ids=["output", "returning"]
)
def test_load_store_shared_array_roundtrip(api, items_per_thread, returning):
    tile_items = 32 * items_per_thread

    @cuda.jit
    def kernel(source, destination, items_per_thread):
        group = api.this_block()
        shared = cuda.shared.array(tile_items, dtype=types.int32)
        values = api.load(group, source, items_per_thread=items_per_thread)
        api.store(group, shared, values)
        cuda.syncthreads()
        # Shifting the source makes threads consume one another's writes.
        memory = shared[1:]
        if returning:
            loaded = api.load(
                group,
                memory,
                items_per_thread=items_per_thread,
                valid_items=tile_items - 1,
                oob_default=-7,
            )
        else:
            loaded = api.ThreadData(items_per_thread)
            api.load(
                group,
                memory,
                loaded,
                valid_items=tile_items - 1,
                oob_default=-7,
            )
        api.store(group, destination, loaded)

    source = np.arange(32 * items_per_thread, dtype=np.int32)
    destination = cuda.device_array_like(source)
    kernel[1, 32](cuda.to_device(source), destination, items_per_thread)
    expected = np.concatenate((source[1:], np.array([-7], dtype=np.int32)))
    np.testing.assert_array_equal(destination.copy_to_host(), expected)
