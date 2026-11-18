# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Load directly into disjoint shared rows, then consume across threads."""

import numpy as np
import pytest

pytest.importorskip("numba_cuda_mlir")
from numba_cuda_mlir import cuda

import cuda.coop.numba_mlir as numba_coop
from cuda import coop

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


@pytest.mark.parametrize("api", [coop, numba_coop], ids=["common", "qualified"])
@pytest.mark.parametrize("items_per_thread", [1, 4])
@pytest.mark.parametrize(
    "scope,algorithm",
    [
        ("block", "direct"),
        ("block", "striped"),
        ("block", "transpose"),
        ("warp", "striped"),
        ("logical", "transpose"),
    ],
)
def test_shared_destination_rows(api, items_per_thread, scope, algorithm):
    threads = 64
    width = 64 if scope == "block" else 32 if scope == "warp" else 8
    tile_items = threads * items_per_thread
    capacity = tile_items + 4
    valid = width * items_per_thread - 1

    @cuda.jit
    def kernel(source, result, items_per_thread):
        shared = cuda.shared.array(capacity, source.dtype)
        destination = shared[2:-2]
        if scope == "block":
            group = api.this_block()
        elif scope == "warp":
            group = api.this_warp()
        else:
            group = api.this_warp().group_by(8)
        api.load(
            group,
            source,
            destination,
            items_per_thread=items_per_thread,
            algorithm=algorithm,
            offset=3,
            valid_items=valid,
            oob_default=-97,
        )
        cuda.syncthreads()
        thread = cuda.threadIdx.x + cuda.blockDim.x * cuda.threadIdx.y
        neighbor = (thread + 1) % threads
        for item in range(items_per_thread):
            result[thread * items_per_thread + item] = destination[
                neighbor * items_per_thread + item
            ]

    source = np.arange(tile_items + 3, dtype=np.int32) * 11 - 3
    result = cuda.device_array(tile_items, dtype=np.int32)
    kernel[1, (8, 8)](cuda.to_device(source), result, items_per_thread)
    expected_rows = np.empty((threads, items_per_thread), dtype=np.int32)
    for thread in range(threads):
        group_base = thread // width * width * items_per_thread
        lane = thread % width
        for item in range(items_per_thread):
            index = (
                item * width + lane
                if algorithm == "striped"
                else lane * items_per_thread + item
            )
            expected_rows[thread, item] = (
                source[3 + group_base + index] if index < valid else -97
            )
    np.testing.assert_array_equal(
        result.copy_to_host(), np.roll(expected_rows, -1, axis=0).ravel()
    )


def test_shared_load_example():
    # shared-load-example-begin
    import numpy as np
    from numba_cuda_mlir import cuda

    from cuda import coop

    threads = 128
    items_per_thread = 4
    tile_items = threads * items_per_thread

    @cuda.jit
    def stage(source, result, items_per_thread):
        shared = cuda.shared.array(tile_items, source.dtype)
        coop.load(
            coop.this_block(),
            source,
            shared,
            items_per_thread=items_per_thread,
        )
        # Publish each thread's row before reading a neighbor's values.
        cuda.syncthreads()
        for item in range(items_per_thread):
            index = cuda.threadIdx.x * items_per_thread + item
            result[index] = shared[(index + 1) % tile_items]

    source = np.arange(tile_items, dtype=np.float64) + 2**-30
    result = cuda.device_array(tile_items, dtype=source.dtype)
    stage[1, threads](cuda.to_device(source), result, items_per_thread)
    np.testing.assert_array_equal(result.copy_to_host(), np.roll(source, -1))
    # shared-load-example-end
