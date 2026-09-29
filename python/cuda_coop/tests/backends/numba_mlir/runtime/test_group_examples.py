# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Executable examples included in the thread-group API documentation."""

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


def test_group_queries_example():
    # queries-example-begin
    import numpy as np
    from numba_cuda_mlir import cuda, types

    import cuda.coop.numba_mlir as numba_coop  # noqa: F401 -- activate the backend
    from cuda import coop

    @cuda.jit
    def coordinates(output):
        thread = coop.this_thread()
        warp = coop.this_warp()
        block = coop.this_block()
        grid = coop.this_grid()
        index = cuda.grid(1)

        thread.sync()
        warp.sync_aligned()
        block.sync()
        output[0, index] = thread.rank("block")
        output[1, index] = warp.rank()
        output[2, index] = block.rank_as(types.int64, "grid")
        output[3, index] = block.count_as(types.int64)
        output[4, index] = grid.rank()
        output[5, index] = grid.count()

    output = cuda.device_array((6, 192), dtype=np.int64)
    coordinates[2, 96](output)
    observed = output.copy_to_host()
    indices = np.arange(192)
    expected = np.stack(
        (
            indices % 96,  # Thread within its block.
            indices % 32,  # Thread within its physical warp.
            indices // 96,  # Block within the grid.
            np.full(192, 96),  # Threads per block.
            indices,  # Thread within the grid.
            np.full(192, 192),  # Threads in the grid.
        )
    )
    np.testing.assert_array_equal(observed, expected)
    # queries-example-end


def test_group_by_example():
    # partition-example-begin
    import numpy as np
    from numba_cuda_mlir import cuda

    import cuda.coop.numba_mlir as numba_coop  # noqa: F401 -- activate the backend
    from cuda import coop

    @cuda.jit
    def partitions(output):
        index = cuda.threadIdx.x
        lanes = coop.this_warp().group_by(8)
        pair = coop.this_block().group_by(2, exhaustive=False)

        lanes.sync_aligned()
        output[0, index] = lanes.rank()
        output[1, index] = lanes.count()
        member = pair.is_member()
        output[2, index] = member
        if member:
            output[3, index] = pair.rank()
        else:
            output[3, index] = -1

    output = cuda.device_array((4, 96), dtype=np.int64)
    partitions[1, 96](output)
    indices = np.arange(96)
    expected = np.stack(
        (
            indices % 8,
            np.full(96, 8),
            indices < 64,
            np.where(indices < 64, indices, -1),
        )
    )
    np.testing.assert_array_equal(output.copy_to_host(), expected)
    # partition-example-end


@pytest.mark.skipif(
    cuda.get_current_device().compute_capability < (9, 0),
    reason="thread-block clusters require compute capability 9.0 or newer",
)
def test_cluster_group_example():
    # cluster-example-begin
    import numpy as np
    from numba_cuda_mlir import cuda

    import cuda.coop.numba_mlir as numba_coop  # noqa: F401 -- activate the backend
    from cuda import coop

    @cuda.jit
    def cluster_coordinates(output):
        cluster = coop.this_cluster()
        cluster.sync()
        index = cuda.grid(1)
        output[0, index] = cluster.rank("block")
        output[1, index] = cluster.count("block")
        output[2, index] = cluster.is_member()

    output = cuda.device_array((3, 64), dtype=np.int64)
    configured = cluster_coordinates.configure((2, 1, 1), (32, 1, 1), cluster=(2, 1, 1))
    configured(output)
    expected = np.stack(
        (np.arange(64) // 32, np.full(64, 2), np.ones(64, dtype=np.int64))
    )
    np.testing.assert_array_equal(output.copy_to_host(), expected)
    # cluster-example-end
