# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Execute the documented accumulation of fresh per-tile histograms.

Keep the marked example self-contained for Sphinx. Each call counts one full
tile; explicit per-thread addition retains totals across calls. Striped Store
maps each accumulated counter back to its bin, with no padded bins written.
"""

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


def test_histogram_accumulation_example():
    """Check explicit counter accumulation across three complete tiles."""

    # histogram-accumulation-example-begin
    import numpy as np
    from numba_cuda_mlir import cuda

    from cuda import coop

    @cuda.jit
    def histogram_tiles(source, tile_count, destination, items_per_thread):
        block = coop.this_block()
        samples = coop.ThreadData(items_per_thread)
        total = coop.ThreadData(items_per_thread=2)
        for i in range(2):
            total[i] = np.int64(0)
        for tile in range(tile_count):
            coop.load(
                block,
                source,
                samples,
                offset=tile * cuda.blockDim.x * items_per_thread,
            )
            counts = coop.histogram(
                block,
                samples,
                bins=65,
                bins_per_thread=2,
                counter_dtype=np.int64,
            )
            for i in range(2):
                total[i] += counts[i]
        coop.store(
            block, destination, total, algorithm="striped", valid_items=65
        )

    for items_per_thread in (1, 4):
        source = (np.arange(3 * 64 * items_per_thread) % 65).astype(np.int32)
        destination = np.empty(65, dtype=np.int64)
        histogram_tiles[1, 64](source, 3, destination, items_per_thread)
        cuda.synchronize()
        np.testing.assert_array_equal(
            destination, np.bincount(source, minlength=65)
        )
    # histogram-accumulation-example-end
