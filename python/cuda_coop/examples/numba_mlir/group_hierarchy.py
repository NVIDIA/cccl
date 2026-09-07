# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Block and mapped-warp groups with the Numba-CUDA-MLIR backend."""

import numpy as np
from numba_cuda_mlir import cuda

import cuda.coop.numba_mlir as coop

THREADS = 32
LANES_PER_GROUP = 8


# docs: start numba-mlir-group-hierarchy
@cuda.jit
def group_hierarchy_kernel(values, block_total, lane_group_totals):
    tid = cuda.threadIdx.x
    block = coop.this_block()
    lanes = coop.this_warp().group_by(LANES_PER_GROUP)

    total = coop.reduce(block, values[tid])
    lane_total = coop.reduce(lanes, values[tid])

    if block.rank() == 0:
        block_total[0] = total
    if lanes.rank() == 0:
        lane_group_totals[tid // LANES_PER_GROUP] = lane_total


# docs: end numba-mlir-group-hierarchy


def run_example() -> tuple[int, np.ndarray]:
    """Run one block and return its block and mapped-group sums."""

    values = np.arange(1, THREADS + 1, dtype=np.int32)
    block_total = np.zeros(1, dtype=np.int32)
    lane_group_totals = np.zeros(THREADS // LANES_PER_GROUP, dtype=np.int32)

    group_hierarchy_kernel[1, THREADS](
        values,
        block_total,
        lane_group_totals,
    )

    expected_block = int(np.sum(values, dtype=np.int64))
    expected_groups = values.reshape(-1, LANES_PER_GROUP).sum(axis=1)
    assert int(block_total[0]) == expected_block
    np.testing.assert_array_equal(lane_group_totals, expected_groups)
    return expected_block, lane_group_totals


def main() -> int:
    block_total, lane_group_totals = run_example()
    print(f"block total: {block_total}")
    print(f"lane-group totals: {lane_group_totals}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
