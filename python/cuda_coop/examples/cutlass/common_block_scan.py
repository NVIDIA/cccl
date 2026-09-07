# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Run a portable block load, exclusive scan, and store with CUTLASS."""

from __future__ import annotations

import cutlass
import numpy as np
import torch
from cutlass import cute
from cutlass.cute.runtime import from_dlpack

from cuda import coop

THREADS = 32
ITEMS_PER_THREAD = 2
TILE_ITEMS = THREADS * ITEMS_PER_THREAD


# docs: start cutlass-common-block-scan
@cute.kernel
def block_scan_kernel(values: cute.Tensor, prefixes: cute.Tensor):
    block = coop.this_block()
    items = coop.ThreadData(ITEMS_PER_THREAD, dtype=np.int32)
    loaded = coop.load(block, values, items)
    scanned = coop.exclusive_sum(block, loaded)
    coop.store(block, prefixes, scanned)


# docs: end cutlass-common-block-scan


@cute.jit
def launch(values: cute.Tensor, prefixes: cute.Tensor):
    block_scan_kernel(values, prefixes).launch(
        grid=(1, 1, 1),
        block=(THREADS, 1, 1),
    )


def run_example() -> torch.Tensor:
    """Run one tile and return its exclusive prefix sum on the host."""

    cutlass.cuda.initialize_cuda_context()
    values_host = torch.arange(1, TILE_ITEMS + 1, dtype=torch.int32)
    values = values_host.cuda()
    prefixes = torch.zeros_like(values)

    launch(from_dlpack(values), from_dlpack(prefixes))
    torch.cuda.synchronize()

    expected = torch.zeros_like(values_host)
    expected[1:] = torch.cumsum(values_host[:-1], dim=0)
    observed = prefixes.cpu()
    torch.testing.assert_close(observed, expected, atol=0, rtol=0)
    return observed


def main() -> int:
    print(run_example())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
