# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Portable root Load, Reduce, and Store with the CUTLASS backend."""

from __future__ import annotations

import functools
from typing import Any

from examples.cutlass._runtime import require_runtime

THREADS = 32
ITEMS_PER_THREAD = 2
TILE_ITEMS = THREADS * ITEMS_PER_THREAD


@functools.lru_cache(maxsize=1)
def make_runner() -> tuple[Any, Any, Any, Any]:
    """Build and return the CuTe JIT runner plus runtime helpers."""

    cutlass, cute, torch, from_dlpack, _ = require_runtime()
    globals()["cute"] = cute

    # docs: start portable-root-sum
    import numpy as np

    from cuda import coop

    @cute.kernel
    def portable_root_sum_kernel(
        values: cute.Tensor,
        copied: cute.Tensor,
        totals: cute.Tensor,
    ):
        block = coop.this_block()
        items = coop.ThreadData(2, dtype=np.int32)
        loaded = coop.load(block, values, items)
        coop.store(block, copied, loaded)
        total = coop.reduce(block, loaded)
        if block.rank() == 0:
            totals[0] = total

    # docs: end portable-root-sum

    globals()["coop"] = coop

    @cute.jit
    def _run(
        values: cute.Tensor,
        copied: cute.Tensor,
        totals: cute.Tensor,
    ):
        portable_root_sum_kernel(values, copied, totals).launch(
            grid=(1, 1, 1),
            block=(THREADS, 1, 1),
        )

    return _run, torch, from_dlpack, cutlass


def run_example() -> int:
    """Run one block, validate its round trip and sum, and return the sum."""

    run, torch, from_dlpack, cutlass = make_runner()
    cutlass.cuda.initialize_cuda_context()

    values_host = torch.arange(1, TILE_ITEMS + 1, dtype=torch.int32)
    values = values_host.cuda()
    copied = torch.zeros_like(values)
    totals = torch.zeros((1,), dtype=torch.int32, device="cuda")

    run(from_dlpack(values), from_dlpack(copied), from_dlpack(totals))
    torch.cuda.synchronize()

    expected_total = int(values_host.sum().item())
    torch.testing.assert_close(copied.cpu(), values_host, atol=0, rtol=0)
    torch.testing.assert_close(
        totals.cpu(),
        torch.tensor([expected_total], dtype=torch.int32),
        atol=0,
        rtol=0,
    )
    return expected_total


def main() -> int:
    print(run_example())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
