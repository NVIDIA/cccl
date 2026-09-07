# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Private runtime and code-generation probe for group Reduce providers."""

from __future__ import annotations

import functools
from typing import Any

from examples.cutlass._runtime import require_runtime


@functools.lru_cache(maxsize=1)
def make_runner() -> tuple[Any, Any, Any, Any]:
    cutlass, cute, torch, from_dlpack, coop, Int32 = require_runtime(include_int32=True)
    globals()["cute"] = cute
    globals()["coop"] = coop
    globals()["Int32"] = Int32

    @cute.kernel
    def _kernel(
        values: cute.Tensor,
        block_root_out: cute.Tensor,
        block_root_second_out: cute.Tensor,
        block_partial_out: cute.Tensor,
        block_array_out: cute.Tensor,
        warp_partial_out: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        block = coop.this_block()
        value = values[tidx]

        block_root = coop.reduce(block, value, broadcast=False)
        block_root_second = coop.reduce(
            block,
            values[tidx + Int32(64)],
            broadcast=False,
        )
        block_partial = coop.sum(
            block,
            value,
            valid_items=Int32(48),
            algorithm="raking",
            broadcast=False,
        )
        items = coop.ThreadData.from_values(
            value,
            values[tidx + Int32(64)],
            dtype=Int32,
        )
        block_array = coop.sum(
            block,
            items,
            algorithm="warp_reductions",
            broadcast=False,
        )
        warp_partial = coop.sum(
            coop.this_warp(),
            value,
            valid_items=24,
            broadcast=False,
        )

        if tidx == 0:
            block_root_out[0] = block_root
            block_root_second_out[0] = block_root_second
            block_partial_out[0] = block_partial
            block_array_out[0] = block_array
        warp_id = tidx // Int32(32)
        lane = tidx - warp_id * Int32(32)
        if lane == 0:
            warp_partial_out[warp_id] = warp_partial

    @cute.jit
    def _run(
        values: cute.Tensor,
        block_root_out: cute.Tensor,
        block_root_second_out: cute.Tensor,
        block_partial_out: cute.Tensor,
        block_array_out: cute.Tensor,
        warp_partial_out: cute.Tensor,
    ):
        _kernel(
            values,
            block_root_out,
            block_root_second_out,
            block_partial_out,
            block_array_out,
            warp_partial_out,
        ).launch(grid=(1, 1, 1), block=(64, 1, 1))

    return _run, torch, from_dlpack, cutlass


def run_example() -> dict[str, Any]:
    run, torch, from_dlpack, cutlass = make_runner()
    cutlass.cuda.initialize_cuda_context()

    values_host = torch.arange(1, 129, dtype=torch.int32)
    values = values_host.cuda()
    block_root = torch.zeros((1,), dtype=torch.int32, device="cuda")
    block_root_second = torch.zeros((1,), dtype=torch.int32, device="cuda")
    block_partial = torch.zeros((1,), dtype=torch.int32, device="cuda")
    block_array = torch.zeros((1,), dtype=torch.int32, device="cuda")
    warp_partial = torch.zeros((2,), dtype=torch.int32, device="cuda")

    run(
        from_dlpack(values),
        from_dlpack(block_root),
        from_dlpack(block_root_second),
        from_dlpack(block_partial),
        from_dlpack(block_array),
        from_dlpack(warp_partial),
    )
    torch.cuda.synchronize()

    actual = {
        "block_root": int(block_root.item()),
        "block_root_second": int(block_root_second.item()),
        "block_partial": int(block_partial.item()),
        "block_array": int(block_array.item()),
        "warp_partial": [int(value) for value in warp_partial.cpu().tolist()],
    }
    expected = {
        "block_root": sum(range(1, 65)),
        "block_root_second": sum(range(65, 129)),
        "block_partial": sum(range(1, 49)),
        "block_array": sum(range(1, 129)),
        "warp_partial": [sum(range(1, 25)), sum(range(33, 57))],
    }
    if actual != expected:
        raise AssertionError(f"unexpected group Reduce results: {actual!r}")
    return actual
