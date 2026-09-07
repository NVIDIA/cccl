# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Test-only probe for common-root and private CUTLASS load/store parity."""

from __future__ import annotations

import functools
from typing import Any

from examples.cutlass._runtime import require_runtime


@functools.lru_cache(maxsize=1)
def make_runner() -> tuple[Any, Any, Any, Any]:
    cutlass, cute, torch, from_dlpack, cutlass_coop, _ = require_runtime(
        include_int32=True
    )
    from cuda import coop as common_coop

    globals()["cute"] = cute
    globals()["common_coop"] = common_coop
    globals()["cutlass_coop"] = cutlass_coop

    @cute.kernel
    def _kernel(
        values_in: cute.Tensor,
        root_block_out: cute.Tensor,
        scoped_block_out: cute.Tensor,
        root_warp_out: cute.Tensor,
        scoped_warp_out: cute.Tensor,
    ):
        block = common_coop.this_block()
        root_block_items = common_coop.ThreadData(2)
        loaded_block = common_coop.load(
            block,
            values_in,
            root_block_items,
            offset=3,
        )
        common_coop.store(block, root_block_out, loaded_block, offset=5)

        scoped_block_items = cutlass_coop.ThreadData(2)
        cutlass_coop._block.load(values_in, scoped_block_items, offset=3)
        cutlass_coop._block.store(scoped_block_out, scoped_block_items, offset=5)

        warp = common_coop.this_warp()
        root_warp_items = common_coop.ThreadData(2)
        loaded_warp = common_coop.load(
            warp,
            values_in,
            root_warp_items,
            algorithm="striped",
            offset=7,
        )
        common_coop.store(
            warp,
            root_warp_out,
            loaded_warp,
            algorithm="striped",
            offset=9,
        )

        scoped_warp_items = cutlass_coop.ThreadData(2)
        cutlass_coop._warp.load(
            values_in,
            scoped_warp_items,
            algorithm="striped",
            offset=7,
        )
        cutlass_coop._warp.store(
            scoped_warp_out,
            scoped_warp_items,
            algorithm="striped",
            offset=9,
        )

    @cute.jit
    def _run(
        values_in: cute.Tensor,
        root_block_out: cute.Tensor,
        scoped_block_out: cute.Tensor,
        root_warp_out: cute.Tensor,
        scoped_warp_out: cute.Tensor,
    ):
        _kernel(
            values_in,
            root_block_out,
            scoped_block_out,
            root_warp_out,
            scoped_warp_out,
        ).launch(grid=(1, 1, 1), block=(8, 4, 2))

    return _run, torch, from_dlpack, cutlass


def run_example() -> dict[str, Any]:
    run, torch, from_dlpack, cutlass = make_runner()
    cutlass.cuda.initialize_cuda_context()

    values_host = torch.arange(140, dtype=torch.int32)
    values_in = values_host.cuda()
    outputs = [
        torch.full((144,), -999, dtype=torch.int32, device="cuda") for _ in range(4)
    ]
    run(from_dlpack(values_in), *(from_dlpack(output) for output in outputs))
    torch.cuda.synchronize()

    expected_block = torch.full((144,), -999, dtype=torch.int32)
    expected_block[5:133] = values_host[3:131]
    expected_warp = torch.full((144,), -999, dtype=torch.int32)
    expected_warp[9:137] = values_host[7:135]
    for output, expected in (
        (outputs[0], expected_block),
        (outputs[1], expected_block),
        (outputs[2], expected_warp),
        (outputs[3], expected_warp),
    ):
        torch.testing.assert_close(output.cpu(), expected, atol=0, rtol=0)

    return {
        "root_block_sum": int(outputs[0].sum().item()),
        "scoped_block_sum": int(outputs[1].sum().item()),
        "root_warp_sum": int(outputs[2].sum().item()),
        "scoped_warp_sum": int(outputs[3].sum().item()),
    }
