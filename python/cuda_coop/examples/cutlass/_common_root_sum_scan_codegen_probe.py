# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Private final-link probe for common-root CUTLASS Reduce and Scan aliases."""

from __future__ import annotations

import functools
from typing import Any

from examples.cutlass._runtime import require_runtime

_BLOCK_THREADS = 32
_SEGMENT_COUNT = 30
_INT32_LOWEST = -2_147_483_648


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
    def _kernel(values: cute.Tensor, output: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        common_group = common_coop.this_block()
        qualified_group = cutlass_coop.this_block()
        value = values[tidx]

        output[0 * _BLOCK_THREADS + tidx] = common_coop.sum(common_group, value)
        output[1 * _BLOCK_THREADS + tidx] = cutlass_coop.sum(qualified_group, value)
        output[2 * _BLOCK_THREADS + tidx] = common_coop.scan(common_group, value)
        output[3 * _BLOCK_THREADS + tidx] = common_coop.exclusive_sum(
            common_group, value
        )
        output[4 * _BLOCK_THREADS + tidx] = cutlass_coop.exclusive_sum(
            qualified_group, value
        )
        output[5 * _BLOCK_THREADS + tidx] = common_coop.scan(
            common_group, value, mode="inclusive"
        )
        output[6 * _BLOCK_THREADS + tidx] = common_coop.inclusive_sum(
            common_group, value
        )
        output[7 * _BLOCK_THREADS + tidx] = cutlass_coop.inclusive_sum(
            qualified_group, value
        )
        output[8 * _BLOCK_THREADS + tidx] = common_coop.exclusive_scan(
            common_group,
            value,
            scan_op="max",
            initial_value=_INT32_LOWEST,
        )
        output[9 * _BLOCK_THREADS + tidx] = cutlass_coop.exclusive_scan(
            qualified_group,
            value,
            scan_op="max",
            initial_value=_INT32_LOWEST,
        )
        output[10 * _BLOCK_THREADS + tidx] = common_coop.inclusive_scan(
            common_group,
            value,
            scan_op="max",
        )
        output[11 * _BLOCK_THREADS + tidx] = cutlass_coop.inclusive_scan(
            qualified_group,
            value,
            scan_op="max",
        )
        common_block_max = common_coop.reduce(
            common_group,
            value,
            binary_op="max",
            broadcast=False,
            algorithm="raking",
        )
        qualified_block_max = cutlass_coop.reduce(
            qualified_group,
            value,
            binary_op="max",
            broadcast=False,
            algorithm="raking",
        )
        if tidx == 0:
            output[26 * _BLOCK_THREADS] = common_block_max
            output[27 * _BLOCK_THREADS] = qualified_block_max

        common_warp = common_coop.this_warp()
        qualified_warp = cutlass_coop.this_warp()
        output[12 * _BLOCK_THREADS + tidx] = common_coop.sum(common_warp, value)
        output[13 * _BLOCK_THREADS + tidx] = cutlass_coop.sum(qualified_warp, value)
        output[14 * _BLOCK_THREADS + tidx] = common_coop.scan(common_warp, value)
        output[15 * _BLOCK_THREADS + tidx] = cutlass_coop.scan(qualified_warp, value)
        output[16 * _BLOCK_THREADS + tidx] = common_coop.exclusive_sum(
            common_warp, value
        )
        output[17 * _BLOCK_THREADS + tidx] = cutlass_coop.exclusive_sum(
            qualified_warp, value
        )
        output[18 * _BLOCK_THREADS + tidx] = common_coop.scan(
            common_warp, value, mode="inclusive"
        )
        output[19 * _BLOCK_THREADS + tidx] = cutlass_coop.scan(
            qualified_warp, value, mode="inclusive"
        )
        output[20 * _BLOCK_THREADS + tidx] = common_coop.inclusive_sum(
            common_warp, value
        )
        output[21 * _BLOCK_THREADS + tidx] = cutlass_coop.inclusive_sum(
            qualified_warp, value
        )
        output[22 * _BLOCK_THREADS + tidx] = common_coop.exclusive_scan(
            common_warp,
            value,
            scan_op="max",
            initial_value=_INT32_LOWEST,
        )
        output[23 * _BLOCK_THREADS + tidx] = cutlass_coop.exclusive_scan(
            qualified_warp,
            value,
            scan_op="max",
            initial_value=_INT32_LOWEST,
        )
        output[24 * _BLOCK_THREADS + tidx] = common_coop.inclusive_scan(
            common_warp,
            value,
            scan_op="max",
        )
        output[25 * _BLOCK_THREADS + tidx] = cutlass_coop.inclusive_scan(
            qualified_warp,
            value,
            scan_op="max",
        )
        common_warp_max = common_coop.reduce(
            common_warp,
            value,
            binary_op="max",
            broadcast=False,
            valid_items=24,
        )
        qualified_warp_max = cutlass_coop.reduce(
            qualified_warp,
            value,
            binary_op="max",
            broadcast=False,
            valid_items=24,
        )
        if tidx % 32 == 0:
            output[28 * _BLOCK_THREADS + tidx] = common_warp_max
            output[29 * _BLOCK_THREADS + tidx] = qualified_warp_max

    @cute.jit
    def _run(values: cute.Tensor, output: cute.Tensor):
        _kernel(values, output).launch(
            grid=(1, 1, 1),
            block=(_BLOCK_THREADS, 1, 1),
        )

    return _run, torch, from_dlpack, cutlass


def run_example() -> dict[str, Any]:
    run, torch, from_dlpack, cutlass = make_runner()
    cutlass.cuda.initialize_cuda_context()

    values_host = (torch.arange(_BLOCK_THREADS, dtype=torch.int32) * 17 % 29) - 14
    values = values_host.cuda()
    output = torch.zeros(
        (_SEGMENT_COUNT * _BLOCK_THREADS,), dtype=torch.int32, device="cuda"
    )
    run(from_dlpack(values), from_dlpack(output))
    torch.cuda.synchronize()

    inclusive = torch.cumsum(values_host.to(torch.int64), dim=0).to(torch.int32)
    exclusive = inclusive - values_host
    inclusive_max = torch.cummax(values_host, dim=0).values
    exclusive_max = torch.cat(
        (torch.tensor([_INT32_LOWEST], dtype=torch.int32), inclusive_max[:-1])
    )
    block_max = torch.zeros_like(values_host)
    block_max[0] = values_host.max()
    warp_max = torch.zeros_like(values_host)
    warp_max[0] = values_host[:24].max()
    expected = torch.stack(
        (
            torch.full_like(values_host, int(values_host.sum())),
            torch.full_like(values_host, int(values_host.sum())),
            exclusive,
            exclusive,
            exclusive,
            inclusive,
            inclusive,
            inclusive,
            exclusive_max,
            exclusive_max,
            inclusive_max,
            inclusive_max,
            torch.full_like(values_host, int(values_host.sum())),
            torch.full_like(values_host, int(values_host.sum())),
            exclusive,
            exclusive,
            exclusive,
            exclusive,
            inclusive,
            inclusive,
            inclusive,
            inclusive,
            exclusive_max,
            exclusive_max,
            inclusive_max,
            inclusive_max,
            block_max,
            block_max,
            warp_max,
            warp_max,
        )
    ).reshape(-1)
    torch.testing.assert_close(output.cpu(), expected, atol=0, rtol=0)

    return {
        "sum": int(output[0].item()),
        "exclusive_tail": int(output[4 * _BLOCK_THREADS - 1].item()),
        "inclusive_tail": int(output[8 * _BLOCK_THREADS - 1].item()),
    }
