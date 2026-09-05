# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Private runtime and code-generation probe for public-CUB block collectives."""

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
        scalar_values: cute.Tensor,
        item_values: cute.Tensor,
        offset_out: cute.Tensor,
        rotate_out: cute.Tensor,
        up_out: cute.Tensor,
        suffix_out: cute.Tensor,
        heads_out: cute.Tensor,
        tails_out: cute.Tensor,
    ):
        tidx, tidy, tidz = cute.arch.thread_idx()
        linear_tid = tidx + Int32(8) * (tidy + Int32(4) * tidz)
        base = linear_tid * Int32(2)
        group = coop.this_block()

        offset_out[linear_tid] = coop.shuffle(
            group,
            scalar_values[linear_tid],
            mode="offset",
            distance=-1,
        )
        rotate_out[linear_tid] = coop.shuffle(
            group,
            scalar_values[linear_tid],
            mode="rotate",
            distance=67,
        )

        items = coop.ThreadData.from_values(
            item_values[base],
            item_values[base + Int32(1)],
            dtype=Int32,
        )
        suffix = coop.ThreadData(1, dtype=Int32)
        up = coop.shuffle(group, items, mode="up", block_suffix=suffix)
        heads, tails = coop.discontinuity(
            group,
            items,
            mode="heads_and_tails",
            tile_predecessor_item=Int32(-1),
            tile_successor_item=Int32(-1),
        )
        up_out[base] = up[0]
        up_out[base + Int32(1)] = up[1]
        suffix_out[linear_tid] = suffix[0]
        heads_out[base] = heads[0]
        heads_out[base + Int32(1)] = heads[1]
        tails_out[base] = tails[0]
        tails_out[base + Int32(1)] = tails[1]

    @cute.jit
    def _run(
        scalar_values: cute.Tensor,
        item_values: cute.Tensor,
        offset_out: cute.Tensor,
        rotate_out: cute.Tensor,
        up_out: cute.Tensor,
        suffix_out: cute.Tensor,
        heads_out: cute.Tensor,
        tails_out: cute.Tensor,
    ):
        _kernel(
            scalar_values,
            item_values,
            offset_out,
            rotate_out,
            up_out,
            suffix_out,
            heads_out,
            tails_out,
        ).launch(grid=(1, 1, 1), block=(8, 4, 2))

    return _run, torch, from_dlpack, cutlass


def run_example() -> dict[str, Any]:
    run, torch, from_dlpack, cutlass = make_runner()
    cutlass.cuda.initialize_cuda_context()

    scalar_values_host = torch.arange(64, dtype=torch.int32)
    item_values_host = torch.tensor(
        [index // 3 + (index % 11 == 0) for index in range(128)],
        dtype=torch.int32,
    )
    scalar_values = scalar_values_host.cuda()
    item_values = item_values_host.cuda()
    offset_out = torch.zeros_like(scalar_values)
    rotate_out = torch.zeros_like(scalar_values)
    up_out = torch.zeros_like(item_values)
    suffix_out = torch.zeros_like(scalar_values)
    heads_out = torch.zeros_like(item_values)
    tails_out = torch.zeros_like(item_values)
    run(
        from_dlpack(scalar_values),
        from_dlpack(item_values),
        from_dlpack(offset_out),
        from_dlpack(rotate_out),
        from_dlpack(up_out),
        from_dlpack(suffix_out),
        from_dlpack(heads_out),
        from_dlpack(tails_out),
    )
    torch.cuda.synchronize()

    expected_offset = scalar_values_host.clone()
    expected_offset[1:] = scalar_values_host[:-1]
    expected_rotate = torch.roll(scalar_values_host, shifts=-3)
    expected_up = item_values_host.clone()
    expected_up[1:] = item_values_host[:-1]
    expected_suffix = torch.full_like(
        scalar_values_host,
        int(item_values_host[-1].item()),
    )
    expected_heads = torch.zeros_like(item_values_host)
    expected_tails = torch.zeros_like(item_values_host)
    expected_heads[0] = int(item_values_host[0] != -1)
    expected_heads[1:] = (item_values_host[:-1] != item_values_host[1:]).to(torch.int32)
    expected_tails[:-1] = (item_values_host[:-1] != item_values_host[1:]).to(
        torch.int32
    )
    expected_tails[-1] = int(item_values_host[-1] != -1)
    for output, expected in (
        (offset_out, expected_offset),
        (rotate_out, expected_rotate),
        (up_out, expected_up),
        (suffix_out, expected_suffix),
        (heads_out, expected_heads),
        (tails_out, expected_tails),
    ):
        torch.testing.assert_close(output.cpu(), expected, atol=0, rtol=0)

    return {
        "offset_sum": int(offset_out.sum().item()),
        "rotate_sum": int(rotate_out.sum().item()),
        "up_sum": int(up_out.sum().item()),
        "heads_sum": int(heads_out.sum().item()),
        "tails_sum": int(tails_out.sum().item()),
    }
