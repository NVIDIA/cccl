# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Private runtime and final-cubin probe for group MergeSort providers."""

from __future__ import annotations

import functools
from typing import Any

from examples.cutlass._runtime import require_runtime

BLOCK_DIM = (8, 4, 2)
BLOCK_THREADS = 64
ITEMS_PER_THREAD = 2
TOTAL_ITEMS = BLOCK_THREADS * ITEMS_PER_THREAD
VALID_ITEMS = 93


@functools.lru_cache(maxsize=1)
def make_runner() -> tuple[Any, Any, Any, Any]:
    cutlass, cute, torch, from_dlpack, coop, Int32 = require_runtime(include_int32=True)
    globals()["cute"] = cute
    globals()["coop"] = coop
    globals()["Int32"] = Int32

    @cute.kernel
    def _kernel(
        keys_in: cute.Tensor,
        values_in: cute.Tensor,
        block_keys_out: cute.Tensor,
        block_values_out: cute.Tensor,
        partial_keys_out: cute.Tensor,
        warp_keys_out: cute.Tensor,
        warp_values_out: cute.Tensor,
    ):
        tidx, tidy, tidz = cute.arch.thread_idx()
        linear_tid = tidx + Int32(8) * (tidy + Int32(4) * tidz)
        offset = linear_tid * Int32(ITEMS_PER_THREAD)
        keys = coop.ThreadData.from_values(
            keys_in[offset],
            keys_in[offset + Int32(1)],
            dtype=Int32,
        )
        values = coop.ThreadData.from_values(
            values_in[offset],
            values_in[offset + Int32(1)],
            dtype=Int32,
        )

        block = coop.this_block()
        block_keys, block_values = coop.merge_sort_pairs(
            block,
            keys,
            values,
            descending=True,
        )
        block_keys_out[offset] = block_keys[0]
        block_keys_out[offset + Int32(1)] = block_keys[1]
        block_values_out[offset] = block_values[0]
        block_values_out[offset + Int32(1)] = block_values[1]

        partial_keys = coop.merge_sort_keys(
            block,
            keys,
            valid_items=Int32(VALID_ITEMS),
            oob_default=Int32(1000000),
        )
        partial_keys_out[offset] = partial_keys[0]
        partial_keys_out[offset + Int32(1)] = partial_keys[1]

        warp = coop.this_warp()
        warp_keys, warp_values = coop.merge_sort_pairs(warp, keys, values)
        warp_keys_out[offset] = warp_keys[0]
        warp_keys_out[offset + Int32(1)] = warp_keys[1]
        warp_values_out[offset] = warp_values[0]
        warp_values_out[offset + Int32(1)] = warp_values[1]

    @cute.jit
    def _run(
        keys_in: cute.Tensor,
        values_in: cute.Tensor,
        block_keys_out: cute.Tensor,
        block_values_out: cute.Tensor,
        partial_keys_out: cute.Tensor,
        warp_keys_out: cute.Tensor,
        warp_values_out: cute.Tensor,
    ):
        _kernel(
            keys_in,
            values_in,
            block_keys_out,
            block_values_out,
            partial_keys_out,
            warp_keys_out,
            warp_values_out,
        ).launch(grid=(1, 1, 1), block=BLOCK_DIM)

    return _run, torch, from_dlpack, cutlass


def _warp_expected(keys: Any, values: Any, *, torch: Any) -> tuple[Any, Any]:
    expected_keys = torch.empty_like(keys)
    expected_values = torch.empty_like(values)
    warp_items = 32 * ITEMS_PER_THREAD
    for base in range(0, TOTAL_ITEMS, warp_items):
        tile = keys[base : base + warp_items]
        order = torch.argsort(tile)
        expected_keys[base : base + warp_items] = tile[order]
        expected_values[base : base + warp_items] = values[base : base + warp_items][
            order
        ]
    return expected_keys, expected_values


def run_example() -> dict[str, Any]:
    run, torch, from_dlpack, cutlass = make_runner()
    cutlass.cuda.initialize_cuda_context()

    keys_host = torch.tensor(
        [((index * 37 + 11) % TOTAL_ITEMS) - 64 for index in range(TOTAL_ITEMS)],
        dtype=torch.int32,
    )
    values_host = torch.arange(TOTAL_ITEMS, dtype=torch.int32) * 3 + 5
    keys = keys_host.cuda()
    values = values_host.cuda()
    outputs = [torch.empty_like(keys, device="cuda") for _ in range(5)]
    run(
        from_dlpack(keys),
        from_dlpack(values),
        *(from_dlpack(output) for output in outputs),
    )
    torch.cuda.synchronize()

    descending_order = torch.argsort(keys_host, descending=True)
    torch.testing.assert_close(
        outputs[0].cpu(), keys_host[descending_order], atol=0, rtol=0
    )
    torch.testing.assert_close(
        outputs[1].cpu(), values_host[descending_order], atol=0, rtol=0
    )

    partial_order = torch.argsort(keys_host[:VALID_ITEMS])
    torch.testing.assert_close(
        outputs[2][:VALID_ITEMS].cpu(),
        keys_host[:VALID_ITEMS][partial_order],
        atol=0,
        rtol=0,
    )

    expected_warp_keys, expected_warp_values = _warp_expected(
        keys_host,
        values_host,
        torch=torch,
    )
    torch.testing.assert_close(outputs[3].cpu(), expected_warp_keys, atol=0, rtol=0)
    torch.testing.assert_close(outputs[4].cpu(), expected_warp_values, atol=0, rtol=0)

    return {
        "block_keys": [int(value) for value in outputs[0].cpu().tolist()],
        "partial_keys": [
            int(value) for value in outputs[2][:VALID_ITEMS].cpu().tolist()
        ],
        "warp_keys": [int(value) for value in outputs[3].cpu().tolist()],
    }


if __name__ == "__main__":
    print(run_example())
