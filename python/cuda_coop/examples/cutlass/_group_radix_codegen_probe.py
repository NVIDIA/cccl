# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Private runtime and final-cubin probe for group-first radix providers."""

from __future__ import annotations

import functools
from typing import Any

from examples.cutlass._runtime import require_runtime

BLOCK_DIM = (8, 4, 2)
BLOCK_THREADS = 64
ITEMS_PER_THREAD = 2
TOTAL_ITEMS = BLOCK_THREADS * ITEMS_PER_THREAD


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
        sorted_keys_out: cute.Tensor,
        sorted_values_out: cute.Tensor,
        ranks_out: cute.Tensor,
        prefix_out: cute.Tensor,
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
        sorted_keys, sorted_values = coop.radix_sort_pairs(
            block,
            keys,
            values,
            begin_bit=0,
            end_bit=8,
        )
        prefix = coop.ThreadData(1, dtype=Int32)
        ranks = coop.radix_rank(
            block,
            keys,
            begin_bit=0,
            end_bit=4,
            exclusive_digit_prefix=prefix,
        )

        sorted_keys_out[offset] = sorted_keys[0]
        sorted_keys_out[offset + Int32(1)] = sorted_keys[1]
        sorted_values_out[offset] = sorted_values[0]
        sorted_values_out[offset + Int32(1)] = sorted_values[1]
        ranks_out[offset] = ranks[0]
        ranks_out[offset + Int32(1)] = ranks[1]
        prefix_out[linear_tid] = prefix[0]

    @cute.jit
    def _run(
        keys_in: cute.Tensor,
        values_in: cute.Tensor,
        sorted_keys_out: cute.Tensor,
        sorted_values_out: cute.Tensor,
        ranks_out: cute.Tensor,
        prefix_out: cute.Tensor,
    ):
        _kernel(
            keys_in,
            values_in,
            sorted_keys_out,
            sorted_values_out,
            ranks_out,
            prefix_out,
        ).launch(grid=(1, 1, 1), block=BLOCK_DIM)

    return _run, torch, from_dlpack, cutlass


def run_example() -> dict[str, Any]:
    run, torch, from_dlpack, cutlass = make_runner()
    cutlass.cuda.initialize_cuda_context()

    keys_host = torch.arange(-64, 64, dtype=torch.int32)
    values_host = torch.arange(TOTAL_ITEMS, dtype=torch.int32) * 7 + 3
    keys = keys_host.cuda()
    values = values_host.cuda()
    sorted_keys = torch.zeros_like(keys)
    sorted_values = torch.zeros_like(values)
    ranks = torch.zeros_like(keys)
    prefix = torch.zeros((BLOCK_THREADS,), dtype=torch.int32, device="cuda")
    run(
        from_dlpack(keys),
        from_dlpack(values),
        from_dlpack(sorted_keys),
        from_dlpack(sorted_values),
        from_dlpack(ranks),
        from_dlpack(prefix),
    )
    torch.cuda.synchronize()

    sort_order = sorted(
        range(TOTAL_ITEMS),
        key=lambda index: (int(keys_host[index]) & 0xFF, index),
    )
    expected_sorted_keys = keys_host[sort_order]
    expected_sorted_values = values_host[sort_order]
    rank_order = sorted(
        range(TOTAL_ITEMS),
        key=lambda index: (int(keys_host[index]) & 0xF, index),
    )
    expected_ranks = torch.empty_like(keys_host)
    for rank, index in enumerate(rank_order):
        expected_ranks[index] = rank
    digit_counts = [0] * 16
    for key in keys_host:
        digit_counts[int(key) & 0xF] += 1
    expected_prefix = torch.full((BLOCK_THREADS,), -1, dtype=torch.int32)
    running = 0
    for digit, count in enumerate(digit_counts):
        expected_prefix[digit] = running
        running += count

    torch.testing.assert_close(sorted_keys.cpu(), expected_sorted_keys, atol=0, rtol=0)
    torch.testing.assert_close(
        sorted_values.cpu(), expected_sorted_values, atol=0, rtol=0
    )
    torch.testing.assert_close(ranks.cpu(), expected_ranks, atol=0, rtol=0)
    torch.testing.assert_close(prefix.cpu(), expected_prefix, atol=0, rtol=0)
    return {
        "sorted_keys": [int(value) for value in sorted_keys.cpu().tolist()],
        "ranks": [int(value) for value in ranks.cpu().tolist()],
        "prefix": [int(value) for value in prefix.cpu().tolist()],
    }
