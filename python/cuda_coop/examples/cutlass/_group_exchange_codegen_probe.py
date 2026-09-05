# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Private runtime and code-generation probe for group Exchange providers."""

from __future__ import annotations

import functools
from typing import Any

from examples.cutlass._runtime import require_runtime

_BLOCK_DIM = (8, 4, 2)
_BLOCK_THREADS = 64
_ITEMS_PER_THREAD = 5
_TILE_ITEMS = _BLOCK_THREADS * _ITEMS_PER_THREAD
_SEGMENT_COUNT = 10


def _store_items(output, segment: int, thread_rank, items) -> None:
    offset = segment * _TILE_ITEMS + thread_rank * _ITEMS_PER_THREAD
    output[offset] = items[0]
    output[offset + 1] = items[1]
    output[offset + 2] = items[2]
    output[offset + 3] = items[3]
    output[offset + 4] = items[4]


def _exchange_oracle(values, *, group_threads: int, mode: str):
    """Return an indexing-only reference for one portable Exchange mode."""

    result = values.new_empty(values.shape)
    group_items = group_threads * _ITEMS_PER_THREAD
    for group_base in range(0, values.numel(), group_items):
        for rank in range(group_threads):
            for item in range(_ITEMS_PER_THREAD):
                output_index = rank * _ITEMS_PER_THREAD + item
                if mode == "blocked_to_striped":
                    input_index = item * group_threads + rank
                elif mode == "striped_to_blocked":
                    input_index = (
                        output_index % group_threads
                    ) * _ITEMS_PER_THREAD + output_index // group_threads
                else:
                    raise AssertionError(f"unexpected Exchange mode: {mode}")
                result[group_base + output_index] = values[group_base + input_index]
    return result


@functools.lru_cache(maxsize=1)
def make_runner() -> tuple[Any, Any, Any, Any]:
    cutlass, cute, torch, from_dlpack, cutlass_coop, Int32 = require_runtime(
        include_int32=True
    )
    from cuda import coop as common_coop

    globals()["cute"] = cute
    globals()["common_coop"] = common_coop
    globals()["cutlass_coop"] = cutlass_coop
    globals()["Int32"] = Int32

    @cute.kernel
    def _kernel(values: cute.Tensor, output: cute.Tensor):
        tidx, tidy, tidz = cute.arch.thread_idx()
        tidx = tidx + Int32(8) * (tidy + Int32(4) * tidz)
        input_offset = tidx * _ITEMS_PER_THREAD
        common_items = common_coop.ThreadData(_ITEMS_PER_THREAD, dtype=Int32)
        qualified_items = cutlass_coop.ThreadData(_ITEMS_PER_THREAD, dtype=Int32)
        common_items[0] = values[input_offset]
        common_items[1] = values[input_offset + 1]
        common_items[2] = values[input_offset + 2]
        common_items[3] = values[input_offset + 3]
        common_items[4] = values[input_offset + 4]
        qualified_items[0] = values[input_offset]
        qualified_items[1] = values[input_offset + 1]
        qualified_items[2] = values[input_offset + 2]
        qualified_items[3] = values[input_offset + 3]
        qualified_items[4] = values[input_offset + 4]

        common_block = common_coop.this_block()
        qualified_block = cutlass_coop.this_block()
        common_block_blocked = common_coop.exchange(common_block, common_items)
        qualified_block_blocked = cutlass_coop.exchange(
            qualified_block,
            qualified_items,
        )
        common_block_striped = common_coop.exchange(
            common_block,
            common_items,
            mode="blocked_to_striped",
        )
        qualified_block_striped = cutlass_coop.exchange(
            qualified_block,
            qualified_items,
            mode="blocked_to_striped",
        )

        common_warp = common_coop.this_warp()
        qualified_warp = cutlass_coop.this_warp()
        common_warp_blocked = common_coop.exchange(common_warp, common_items)
        qualified_warp_blocked = cutlass_coop.exchange(
            qualified_warp,
            qualified_items,
        )
        common_warp_striped = common_coop.exchange(
            common_warp,
            common_items,
            mode="blocked_to_striped",
        )
        qualified_warp_striped = cutlass_coop.exchange(
            qualified_warp,
            qualified_items,
            mode="blocked_to_striped",
        )

        # Observe both sources only after every transforming operation. This
        # directly qualifies the common V1 non-mutation rule.
        _store_items(output, 0, tidx, common_items)
        _store_items(output, 1, tidx, qualified_items)
        _store_items(output, 2, tidx, common_block_blocked)
        _store_items(output, 3, tidx, qualified_block_blocked)
        _store_items(output, 4, tidx, common_block_striped)
        _store_items(output, 5, tidx, qualified_block_striped)
        _store_items(output, 6, tidx, common_warp_blocked)
        _store_items(output, 7, tidx, qualified_warp_blocked)
        _store_items(output, 8, tidx, common_warp_striped)
        _store_items(output, 9, tidx, qualified_warp_striped)

    @cute.jit
    def _run(values: cute.Tensor, output: cute.Tensor):
        _kernel(values, output).launch(grid=(1, 1, 1), block=_BLOCK_DIM)

    return _run, torch, from_dlpack, cutlass


def run_example() -> dict[str, Any]:
    run, torch, from_dlpack, cutlass = make_runner()
    cutlass.cuda.initialize_cuda_context()

    values_host = torch.arange(_TILE_ITEMS, dtype=torch.int32) * 17 - 91
    values = values_host.cuda()
    output = torch.zeros(
        (_SEGMENT_COUNT * _TILE_ITEMS,),
        dtype=torch.int32,
        device="cuda",
    )
    run(from_dlpack(values), from_dlpack(output))
    torch.cuda.synchronize()

    segments = output.cpu().reshape(_SEGMENT_COUNT, _TILE_ITEMS)
    block_blocked = _exchange_oracle(
        values_host,
        group_threads=_BLOCK_THREADS,
        mode="striped_to_blocked",
    )
    block_striped = _exchange_oracle(
        values_host,
        group_threads=_BLOCK_THREADS,
        mode="blocked_to_striped",
    )
    warp_blocked = _exchange_oracle(
        values_host,
        group_threads=32,
        mode="striped_to_blocked",
    )
    warp_striped = _exchange_oracle(
        values_host,
        group_threads=32,
        mode="blocked_to_striped",
    )
    expected = (
        values_host,
        values_host,
        block_blocked,
        block_blocked,
        block_striped,
        block_striped,
        warp_blocked,
        warp_blocked,
        warp_striped,
        warp_striped,
    )
    for actual, reference in zip(segments, expected, strict=True):
        torch.testing.assert_close(actual, reference, atol=0, rtol=0)

    return {
        "block_threads": _BLOCK_THREADS,
        "items_per_thread": _ITEMS_PER_THREAD,
        "input_preserved": True,
        "portable_modes": ("striped_to_blocked", "blocked_to_striped"),
    }
