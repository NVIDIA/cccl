# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Private runtime and code-generation probe for common comparison collectives."""

from __future__ import annotations

import functools
from typing import Any

from examples.cutlass._runtime import require_runtime

_BLOCK_DIM = (8, 4, 2)
_BLOCK_THREADS = 64
_ITEMS_PER_THREAD = 2
_TILE_ITEMS = _BLOCK_THREADS * _ITEMS_PER_THREAD
_VALID_ITEMS = _TILE_ITEMS - 3
_PREDECESSOR = -7
_SUCCESSOR = 211
_SEGMENT_COUNT = 10


def _store_items(output, segment: int, thread_rank, items) -> None:
    offset = segment * _TILE_ITEMS + thread_rank * _ITEMS_PER_THREAD
    output[offset] = items[0]
    output[offset + 1] = items[1]


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
        thread_rank = tidx + Int32(8) * (tidy + Int32(4) * tidz)
        input_offset = thread_rank * _ITEMS_PER_THREAD

        common_items = common_coop.ThreadData(_ITEMS_PER_THREAD, dtype=Int32)
        qualified_items = cutlass_coop.ThreadData(_ITEMS_PER_THREAD, dtype=Int32)
        common_items[0] = values[input_offset]
        common_items[1] = values[input_offset + 1]
        qualified_items[0] = values[input_offset]
        qualified_items[1] = values[input_offset + 1]

        common_group = common_coop.this_block()
        qualified_group = cutlass_coop.this_block()
        common_storage = common_coop.TempStorage()
        qualified_storage = cutlass_coop.TempStorage()

        common_left = common_coop.adjacent_difference(
            common_group,
            common_items,
            valid_items=_VALID_ITEMS,
            tile_predecessor_item=Int32(_PREDECESSOR),
            temp_storage=common_storage,
        )
        qualified_left = cutlass_coop.adjacent_difference(
            qualified_group,
            qualified_items,
            valid_items=_VALID_ITEMS,
            tile_predecessor_item=Int32(_PREDECESSOR),
            temp_storage=qualified_storage,
        )
        common_right = common_coop.adjacent_difference(
            common_group,
            common_items,
            direction="right",
            tile_successor_item=Int32(_SUCCESSOR),
            temp_storage=common_storage,
        )
        qualified_right = cutlass_coop.adjacent_difference(
            qualified_group,
            qualified_items,
            direction="right",
            tile_successor_item=Int32(_SUCCESSOR),
            temp_storage=qualified_storage,
        )
        common_heads = common_coop.discontinuity(
            common_group,
            common_items,
            tile_predecessor_item=Int32(_PREDECESSOR),
            temp_storage=common_storage,
        )
        qualified_heads = cutlass_coop.discontinuity(
            qualified_group,
            qualified_items,
            tile_predecessor_item=Int32(_PREDECESSOR),
            temp_storage=qualified_storage,
        )
        common_tails = common_coop.discontinuity(
            common_group,
            common_items,
            mode="tails",
            tile_successor_item=Int32(_SUCCESSOR),
            temp_storage=common_storage,
        )
        qualified_tails = cutlass_coop.discontinuity(
            qualified_group,
            qualified_items,
            mode="tails",
            tile_successor_item=Int32(_SUCCESSOR),
            temp_storage=qualified_storage,
        )

        # Observe both inputs only after all transforming calls. This directly
        # qualifies the common V1 non-mutation rule.
        _store_items(output, 0, thread_rank, common_items)
        _store_items(output, 1, thread_rank, qualified_items)
        _store_items(output, 2, thread_rank, common_left)
        _store_items(output, 3, thread_rank, qualified_left)
        _store_items(output, 4, thread_rank, common_right)
        _store_items(output, 5, thread_rank, qualified_right)
        _store_items(output, 6, thread_rank, common_heads)
        _store_items(output, 7, thread_rank, qualified_heads)
        _store_items(output, 8, thread_rank, common_tails)
        _store_items(output, 9, thread_rank, qualified_tails)

    @cute.jit
    def _run(values: cute.Tensor, output: cute.Tensor):
        _kernel(values, output).launch(grid=(1, 1, 1), block=_BLOCK_DIM)

    return _run, torch, from_dlpack, cutlass


def run_example() -> dict[str, Any]:
    run, torch, from_dlpack, cutlass = make_runner()
    cutlass.cuda.initialize_cuda_context()

    values_host = torch.tensor(
        [((index // 3) + 2 * (index % 11 == 0)) for index in range(_TILE_ITEMS)],
        dtype=torch.int32,
    )
    values = values_host.cuda()
    output = torch.full(
        (_SEGMENT_COUNT * _TILE_ITEMS,),
        -999,
        dtype=torch.int32,
        device="cuda",
    )
    run(from_dlpack(values), from_dlpack(output))
    torch.cuda.synchronize()

    segments = output.cpu().reshape(_SEGMENT_COUNT, _TILE_ITEMS)
    left = values_host.clone()
    left[0] = values_host[0] - _PREDECESSOR
    left[1:_VALID_ITEMS] = values_host[1:_VALID_ITEMS] - values_host[: _VALID_ITEMS - 1]
    right = torch.empty_like(values_host)
    right[:-1] = values_host[:-1] - values_host[1:]
    right[-1] = values_host[-1] - _SUCCESSOR
    heads = torch.empty_like(values_host)
    heads[0] = int(values_host[0] != _PREDECESSOR)
    heads[1:] = (values_host[1:] != values_host[:-1]).to(torch.int32)
    tails = torch.empty_like(values_host)
    tails[:-1] = (values_host[:-1] != values_host[1:]).to(torch.int32)
    tails[-1] = int(values_host[-1] != _SUCCESSOR)

    torch.testing.assert_close(segments[0], values_host, atol=0, rtol=0)
    torch.testing.assert_close(segments[1], values_host, atol=0, rtol=0)
    for actual in (segments[2], segments[3]):
        torch.testing.assert_close(actual, left, atol=0, rtol=0)
    for actual, expected in (
        (segments[4], right),
        (segments[5], right),
        (segments[6], heads),
        (segments[7], heads),
        (segments[8], tails),
        (segments[9], tails),
    ):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    return {
        "block_threads": _BLOCK_THREADS,
        "items_per_thread": _ITEMS_PER_THREAD,
        "valid_items": _VALID_ITEMS,
        "input_preserved": True,
        "flag_dtype": "int32",
    }
