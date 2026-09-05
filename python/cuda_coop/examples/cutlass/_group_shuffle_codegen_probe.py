# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Private runtime and code-generation probe for common block Shuffle."""

from __future__ import annotations

import functools
from typing import Any

from examples.cutlass._runtime import require_runtime

_BLOCK_DIM = (8, 4, 2)
_BLOCK_THREADS = 64
_ITEMS_PER_THREAD = 4
_TILE_ITEMS = _BLOCK_THREADS * _ITEMS_PER_THREAD
_SEGMENT_COUNT = 6


def _store_items(output, segment: int, thread_rank, items) -> None:
    offset = segment * _TILE_ITEMS + thread_rank * _ITEMS_PER_THREAD
    output[offset] = items[0]
    output[offset + 1] = items[1]
    output[offset + 2] = items[2]
    output[offset + 3] = items[3]


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
        qualified_items[0] = values[input_offset]
        qualified_items[1] = values[input_offset + 1]
        qualified_items[2] = values[input_offset + 2]
        qualified_items[3] = values[input_offset + 3]

        common_group = common_coop.this_block()
        qualified_group = cutlass_coop.this_block()
        common_up = common_coop.shuffle(common_group, common_items, mode="up")
        qualified_up = cutlass_coop.shuffle(
            qualified_group,
            qualified_items,
            mode="up",
            distance=1,
        )
        common_down = common_coop.shuffle(
            common_group,
            common_items,
            mode="down",
            distance=1,
        )
        qualified_down = cutlass_coop.shuffle(
            qualified_group,
            qualified_items,
            mode="down",
        )

        # Observe both sources only after every transforming operation. This
        # directly qualifies the common V1 non-mutation rule.
        _store_items(output, 0, tidx, common_items)
        _store_items(output, 1, tidx, qualified_items)
        _store_items(output, 2, tidx, common_up)
        _store_items(output, 3, tidx, qualified_up)
        _store_items(output, 4, tidx, common_down)
        _store_items(output, 5, tidx, qualified_down)

    @cute.jit
    def _run(values: cute.Tensor, output: cute.Tensor):
        _kernel(values, output).launch(grid=(1, 1, 1), block=_BLOCK_DIM)

    return _run, torch, from_dlpack, cutlass


def _assert_result(torch, values_host, output) -> None:
    segments = output.cpu().reshape(_SEGMENT_COUNT, _TILE_ITEMS)
    torch.testing.assert_close(segments[0], values_host, atol=0, rtol=0)
    torch.testing.assert_close(segments[1], values_host, atol=0, rtol=0)

    # CUB deliberately leaves the vacated edge undefined. Qualify only the
    # positions for which the common contract promises a value.
    torch.testing.assert_close(segments[2, 1:], values_host[:-1], atol=0, rtol=0)
    torch.testing.assert_close(segments[3, 1:], values_host[:-1], atol=0, rtol=0)
    torch.testing.assert_close(segments[4, :-1], values_host[1:], atol=0, rtol=0)
    torch.testing.assert_close(segments[5, :-1], values_host[1:], atol=0, rtol=0)


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

    # Launch the same compiled runner twice to cover repeat execution without
    # creating an additional provider specialization.
    for _ in range(2):
        output.zero_()
        run(from_dlpack(values), from_dlpack(output))
        torch.cuda.synchronize()
        _assert_result(torch, values_host, output)

    return {
        "block_dim": _BLOCK_DIM,
        "items_per_thread": _ITEMS_PER_THREAD,
        "input_preserved": True,
        "portable_modes": ("up", "down"),
        "repeat_launches": 2,
        "vacated_edges_defined": False,
    }
