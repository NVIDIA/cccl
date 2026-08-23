# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Private compile and final-cubin probe for common pair collectives."""

from __future__ import annotations

import functools
from typing import Any

import numpy as np

from examples.cutlass._runtime import require_runtime

BLOCK_DIM = (64, 1, 1)
BLOCK_THREADS = 64
ITEMS_PER_THREAD = 2
TOTAL_ITEMS = BLOCK_THREADS * ITEMS_PER_THREAD
RESULT_SEGMENTS = 16
BLOCK_VALID_ITEMS = TOTAL_ITEMS - 11
WARP_VALID_ITEMS = 53
TOPK_K = 11


def _store(output, segment: int, rank, values) -> None:
    offset = segment * TOTAL_ITEMS + rank * ITEMS_PER_THREAD
    output[offset] = values[0]
    output[offset + 1] = values[1]


@functools.lru_cache(maxsize=1)
def make_runner() -> tuple[Any, Any, Any, Any, Any]:
    cutlass, cute, torch, from_dlpack, qualified, Int32 = require_runtime(
        include_int32=True
    )
    from cutlass.base_dsl.typing import Int64

    from cuda import coop as common

    globals().update(
        cutlass=cutlass,
        cute=cute,
        common=common,
        qualified=qualified,
        Int32=Int32,
        Int64=Int64,
    )

    @cute.kernel
    def _kernel(
        keys_in: cute.Tensor,
        values_in: cute.Tensor,
        key_output: cute.Tensor,
        value_output: cute.Tensor,
    ):
        tidx, tidy, tidz = cute.arch.thread_idx()
        rank = tidx + Int32(64) * (tidy + tidz)
        offset = rank * Int32(ITEMS_PER_THREAD)
        common_keys = common.ThreadData(ITEMS_PER_THREAD, dtype=int)
        common_values = common.ThreadData(ITEMS_PER_THREAD, dtype=np.int64)
        qualified_keys = qualified.ThreadData(ITEMS_PER_THREAD, dtype=Int32)
        qualified_values = qualified.ThreadData(ITEMS_PER_THREAD, dtype=Int64)
        for index in cutlass.range_constexpr(ITEMS_PER_THREAD):
            common_keys[index] = keys_in[offset + index]
            common_values[index] = values_in[offset + index]
            qualified_keys[index] = keys_in[offset + index]
            qualified_values[index] = values_in[offset + index]

        common_block = common.this_block()
        qualified_block = qualified.this_block()
        common_storage = common.TempStorage()
        qualified_storage = qualified.TempStorage()
        results = (
            common.merge_sort_pairs(
                common_block,
                common_keys,
                common_values,
                temp_storage=common_storage,
            ),
            qualified.merge_sort_pairs(
                qualified_block,
                qualified_keys,
                qualified_values,
                temp_storage=qualified_storage,
            ),
            common.merge_sort_pairs(
                common_block,
                common_keys,
                common_values,
                descending=True,
                valid_items=BLOCK_VALID_ITEMS,
                oob_default=-2_147_483_648,
                temp_storage=common_storage,
            ),
            qualified.merge_sort_pairs(
                qualified_block,
                qualified_keys,
                qualified_values,
                descending=True,
                valid_items=Int32(BLOCK_VALID_ITEMS),
                oob_default=Int32(-2_147_483_648),
                temp_storage=qualified_storage,
            ),
            common.merge_sort_pairs(
                common.this_warp(),
                common_keys,
                common_values,
                descending=True,
            ),
            qualified.merge_sort_pairs(
                qualified.this_warp(),
                qualified_keys,
                qualified_values,
                descending=True,
            ),
            common.merge_sort_pairs(
                common.this_warp(),
                common_keys,
                common_values,
                valid_items=WARP_VALID_ITEMS,
                oob_default=2_147_483_647,
            ),
            qualified.merge_sort_pairs(
                qualified.this_warp(),
                qualified_keys,
                qualified_values,
                valid_items=Int32(WARP_VALID_ITEMS),
                oob_default=Int32(2_147_483_647),
            ),
            common.radix_sort_pairs(
                common_block,
                common_keys,
                common_values,
                temp_storage=common_storage,
            ),
            qualified.radix_sort_pairs(
                qualified_block,
                qualified_keys,
                qualified_values,
                temp_storage=qualified_storage,
            ),
            common.radix_sort_pairs(
                common_block,
                common_keys,
                common_values,
                begin_bit=4,
                end_bit=16,
                descending=True,
            ),
            qualified.radix_sort_pairs(
                qualified_block,
                qualified_keys,
                qualified_values,
                begin_bit=4,
                end_bit=16,
                descending=True,
            ),
            common.topk_max_pairs(
                common_block,
                common_keys,
                common_values,
                TOPK_K,
            ),
            qualified.topk_max_pairs(
                qualified_block,
                qualified_keys,
                qualified_values,
                TOPK_K,
            ),
            common.topk_min_pairs(
                common_block,
                common_keys,
                common_values,
                TOPK_K,
                valid_items=BLOCK_VALID_ITEMS,
                begin_bit=0,
                end_bit=16,
            ),
            qualified.topk_min_pairs(
                qualified_block,
                qualified_keys,
                qualified_values,
                TOPK_K,
                valid_items=BLOCK_VALID_ITEMS,
                begin_bit=0,
                end_bit=16,
            ),
        )
        for segment in cutlass.range_constexpr(RESULT_SEGMENTS):
            result_keys, result_values = results[segment]
            _store(key_output, segment, rank, result_keys)
            _store(value_output, segment, rank, result_values)

    @cute.jit
    def _run(
        keys_in: cute.Tensor,
        values_in: cute.Tensor,
        key_output: cute.Tensor,
        value_output: cute.Tensor,
    ):
        _kernel(keys_in, values_in, key_output, value_output).launch(
            grid=(1, 1, 1), block=BLOCK_DIM
        )

    return _run, torch, from_dlpack, cutlass, Int64


def run_example() -> dict[str, Any]:
    run, torch, from_dlpack, cutlass, _ = make_runner()
    cutlass.cuda.initialize_cuda_context()
    keys = torch.arange(TOTAL_ITEMS, dtype=torch.int32, device="cuda")
    values = torch.arange(TOTAL_ITEMS, dtype=torch.int64, device="cuda")
    key_output = torch.empty(
        RESULT_SEGMENTS * TOTAL_ITEMS, dtype=torch.int32, device="cuda"
    )
    value_output = torch.empty(
        RESULT_SEGMENTS * TOTAL_ITEMS, dtype=torch.int64, device="cuda"
    )
    run(
        from_dlpack(keys),
        from_dlpack(values),
        from_dlpack(key_output),
        from_dlpack(value_output),
    )
    torch.cuda.synchronize()
    return {"result_segments": RESULT_SEGMENTS, "value_dtype": "int64"}


if __name__ == "__main__":
    print(run_example())
