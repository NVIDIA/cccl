# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Sort key-value pairs and select a valid prefix with CUTLASS."""

from __future__ import annotations

import cutlass
import numpy as np
import torch
from cutlass import cute
from cutlass.base_dsl.typing import Int32
from cutlass.cute.runtime import from_dlpack

import cuda.coop.cutlass as coop

THREADS = 32
ITEMS_PER_THREAD = 2
TILE_ITEMS = THREADS * ITEMS_PER_THREAD
TOPK = 7
VALID_ITEMS = TILE_ITEMS - 9


@cute.kernel
def ordering_kernel(
    keys_in: cute.Tensor,
    values_in: cute.Tensor,
    sorted_keys_out: cute.Tensor,
    sorted_values_out: cute.Tensor,
    top_keys_out: cute.Tensor,
    top_values_out: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    offset = tidx * Int32(ITEMS_PER_THREAD)
    keys = coop.ThreadData(ITEMS_PER_THREAD, dtype=Int32)
    values = coop.ThreadData(ITEMS_PER_THREAD, dtype=Int32)
    keys[0] = keys_in[offset]
    keys[1] = keys_in[offset + Int32(1)]
    values[0] = values_in[offset]
    values[1] = values_in[offset + Int32(1)]

    # docs: start cutlass-qualified-ordering
    sorted_keys, sorted_values = coop.radix_sort_pairs(
        coop.this_block(),
        keys,
        values,
    )
    scratch = coop.TempStorage(16_384, alignment=16)
    top_keys, top_values = coop.topk_min_pairs(
        coop.this_block(),
        keys,
        values,
        TOPK,
        valid_items=VALID_ITEMS,
        temp_storage=scratch,
    )
    # docs: end cutlass-qualified-ordering

    sorted_keys_out[offset] = sorted_keys[0]
    sorted_keys_out[offset + Int32(1)] = sorted_keys[1]
    sorted_values_out[offset] = sorted_values[0]
    sorted_values_out[offset + Int32(1)] = sorted_values[1]
    if offset < TOPK:
        top_keys_out[offset] = top_keys[0]
        top_values_out[offset] = top_values[0]
    if offset + Int32(1) < TOPK:
        top_keys_out[offset + Int32(1)] = top_keys[1]
        top_values_out[offset + Int32(1)] = top_values[1]


@cute.jit
def launch(
    keys_in: cute.Tensor,
    values_in: cute.Tensor,
    sorted_keys_out: cute.Tensor,
    sorted_values_out: cute.Tensor,
    top_keys_out: cute.Tensor,
    top_values_out: cute.Tensor,
):
    ordering_kernel(
        keys_in,
        values_in,
        sorted_keys_out,
        sorted_values_out,
        top_keys_out,
        top_values_out,
    ).launch(grid=(1, 1, 1), block=(THREADS, 1, 1))


def run_example() -> tuple[torch.Tensor, torch.Tensor]:
    """Run the ordering operations and return the selected pairs."""

    cutlass.cuda.initialize_cuda_context()
    indices = np.arange(TILE_ITEMS, dtype=np.int32)
    keys_host = ((indices * 29 + 17) % 67).astype(np.int32)
    values_host = indices * np.int32(11) + np.int32(3)
    keys = torch.from_numpy(keys_host).cuda()
    values = torch.from_numpy(values_host).cuda()
    sorted_keys = torch.zeros_like(keys)
    sorted_values = torch.zeros_like(values)
    top_keys = torch.zeros(TOPK, dtype=torch.int32, device="cuda")
    top_values = torch.zeros(TOPK, dtype=torch.int32, device="cuda")

    launch(
        from_dlpack(keys),
        from_dlpack(values),
        from_dlpack(sorted_keys),
        from_dlpack(sorted_values),
        from_dlpack(top_keys),
        from_dlpack(top_values),
    )
    torch.cuda.synchronize()

    order = np.argsort(keys_host, kind="stable")
    np.testing.assert_array_equal(sorted_keys.cpu().numpy(), keys_host[order])
    np.testing.assert_array_equal(sorted_values.cpu().numpy(), values_host[order])
    expected_top = sorted(
        zip(keys_host[:VALID_ITEMS], values_host[:VALID_ITEMS], strict=True)
    )[:TOPK]
    observed_top = sorted(
        zip(top_keys.cpu().tolist(), top_values.cpu().tolist(), strict=True)
    )
    assert observed_top == expected_top
    return top_keys.cpu(), top_values.cpu()


def main() -> int:
    keys, values = run_example()
    print({"keys": keys, "values": values})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
