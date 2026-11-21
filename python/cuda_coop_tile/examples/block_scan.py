# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compute and check the exclusive prefix sums of two independent tiles."""

from __future__ import annotations

import cuda.tile as ct

import cuda.coop.tile as coop

TILE_ITEMS = 256


@ct.kernel
def block_exclusive_sum_kernel(input_array, output_array):
    tile_index = ct.bid(0)
    values = ct.load(input_array, index=(tile_index,), shape=(TILE_ITEMS,))
    prefix_sums = coop.exclusive_sum(values)
    ct.store(output_array, index=(tile_index,), tile=prefix_sums)


def launch_block_exclusive_sum(stream, grid, input_array, output_array):
    ct.launch(
        stream,
        grid,
        block_exclusive_sum_kernel,
        (input_array, output_array),
    )


def main():
    import torch

    input_rows = torch.arange(
        2 * TILE_ITEMS, device="cuda", dtype=torch.int32
    ).reshape(2, TILE_ITEMS)
    input_array = input_rows.flatten()
    output_array = torch.empty_like(input_array)
    launch_block_exclusive_sum(
        torch.cuda.current_stream(), (2, 1, 1), input_array, output_array
    )

    output_rows = output_array.reshape_as(input_rows)
    expected = input_rows.cumsum(dim=1, dtype=input_rows.dtype) - input_rows
    torch.testing.assert_close(output_rows, expected)
    print(output_rows[:, :4].tolist())


if __name__ == "__main__":
    main()
