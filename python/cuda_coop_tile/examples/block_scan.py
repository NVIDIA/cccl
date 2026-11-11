# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import cuda.tile as ct

import cuda.coop.tile as coop

TILE_ITEMS = 256


@ct.kernel
def block_exclusive_sum_kernel(input_array, output_array):
    tile_index = ct.bid(0)
    # Shaped cuda.tile loads and stores use tile-coordinate indices.
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
