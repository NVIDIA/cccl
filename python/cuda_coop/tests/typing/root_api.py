# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np

from cuda import coop


def block_reduce(value: np.int32, valid_items: np.int32) -> np.int32:
    block = coop.this_block()
    return coop.reduce(
        block,
        value,
        binary_op="maximum",
        valid_items=valid_items,
        algorithm="raking",
    )


def block_sum(value: np.float32) -> np.float32:
    return coop.sum(
        coop.this_block(),
        value,
        algorithm="warp_reductions",
    )


def narrow_integer_sums(
    signed8: np.int8,
    signed16: np.int16,
    unsigned16: np.uint16,
) -> tuple[np.int8, np.int16, np.uint16]:
    block = coop.this_block()
    return (
        coop.sum(block, signed8),
        coop.sum(block, signed16),
        coop.sum(block, unsigned16),
    )


opaque_group = coop.ThreadGroup()  # type: ignore[call-arg]
