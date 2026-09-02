# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np

import cuda.coop.numba_mlir as coop


def block_reduce(value: np.int32, valid_items: np.int64) -> np.int32:
    return coop.reduce(
        coop.this_block(),
        value,
        binary_op="maximum",
        valid_items=valid_items,
        algorithm="raking",
    )


def block_sum(value: np.float64) -> np.float64:
    return coop.sum(
        coop.this_block(),
        value,
        algorithm="warp_reductions",
    )


def warp_reduce(value: np.int32, valid_items: np.int64) -> np.int32:
    return coop.reduce(
        coop.this_warp(),
        value,
        binary_op="maximum",
        valid_items=valid_items,
    )


def warp_sum(value: np.float64) -> np.float64:
    return coop.sum(coop.this_warp(), value)
