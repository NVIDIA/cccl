# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Deliberately invalid calls proving the public stubs reject misuse."""

from __future__ import annotations

from typing import cast

import numpy as np

import cuda.coop as portable
import cuda.coop.numba_mlir as coop

values = coop.ThreadData(2, np.int32)
coop.ThreadData(2, np.int32, alignment=8)  # expected-error: [call-overload]
portable_values = portable.ThreadData(2, np.int32)
portable_block = portable.this_block()
portable_block.rank_as(np.float32)  # expected-error: [arg-type]
portable_block.count_as(np.bool_)  # expected-error: [arg-type]
portable_block.rank_as(bool)  # expected-error: [arg-type]
portable.this_grid().sync()  # expected-error: [misc]
portable_block.group_by(2).sync()  # expected-error: [misc]
portable_block.group_by(2).rank("grid")  # expected-error: [call-overload]
portable.this_warp().group_by(8).count("block")  # expected-error: [call-overload]
qualified_block = coop.this_block()
qualified_block.rank_as(np.float32)  # expected-error: [arg-type]
qualified_block.count_as(np.bool_)  # expected-error: [arg-type]
qualified_block.count_as(bool)  # expected-error: [arg-type]
coop.this_grid().sync_aligned()  # expected-error: [misc]
qualified_block.group_by(2).sync_aligned()  # expected-error: [misc]
qualified_block.group_by(2).count("grid")  # expected-error: [call-overload]
coop.this_warp().group_by(8).rank("cluster")  # expected-error: [call-overload]
portable.load(  # expected-error: [call-overload]
    portable.this_block(),
    object(),
    portable_values,
    algorithm="stripd",
)
portable.load(  # expected-error: [call-overload]
    portable.this_warp(),
    object(),
    portable_values,
    algorithm="warp_transpose",
)
portable.load(
    portable.this_warp(),  # expected-error: [arg-type]
    object(),
    portable_values,
    temp_storage=portable.TempStorage(),
)
portable.load(
    portable.this_warp().group_by(8),  # expected-error: [arg-type]
    object(),
    portable_values,
    temp_storage=portable.TempStorage(),
)
portable.store(  # expected-error: [call-overload]
    portable.this_warp(),
    object(),
    portable_values,
    algorithm="warp_transpose",
)
coop.load(  # expected-error: [call-overload]
    coop.this_warp(),
    object(),
    values,
    algorithm="warp_transpose",
)
coop.load(
    coop.this_warp(),  # expected-error: [arg-type]
    object(),
    values,
    temp_storage=coop.TempStorage(),
)
coop.load(
    coop.this_warp().group_by(8),  # expected-error: [arg-type]
    object(),
    values,
    temp_storage=coop.TempStorage(),
)
coop.load(  # expected-error: [call-overload]
    coop.this_warp(),
    object(),
    values,
    algorithm=0,
)
coop.store(  # expected-error: [call-overload]
    coop.this_warp(),
    object(),
    values,
    algorithm=True,
)
coop.load(  # expected-error: [call-overload]
    coop.this_block(),
    object(),
    values,
    algorithm=0,
)
coop.store(  # expected-error: [call-overload]
    coop.this_block(),
    object(),
    values,
    algorithm=True,
)
coop.load(  # expected-error: [call-overload]
    coop.this_block(),
    object(),
    values,
    algorithm="stripd",
)
coop.load(  # expected-error: [call-overload]
    coop.this_block(),
    object(),
    values,
    valid_items=1.5,
)
coop.load(  # expected-error: [call-overload]
    coop.this_block(),
    object(),
    values,
    oob_default=0,
)
coop.store(  # expected-error: [call-overload]
    coop.this_block(),
    object(),
    values,
    offset="1",
)
coop.BlockLoadAlgorithm  # expected-error: [attr-defined]
coop.BlockStoreAlgorithm  # expected-error: [attr-defined]
coop.WarpLoadAlgorithm  # expected-error: [attr-defined]
coop.WarpStoreAlgorithm  # expected-error: [attr-defined]
portable.exchange(
    portable.this_block(),
    portable_values,
    mode="scatter_to_striped",  # expected-error: [arg-type]
)
portable.shuffle(
    portable.this_block(),
    portable_values,
    distance=2,  # expected-error: [arg-type]
)
coop.exchange(  # expected-error: [call-overload]
    coop.this_block(),
    values,
    mode="scatter_to_blocked",
)
coop.exchange(  # expected-error: [call-overload]
    coop.this_warp(),
    values,
    mode="warp_striped_to_blocked",
)
coop.exchange(  # expected-error: [call-overload]
    coop.this_warp(),
    values,
    mode="scatter_to_striped",
    ranks=coop.ThreadData(2, np.int32),
)
coop.shuffle(  # expected-error: [call-overload]
    coop.this_block(),
    values,
    mode="offset",
)
coop.shuffle(  # expected-error: [call-overload]
    coop.this_block(),
    np.int32(1),
    mode="up",
)
floating_ranks = coop.ThreadData(2, np.float32)
floating_flags = coop.ThreadData(2, np.float32)
coop.exchange(  # expected-error: [call-overload]
    coop.this_block(),
    values,
    mode="scatter_to_blocked",
    ranks=floating_ranks,
)
coop.exchange(  # expected-error: [call-overload]
    coop.this_block(),
    values,
    mode="scatter_to_striped_flagged",
    ranks=np.int32(0),
    valid_flags=floating_flags,
)


def select_left(left: np.int32, right: np.int32) -> np.int32:
    del right
    return left


portable.reduce(  # expected-error: [call-overload]
    portable_block,
    np.int32(1),
    binary_op=select_left,
    broadcast=False,
)
portable.reduce(  # expected-error: [call-overload]
    portable_block,
    np.int32(1),
    binary_op=0,
)
portable.sum(  # expected-error: [call-overload]
    portable_block,
    np.int32(1),
    broadcast=False,
    algorithm=0,
)
portable.sum(  # expected-error: [call-overload]
    portable_block,
    np.complex64(1),
)
coop.sum(  # expected-error: [call-overload]
    qualified_block,
    np.bool_(True),
)
coop.reduce(  # expected-error: [call-overload]
    qualified_block,
    np.complex64(1),
    binary_op="sum",
)
coop.reduce(  # expected-error: [call-overload]
    qualified_block,
    np.int32(1),
    binary_op=0,
)
coop.sum(  # expected-error: [call-overload]
    qualified_block,
    np.int32(1),
    broadcast=False,
    algorithm=0,
)
complex_values = cast(portable.ThreadDataLike[np.complex64], object())
coop.sum(  # expected-error: [type-var]
    qualified_block,
    complex_values,
)
coop.sum(  # expected-error: [call-overload]
    qualified_block,
    values,
    broadcast=False,
    valid_items=2,
)
coop.sum(  # expected-error: [call-overload]
    coop.this_warp(),
    np.int32(1),
    broadcast=False,
    algorithm="raking",
)
coop.reduce(
    qualified_block,
    np.int32(1),
    binary_op=select_left,  # expected-error: [arg-type]
    broadcast=True,
)
coop.reduce(  # expected-error: [call-overload]
    qualified_block,
    np.int32(1),
    binary_op=select_left,
    broadcast=False,
    algorithm="raking_commutative_only",
)
portable.scan(  # expected-error: [call-overload]
    portable_block,
    np.int32(1),
    valid_items=1,
)
portable.inclusive_sum(  # expected-error: [call-overload]
    portable_block,
    np.int32(1),
    aggregate_output=portable.ThreadData(1, np.int32),
)
portable.inclusive_scan(  # expected-error: [call-overload]
    portable_block,
    np.int32(1),
    scan_op=select_left,
)
portable.exclusive_scan(  # expected-error: [call-overload]
    portable_block,
    np.int32(1),
    scan_op="max",
)
portable.scan(  # expected-error: [call-overload]
    portable_block,
    np.int32(1),
    mode="inclusive",
    initial_value=np.int32(0),
)
coop.exclusive_scan(  # expected-error: [call-overload]
    qualified_block,
    np.int32(1),
    scan_op="max",
)
coop.scan(  # expected-error: [call-overload]
    qualified_block,
    np.int32(1),
    mode="inclusive",
    initial_value=np.int32(0),
)
coop.inclusive_sum(
    coop.this_warp(),  # expected-error: [arg-type]
    values,
)
coop.inclusive_sum(  # expected-error: [call-overload]
    coop.this_warp(),
    np.int32(1),
    algorithm="raking",
)
coop.inclusive_sum(
    qualified_block,  # expected-error: [arg-type]
    np.int32(1),
    valid_items=1,
)
coop.inclusive_sum(  # expected-error: [call-overload]
    qualified_block,
    np.int32(1),
    aggregate_output=np.int32(0),
)
coop.inclusive_sum(  # expected-error: [call-overload]
    qualified_block,
    np.bool_(True),
)
coop.inclusive_scan(  # expected-error: [call-overload]
    qualified_block,
    np.complex64(1),
    scan_op="max",
)
coop.scan(  # expected-error: [call-overload]
    qualified_block,
    np.int32(1),
    prefix_op=select_left,
)
