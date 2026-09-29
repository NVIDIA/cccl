# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import numpy as np

import cuda.coop as common
import cuda.coop.numba_mlir as coop

common.register("numba")  # expected-error: [arg-type]

coop.TempStorage(64, 16)  # expected-error: [call-arg]
common.TempStorage(64, 16)  # expected-error: [call-arg]
values = coop.ThreadData(2, np.int32)
common_values = common.ThreadData(2, np.int32)
common_block = common.this_block()
common_block.rank()  # expected-error: [attr-defined]
common_block.count()  # expected-error: [attr-defined]
common_block.rank_as(np.uint32)  # expected-error: [attr-defined]
common_block.count_as(np.uint32)  # expected-error: [attr-defined]
common_block.sync()  # expected-error: [attr-defined]
common_block.sync_aligned()  # expected-error: [attr-defined]
common_block.is_member()  # expected-error: [attr-defined]
qualified_block = coop.this_block()
qualified_block.rank()  # expected-error: [attr-defined]
qualified_block.count()  # expected-error: [attr-defined]
qualified_block.rank_as(np.uint32)  # expected-error: [attr-defined]
qualified_block.count_as(np.uint32)  # expected-error: [attr-defined]
qualified_block.sync()  # expected-error: [attr-defined]
qualified_block.sync_aligned()  # expected-error: [attr-defined]
qualified_block.is_member()  # expected-error: [attr-defined]
common.load(  # expected-error: [call-overload]
    common.this_block(),
    object(),
    common_values,
    algorithm="stripd",
)
common.load(  # expected-error: [call-overload]
    common.this_warp(),
    object(),
    common_values,
    algorithm="warp_transpose",
)
common.load(
    common.this_warp(),  # expected-error: [arg-type]
    object(),
    common_values,
    temp_storage=common.TempStorage(),
)
common.load(
    common.this_warp().group_by(8),  # expected-error: [arg-type]
    object(),
    common_values,
    temp_storage=common.TempStorage(),
)
common.store(  # expected-error: [call-overload]
    common.this_warp(),
    object(),
    common_values,
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
# Test rejected attributes.
coop.BlockLoadAlgorithm  # expected-error: [attr-defined]  # noqa: B018
coop.BlockStoreAlgorithm  # expected-error: [attr-defined]  # noqa: B018
coop.WarpLoadAlgorithm  # expected-error: [attr-defined]  # noqa: B018
coop.WarpStoreAlgorithm  # expected-error: [attr-defined]  # noqa: B018
