# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Deliberately invalid calls proving the public stubs reject misuse."""

from __future__ import annotations

import numpy as np

import cuda.coop as portable

portable.TempStorage(64, 16)  # expected-error: [call-arg]
portable_values = portable.ThreadData(2, np.int32)
portable_block = portable.this_block()
portable_block.rank()  # expected-error: [attr-defined]
portable_block.count()  # expected-error: [attr-defined]
portable_block.rank_as(np.uint32)  # expected-error: [attr-defined]
portable_block.count_as(np.uint32)  # expected-error: [attr-defined]
portable_block.sync()  # expected-error: [attr-defined]
portable_block.sync_aligned()  # expected-error: [attr-defined]
portable_block.is_member()  # expected-error: [attr-defined]
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

portable.register("numba")  # expected-error: [arg-type]
