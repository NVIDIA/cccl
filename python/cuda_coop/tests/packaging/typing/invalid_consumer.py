# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Deliberately invalid calls proving the public stubs reject misuse."""

from __future__ import annotations

import numpy as np

import cuda.coop as portable
import cuda.coop.numba_mlir as coop

values = coop.ThreadData(2, np.int32)
portable_values = portable.ThreadData(2, np.int32)
portable.load(  # expected-error: [call-overload]
    portable.this_block(),
    object(),
    portable_values,
    algorithm="stripd",
)
coop.load(
    coop.this_warp(),  # expected-error: [arg-type]
    object(),
    values,
)
coop.load(
    coop.this_block(),
    object(),
    values,
    algorithm="stripd",  # expected-error: [arg-type]
)
coop.load(
    coop.this_block(),
    object(),
    values,
    valid_items=1.5,  # expected-error: [arg-type]
)
coop.store(
    coop.this_block(),
    object(),
    values,
    offset="1",  # expected-error: [arg-type]
)
bad_algorithm: coop.BlockLoadAlgorithm = 0  # expected-error: [assignment]
