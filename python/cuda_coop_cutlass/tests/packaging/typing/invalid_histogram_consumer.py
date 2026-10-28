# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Reject unsupported histogram calls through stubs."""

from __future__ import annotations

import numpy as np
from cutlass import Int32

import cuda.coop.cutlass as coop

block = coop.this_block()
warp = coop.this_warp()
logical = warp.group_by(8)
mapped = block.group_by(2)
scalar = Int32(1)
values = coop.ThreadData(items_per_thread=2, dtype=np.int32)


coop.histogram(warp, values, bins=32)  # expected-error: [arg-type]
coop.histogram(  # expected-error: [call-overload]
    block, scalar, bins=32
)
coop.histogram(
    block,
    values,
    bins=32,
    counter_dtype=np.float32,  # expected-error: [arg-type]
)
coop.histogram(  # expected-error: [call-overload]
    block, values, bins=32, algorithm="other"
)
