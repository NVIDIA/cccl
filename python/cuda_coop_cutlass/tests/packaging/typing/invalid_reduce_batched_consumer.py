# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Reject unsupported reduce batched calls through stubs."""

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


coop.reduce_batched(block, values)  # expected-error: [arg-type]
coop.reduce_batched(warp, scalar)  # expected-error: [call-overload]
coop.reduce_batched(  # expected-error: [call-overload]
    warp, values, output_layout="other"
)
