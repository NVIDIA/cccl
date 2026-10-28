# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Reject unsupported shuffle calls through stubs."""

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


coop.shuffle(warp, values)  # expected-error: [arg-type]
coop.shuffle(  # expected-error: [call-overload]
    block, values, distance=2
)
coop.shuffle(block, scalar)  # expected-error: [call-overload]
coop.shuffle(  # expected-error: [call-overload]
    block, values, mode="rotate"
)
coop.shuffle(  # expected-error: [call-overload]
    block, scalar, mode="rotate", distance=np.uint64(1)
)
coop.shuffle(  # expected-error: [call-overload]
    block, values, prefix=scalar
)
coop.shuffle(  # expected-error: [call-overload]
    block, values, suffix=scalar
)
coop.shuffle(  # expected-error: [call-overload]
    block, values, temp_storage=coop.TempStorage()
)
