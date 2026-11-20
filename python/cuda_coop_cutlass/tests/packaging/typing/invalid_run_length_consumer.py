# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Reject unsupported run length calls through stubs."""

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


coop.run_length_decode(
    warp,  # expected-error: [arg-type]
    values,
    values,
    decoded_items_per_thread=2,
)
coop.run_length_decode(  # expected-error: [call-overload]
    block, values, values
)
coop.run_length_decode(  # expected-error: [call-overload]
    block, scalar, values, decoded_items_per_thread=2
)
