# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Reject unsupported exchange calls through stubs."""

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

ranks = coop.ThreadData(items_per_thread=2, dtype=np.int32)
flags = coop.ThreadData(items_per_thread=2, dtype=np.uint8)


coop.exchange(block, scalar)  # expected-error: [call-overload]
coop.exchange(  # expected-error: [call-overload]
    block, values, mode="scatter_to_blocked"
)
coop.exchange(  # expected-error: [call-overload]
    block, values, ranks=ranks
)
coop.exchange(  # expected-error: [call-overload]
    warp, values, mode="scatter_to_blocked", ranks=ranks
)
coop.exchange(  # expected-error: [call-overload]
    logical, values, mode="blocked_to_warp_striped"
)
coop.exchange(
    warp,  # expected-error: [arg-type]
    values,
    warp_time_slicing=True,
)
coop.exchange(  # expected-error: [call-overload]
    block,
    values,
    mode="scatter_to_striped_guarded",
    ranks=ranks,
    warp_time_slicing=True,
)
coop.exchange(  # expected-error: [call-overload]
    block, values, mode="scatter_to_striped_flagged", ranks=ranks
)
coop.exchange(  # expected-error: [call-overload]
    block, values, mode="scatter_to_blocked", ranks=flags
)
coop.exchange(  # expected-error: [call-overload]
    block,
    values,
    mode="scatter_to_striped_flagged",
    ranks=ranks,
    valid_flags=coop.ThreadData(items_per_thread=2, dtype=np.bool_),
)
coop.exchange(  # expected-error: [call-overload]
    block, values, temp_storage=coop.TempStorage()
)
