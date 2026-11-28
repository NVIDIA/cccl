# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Reject invalid hierarchy, payload, and storage calls through public stubs."""

import numpy as np
from cutlass import Float32

import cuda.coop.cutlass as coop

block = coop.this_block()
logical = coop.this_warp().group_by(8)
mapped = block.group_by(2)
block.rank_as(Float32)  # expected-error: [arg-type]
block.count_as(np.bool_)  # expected-error: [arg-type]
block.rank_as(bool)  # expected-error: [arg-type]
logical.count("block")  # expected-error: [call-overload]
mapped.rank("grid")  # expected-error: [call-overload]
mapped.sync()  # expected-error: [misc]
coop.this_grid().sync_aligned()  # expected-error: [misc]
coop.TempStorage(sharing="invalid")  # expected-error: [arg-type]
coop.ThreadData(items_per_thread="two")  # expected-error: [call-overload]
coop.ThreadData.from_values(first=1)  # expected-error: [call-arg]
selector = np.dtype(np.uint8)
existing = coop.ThreadData(1, dtype=selector, values=[1])
adapted = coop.ThreadData.from_payload(existing, dtype=selector)
adapted[0].item()  # expected-error: [union-attr]
coop.ThreadData.from_payload(  # expected-error: [call-overload]
    existing, dtype=np.uint8(1)
)
