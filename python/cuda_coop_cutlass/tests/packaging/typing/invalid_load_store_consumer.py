# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Reject unsupported load store calls through stubs."""

from __future__ import annotations

import numpy as np

import cuda.coop.cutlass as coop

block = coop.this_block()
warp = coop.this_warp()
values = coop.ThreadData(items_per_thread=2, dtype=np.int32)

# Returning Load requires a keyword count and has no destination argument.
coop.load(block, object())  # expected-error: [call-overload]
coop.load(block, object(), 4)  # expected-error: [call-overload]
coop.load(  # expected-error: [call-overload]
    block, object(), values, items_per_thread=2
)
coop.load(  # expected-error: [call-overload]
    block, object(), items_per_thread=1.5
)
coop.load(  # expected-error: [call-overload]
    block, object(), items_per_thread=2, alignment=1.5
)
coop.load(  # expected-error: [call-overload]
    block, object(), items_per_thread=2, oob_default=0
)
coop.load(
    coop.this_grid(),  # expected-error: [arg-type]
    object(),
    items_per_thread=2,
)
coop.load(  # expected-error: [call-overload]
    block, object(), items_per_thread=2, algorithm="unknown"
)
coop.load(
    warp,  # expected-error: [arg-type]
    object(),
    items_per_thread=2,
    temp_storage=coop.TempStorage(),
)
coop.store(
    coop.this_thread(),  # expected-error: [arg-type]
    object(),
    values,
)
coop.store(  # expected-error: [call-overload]
    block, object(), values, valid_items=1.5
)
coop.store(  # expected-error: [call-overload]
    block, object(), values, offset=1.5
)
coop.store(  # expected-error: [call-overload]
    warp, object(), values, algorithm="warp_transpose"
)
coop.store(  # expected-error: [call-overload]
    block, object(), values.to_register_tensor()
)
