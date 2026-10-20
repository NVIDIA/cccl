# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Reject unsupported CUTLASS Load/Store calls in public stubs."""

import numpy as np

import cuda.coop.cutlass as cutlass_coop

block = cutlass_coop.this_block()
warp = cutlass_coop.this_warp()
values = cutlass_coop.ThreadData(items_per_thread=2, dtype=np.int32)
cutlass_coop.load(  # expected-error: [call-overload]
    block, object(), values, algorithm="unknown"
)
cutlass_coop.store(  # expected-error: [call-overload]
    block, object(), values, valid_items=1.5
)
cutlass_coop.store(  # expected-error: [call-overload]
    block, object(), values, offset=1.5
)
cutlass_coop.load(  # expected-error: [call-overload]
    block, object(), values, oob_default=0
)
cutlass_coop.load(
    warp,  # expected-error: [arg-type]
    object(),
    values,
    temp_storage=cutlass_coop.TempStorage(),
)
cutlass_coop.store(  # expected-error: [call-overload]
    warp, object(), values, algorithm="warp_transpose"
)
cutlass_coop.store(  # expected-error: [call-overload]
    block, object(), values.to_register_tensor()
)
