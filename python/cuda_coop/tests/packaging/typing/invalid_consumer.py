# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Deliberately invalid calls proving installed stubs reject misuse."""

from __future__ import annotations

from typing_extensions import assert_type

import cuda.coop as portable
import cuda.coop.cutlass as cutlass
import cuda.coop.numba_mlir as numba

portable_values = portable.ThreadData(2, int)
assert_type(
    portable.exchange(portable.this_block(), portable_values),
    portable.ThreadDataLike[float],
)

cutlass_values = cutlass.ThreadData(2, float)
assert_type(cutlass.reduce(cutlass.this_block(), cutlass_values), int)


def prefix_from_aggregate(aggregate: int) -> int:
    """Return a prefix for invalid warp-callback coverage."""

    return aggregate


numba.scan(
    numba.this_warp(),
    1,
    prefix_op=prefix_from_aggregate,
)
bad_block_load: numba.BlockLoadAlgorithm = 0
