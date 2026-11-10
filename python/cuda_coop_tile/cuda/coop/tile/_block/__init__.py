# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Normalize Python options and validate known CUB specialization values.

Lowerings use option records before dispatch, then construct block adapters
with a candidate width and per-thread extent. These objects describe and
check a specialization; constructing one emits no IR and executes no GPU
work. Runtime controls may be represented by validation-only placeholders.
The lowering sends their actual values as foreign-call operands, and the
native provider independently validates the selected implementation.
"""

from .adjacent_difference import (
    BlockAdjacentDifferenceSubtractLeft,
    BlockAdjacentDifferenceSubtractRight,
)
from .discontinuity import (
    BlockDiscontinuityFlagHeads,
    BlockDiscontinuityFlagHeadsAndTails,
    BlockDiscontinuityFlagTails,
)
from .reduce import BlockReduce, BlockReduceSum
from .scan import (
    BlockScanExclusiveScan,
    BlockScanExclusiveSum,
    BlockScanInclusiveScan,
    BlockScanInclusiveSum,
)
from .shuffle import BlockShuffle

__all__ = [
    "BlockAdjacentDifferenceSubtractLeft",
    "BlockAdjacentDifferenceSubtractRight",
    "BlockDiscontinuityFlagHeads",
    "BlockDiscontinuityFlagHeadsAndTails",
    "BlockDiscontinuityFlagTails",
    "BlockReduce",
    "BlockReduceSum",
    "BlockScanExclusiveScan",
    "BlockScanExclusiveSum",
    "BlockScanInclusiveScan",
    "BlockScanInclusiveSum",
    "BlockShuffle",
]
