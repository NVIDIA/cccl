# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Strict positive typing fixture for group-first Reduce and Sum."""

# pyright: strict, reportUnnecessaryTypeIgnoreComment=error

from typing import TYPE_CHECKING

import numpy as np
from typing_extensions import assert_type

if TYPE_CHECKING:
    import cuda.coop.cutlass as cutlass_coop
    import cuda.coop.numba_mlir as numba_coop
    from cuda import coop

    common_thread = coop.this_thread()
    common_warp = coop.this_warp()
    common_threads = common_warp.group_by(8)
    common_block = coop.this_block()
    common_cluster = coop.this_cluster()
    common_payload = coop.ThreadData(2, np.int32)
    common_scalar = np.int32(1)

    # Full-group CUDAX routes accept scalar and multi-item payload operands for
    # every certified reduction group, with either result visibility policy.
    assert_type(coop.reduce(common_thread, common_payload), np.int32)
    assert_type(coop.reduce(common_warp, common_payload), np.int32)
    assert_type(coop.reduce(common_threads, common_payload), np.int32)
    assert_type(coop.reduce(common_block, common_payload), np.int32)
    assert_type(coop.reduce(common_cluster, common_payload), np.int32)
    assert_type(coop.reduce(common_block, common_scalar), np.int32)
    assert_type(
        coop.reduce(common_block, common_scalar, broadcast=False),
        np.int32,
    )
    assert_type(coop.sum(common_cluster, common_payload), np.int32)
    assert_type(coop.sum(common_thread, common_scalar), np.int32)

    # A non-None selector chooses direct CUB. BlockReduce supports both
    # selectors, while physical and logical WarpReduce support only a valid
    # prefix. Both routes require the caller to opt in to root-only result
    # visibility.
    assert_type(
        coop.reduce(
            common_block,
            common_scalar,
            broadcast=False,
            valid_items=np.int32(64),
        ),
        np.int32,
    )
    assert_type(
        coop.reduce(
            common_block,
            common_scalar,
            binary_op="max",
            broadcast=False,
            algorithm="raking",
        ),
        np.int32,
    )
    assert_type(
        coop.sum(
            common_block,
            common_scalar,
            broadcast=False,
            valid_items=64,
            algorithm="warp_reductions",
        ),
        np.int32,
    )
    assert_type(
        coop.reduce(
            common_warp,
            common_scalar,
            broadcast=False,
            valid_items=np.uint64(24),
        ),
        np.int32,
    )
    assert_type(
        coop.sum(
            common_warp,
            common_scalar,
            broadcast=False,
            valid_items=24,
        ),
        np.int32,
    )
    assert_type(
        coop.reduce(
            common_threads,
            common_scalar,
            broadcast=False,
            valid_items=8,
        ),
        np.int32,
    )

    cutlass_block = cutlass_coop.this_block()
    cutlass_warp = cutlass_coop.this_warp()
    cutlass_cluster = cutlass_coop.this_cluster()
    cutlass_block_warps = cutlass_block.group_by(32)
    cutlass_payload = cutlass_coop.ThreadData(2, np.int32)
    assert_type(cutlass_coop.reduce(cutlass_cluster, cutlass_payload), np.int32)
    assert_type(cutlass_coop.sum(cutlass_cluster, cutlass_payload), np.int32)
    assert_type(
        cutlass_coop.reduce(cutlass_block_warps, cutlass_payload),
        np.int32,
    )
    assert_type(cutlass_coop.sum(cutlass_block_warps, common_scalar), np.int32)
    assert_type(
        cutlass_coop.reduce(
            cutlass_block,
            common_scalar,
            broadcast=False,
            valid_items=64,
            algorithm="raking",
        ),
        np.int32,
    )
    assert_type(
        cutlass_coop.sum(
            cutlass_warp,
            common_scalar,
            broadcast=False,
            valid_items=24,
        ),
        np.int32,
    )

    numba_block = numba_coop.this_block()
    numba_warp = numba_coop.this_warp()
    numba_cluster = numba_coop.this_cluster()
    numba_payload = numba_coop.ThreadData(2, np.int32)
    assert_type(numba_coop.reduce(numba_cluster, numba_payload), np.int32)
    assert_type(numba_coop.sum(numba_cluster, numba_payload), np.int32)
    assert_type(
        numba_coop.reduce(
            numba_block,
            common_scalar,
            binary_op="maximum",
            broadcast=False,
            valid_items=64,
            algorithm="warp_reductions",
        ),
        np.int32,
    )
    assert_type(
        numba_coop.sum(
            numba_warp,
            common_scalar,
            broadcast=False,
            valid_items=24,
        ),
        np.int32,
    )
    # Qualified group descriptors remain valid inputs to the one portable root
    # contract without changing the static payload/result relationship.
    assert_type(coop.reduce(cutlass_cluster, common_payload), np.int32)
    assert_type(
        coop.sum(
            numba_block,
            common_scalar,
            broadcast=False,
            algorithm="raking",
        ),
        np.int32,
    )
