# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Strict negative typing fixture for group-specific Load/Store algorithms."""

# pyright: strict, reportUnnecessaryTypeIgnoreComment=error

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import cuda.coop.cutlass as cutlass_coop
    import cuda.coop.numba_mlir as numba_coop
    from cuda import coop

    common_block = coop.this_block()
    common_warp = coop.this_warp()
    common_payload = coop.ThreadData(2, int)
    coop.load(common_warp, object(), common_payload, algorithm="warp_transpose")  # pyright: ignore[reportCallIssue, reportArgumentType]
    coop.store(  # pyright: ignore[reportCallIssue]
        common_warp,
        object(),
        common_payload,
        algorithm="warp_transpose_timesliced",  # pyright: ignore[reportArgumentType]
    )
    coop.load(common_block, object(), common_payload, algorithm="coalesced")  # pyright: ignore[reportCallIssue, reportArgumentType]
    coop.load(common_block, object(), common_payload, oob_default=0)  # pyright: ignore[reportCallIssue, reportArgumentType]
    coop.load(  # pyright: ignore[reportCallIssue]
        common_block,
        object(),
        common_payload,
        valid_items=1,
        oob_default=1.5,  # pyright: ignore[reportArgumentType]
    )
    coop.load(common_block, object(), common_payload, valid_items=1.5)  # pyright: ignore[reportCallIssue, reportArgumentType]
    coop.store(common_block, object(), common_payload, offset=1.5)  # pyright: ignore[reportCallIssue, reportArgumentType]

    cutlass_block = cutlass_coop.this_block()
    cutlass_warp = cutlass_coop.this_warp()
    cutlass_payload = cutlass_coop.ThreadData(2, int)
    cutlass_coop.load(  # pyright: ignore[reportCallIssue]
        cutlass_warp,
        object(),
        cutlass_payload,
        algorithm="warp_transpose",  # pyright: ignore[reportArgumentType]
    )
    cutlass_coop.store(  # pyright: ignore[reportCallIssue]
        cutlass_warp,
        object(),
        cutlass_payload,
        algorithm="warp_transpose_timesliced",  # pyright: ignore[reportArgumentType]
    )
    cutlass_coop.store(  # pyright: ignore[reportCallIssue]
        cutlass_block,
        object(),
        cutlass_payload,
        algorithm="coalesced",  # pyright: ignore[reportArgumentType]
    )
    cutlass_coop.load(  # pyright: ignore[reportCallIssue]
        cutlass_block,
        object(),
        cutlass_payload,
        oob_default=0,  # pyright: ignore[reportArgumentType]
    )
    cutlass_coop.load(  # pyright: ignore[reportCallIssue]
        cutlass_block,
        object(),
        cutlass_payload,
        valid_items=1,
        oob_default=1.5,  # pyright: ignore[reportArgumentType]
    )
    cutlass_coop.store(  # pyright: ignore[reportCallIssue]
        cutlass_warp,
        object(),
        cutlass_payload,
        valid_items="one",  # pyright: ignore[reportArgumentType]
    )

    numba_block = numba_coop.this_block()
    numba_warp = numba_coop.this_warp()
    numba_payload = numba_coop.ThreadData(2, int)
    numba_coop.load(  # pyright: ignore[reportCallIssue]
        numba_warp,
        object(),
        numba_payload,
        algorithm="warp_transpose",  # pyright: ignore[reportArgumentType]
    )
    numba_coop.store(  # pyright: ignore[reportCallIssue]
        numba_warp,
        object(),
        numba_payload,
        algorithm="warp_transpose_timesliced",  # pyright: ignore[reportArgumentType]
    )
    numba_coop.load(  # pyright: ignore[reportCallIssue]
        numba_warp,
        object(),
        numba_payload,
        algorithm=numba_coop.BlockLoadAlgorithm.DIRECT,  # pyright: ignore[reportArgumentType]
    )
    numba_coop.store(  # pyright: ignore[reportCallIssue]
        numba_warp,
        object(),
        numba_payload,
        algorithm=numba_coop.BlockStoreAlgorithm.DIRECT,  # pyright: ignore[reportArgumentType]
    )
    numba_coop.load(  # pyright: ignore[reportCallIssue]
        numba_block,
        object(),
        numba_payload,
        algorithm=numba_coop.WarpLoadAlgorithm.DIRECT,  # pyright: ignore[reportArgumentType]
    )
    numba_coop.store(  # pyright: ignore[reportCallIssue]
        numba_block,
        object(),
        numba_payload,
        algorithm=numba_coop.WarpStoreAlgorithm.DIRECT,  # pyright: ignore[reportArgumentType]
    )
    numba_coop.load(  # pyright: ignore[reportCallIssue]
        numba_block,
        object(),
        numba_payload,
        oob_default=0,  # pyright: ignore[reportArgumentType]
    )
    numba_coop.load(  # pyright: ignore[reportCallIssue]
        numba_warp,
        object(),
        numba_payload,
        valid_items=1,
        oob_default=1.5,  # pyright: ignore[reportArgumentType]
    )
    numba_coop.store(  # pyright: ignore[reportCallIssue]
        numba_block,
        object(),
        numba_payload,
        offset="four",  # pyright: ignore[reportArgumentType]
    )
