# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Strict negative typing fixture for group-first Reduce and Sum."""

# pyright: strict, reportUnnecessaryTypeIgnoreComment=error

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import cuda.coop.cutlass as cutlass_coop
    import cuda.coop.numba_mlir as numba_coop
    from cuda import coop

    def callback(left: int, right: int) -> int:
        return left + right

    common_thread = coop.this_thread()
    common_warp = coop.this_warp()
    common_threads = common_warp.group_by(8)
    common_block = coop.this_block()
    common_cluster = coop.this_cluster()
    common_payload = coop.ThreadData(2, int)
    invalid_object: object = object()
    invalid_list = [1, 2]

    # Direct-CUB selectors are scalar-only and root-only.
    coop.reduce(common_block, common_payload, broadcast=False, valid_items=64)  # pyright: ignore[reportCallIssue, reportArgumentType]
    coop.sum(common_block, common_payload, broadcast=False, algorithm="raking")  # pyright: ignore[reportCallIssue, reportArgumentType]
    coop.reduce(common_block, 1, valid_items=64)  # pyright: ignore[reportCallIssue, reportArgumentType]
    coop.reduce(common_block, 1, broadcast=True, algorithm="raking")  # pyright: ignore[reportCallIssue, reportArgumentType]
    coop.sum(common_warp, 1, broadcast=True, valid_items=24)  # pyright: ignore[reportCallIssue, reportArgumentType]

    # CUB BlockReduce algorithms do not apply to WarpReduce, and no direct
    # selector applies to the other certified CUDAX reduction groups.
    coop.reduce(common_warp, 1, broadcast=False, algorithm="raking")  # pyright: ignore[reportCallIssue, reportArgumentType]
    coop.sum(common_thread, 1, broadcast=False, valid_items=1)  # pyright: ignore[reportCallIssue, reportArgumentType]
    coop.reduce(common_cluster, 1, broadcast=False, algorithm="raking")  # pyright: ignore[reportCallIssue, reportArgumentType]

    coop.reduce(common_block, 1, binary_op=callback)  # pyright: ignore[reportCallIssue, reportArgumentType]
    coop.reduce(common_block, 1, algorithm="tree", broadcast=False)  # pyright: ignore[reportCallIssue, reportArgumentType]
    coop.sum(common_block, 1, broadcast="all")  # pyright: ignore[reportCallIssue, reportArgumentType]
    coop.reduce(common_block, invalid_list)  # pyright: ignore[reportCallIssue, reportArgumentType]
    coop.sum(common_block, invalid_object)  # pyright: ignore[reportCallIssue, reportArgumentType]

    # Valid prefixes accept only Python, NumPy, or structural compiler integer
    # values. Strings, floating values, opaque objects, and callables fail
    # before backend lowering.
    coop.reduce(common_block, 1, broadcast=False, valid_items="64")  # pyright: ignore[reportCallIssue, reportArgumentType]
    coop.reduce(common_block, 1, broadcast=False, valid_items=64.0)  # pyright: ignore[reportCallIssue, reportArgumentType]
    coop.reduce(common_block, 1, broadcast=False, valid_items=invalid_object)  # pyright: ignore[reportCallIssue, reportArgumentType]
    coop.reduce(common_block, 1, broadcast=False, valid_items=callback)  # pyright: ignore[reportCallIssue, reportArgumentType]

    cutlass_block = cutlass_coop.this_block()
    cutlass_warp = cutlass_coop.this_warp()
    cutlass_cluster = cutlass_coop.this_cluster()
    cutlass_payload = cutlass_coop.ThreadData(2, int)
    cutlass_coop.reduce(cutlass_block, invalid_list)  # pyright: ignore[reportCallIssue, reportArgumentType]
    cutlass_coop.sum(cutlass_block, invalid_object)  # pyright: ignore[reportCallIssue, reportArgumentType]
    cutlass_coop.reduce(cutlass_block, 1, broadcast=False, valid_items="64")  # pyright: ignore[reportCallIssue, reportArgumentType]
    cutlass_coop.reduce(cutlass_block, 1, broadcast=False, valid_items=64.0)  # pyright: ignore[reportCallIssue, reportArgumentType]
    cutlass_coop.reduce(cutlass_block, 1, broadcast=False, valid_items=invalid_object)  # pyright: ignore[reportCallIssue, reportArgumentType]
    cutlass_coop.reduce(cutlass_block, 1, broadcast=False, valid_items=callback)  # pyright: ignore[reportCallIssue, reportArgumentType]
    cutlass_coop.reduce(  # pyright: ignore[reportCallIssue]
        cutlass_block,
        cutlass_payload,
        broadcast=False,
        valid_items=64,  # pyright: ignore[reportArgumentType]
    )
    cutlass_coop.sum(  # pyright: ignore[reportCallIssue]
        cutlass_block,
        cutlass_payload,
        broadcast=False,
        algorithm="raking",  # pyright: ignore[reportArgumentType]
    )
    cutlass_coop.reduce(cutlass_block, 1, valid_items=64)  # pyright: ignore[reportCallIssue, reportArgumentType]
    cutlass_coop.reduce(  # pyright: ignore[reportCallIssue]
        cutlass_warp,
        1,
        broadcast=False,
        algorithm="raking",  # pyright: ignore[reportArgumentType]
    )
    cutlass_coop.sum(  # pyright: ignore[reportCallIssue]
        cutlass_cluster,
        1,
        broadcast=False,
        valid_items=2,  # pyright: ignore[reportArgumentType]
    )

    numba_block = numba_coop.this_block()
    numba_warp = numba_coop.this_warp()
    numba_cluster = numba_coop.this_cluster()
    numba_block_warps = numba_block.group_by(32)
    numba_payload = numba_coop.ThreadData(2, int)
    numba_coop.reduce(numba_block, invalid_list)  # pyright: ignore[reportCallIssue, reportArgumentType]
    numba_coop.sum(numba_block, invalid_object)  # pyright: ignore[reportCallIssue, reportArgumentType]
    numba_coop.reduce(numba_block, 1, broadcast=False, valid_items="64")  # pyright: ignore[reportCallIssue, reportArgumentType]
    numba_coop.reduce(numba_block, 1, broadcast=False, valid_items=64.0)  # pyright: ignore[reportCallIssue, reportArgumentType]
    numba_coop.reduce(numba_block, 1, broadcast=False, valid_items=invalid_object)  # pyright: ignore[reportCallIssue, reportArgumentType]
    numba_coop.reduce(numba_block, 1, broadcast=False, valid_items=callback)  # pyright: ignore[reportCallIssue, reportArgumentType]
    numba_coop.reduce(numba_block_warps, 1)  # pyright: ignore[reportCallIssue, reportArgumentType]
    numba_coop.reduce(  # pyright: ignore[reportCallIssue]
        numba_block,
        numba_payload,
        broadcast=False,
        valid_items=64,  # pyright: ignore[reportArgumentType]
    )
    numba_coop.sum(  # pyright: ignore[reportCallIssue]
        numba_block,
        numba_payload,
        broadcast=False,
        algorithm="raking",  # pyright: ignore[reportArgumentType]
    )
    numba_coop.reduce(numba_block, 1, valid_items=64)  # pyright: ignore[reportCallIssue, reportArgumentType]
    numba_coop.reduce(  # pyright: ignore[reportCallIssue]
        numba_warp,
        1,
        broadcast=False,
        algorithm="raking",  # pyright: ignore[reportArgumentType]
    )
    numba_coop.sum(  # pyright: ignore[reportCallIssue]
        numba_cluster,
        1,
        broadcast=False,
        valid_items=2,  # pyright: ignore[reportArgumentType]
    )

    # Group-first Numba Reduce intentionally exposes only portable built-in
    # aliases. Private block/warp compatibility factories retain callback support.
    numba_coop.reduce(numba_block, 1, binary_op=callback)  # pyright: ignore[reportCallIssue, reportArgumentType]
