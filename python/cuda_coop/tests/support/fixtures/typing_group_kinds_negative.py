# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Strict Pyright fixture proving illegal common-profile groups are rejected."""

# pyright: strict, reportUnnecessaryTypeIgnoreComment=error

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import cuda.coop.cutlass as cutlass_coop
    import cuda.coop.numba_mlir as numba_coop
    from cuda import coop

    payload = coop.ThreadData(2, int)
    cutlass_payload = cutlass_coop.ThreadData(2, int)
    numba_payload = numba_coop.ThreadData(2, int)
    thread = coop.this_thread()
    warp = coop.this_warp()
    block = coop.this_block()
    cluster = coop.this_cluster()
    grid = coop.this_grid()
    warp_scalar: int = 1

    coop.load(grid, object(), payload)  # pyright: ignore[reportCallIssue, reportArgumentType]
    coop.store(thread, object(), payload)  # pyright: ignore[reportCallIssue, reportArgumentType]
    coop.scan(thread, payload)  # pyright: ignore[reportCallIssue, reportArgumentType]
    coop.scan(warp, payload)  # pyright: ignore[reportCallIssue, reportArgumentType]
    coop.exclusive_sum(warp, payload)  # pyright: ignore[reportCallIssue, reportArgumentType]
    coop.inclusive_sum(warp, payload)  # pyright: ignore[reportCallIssue, reportArgumentType]
    coop.exclusive_scan(warp, payload)  # pyright: ignore[reportCallIssue, reportArgumentType]
    coop.inclusive_scan(warp, payload)  # pyright: ignore[reportCallIssue, reportArgumentType]
    coop.scan(warp, warp_scalar, algorithm="raking")  # pyright: ignore[reportCallIssue, reportArgumentType]
    coop.scan(warp, warp_scalar, temp_storage=coop.TempStorage())  # pyright: ignore[reportCallIssue, reportArgumentType]
    coop.exchange(thread, payload)  # pyright: ignore[reportArgumentType]
    coop.exchange(cluster, payload)  # pyright: ignore[reportArgumentType]
    coop.exchange(grid, payload)  # pyright: ignore[reportArgumentType]
    coop.reduce(grid, payload)  # pyright: ignore[reportCallIssue, reportArgumentType]
    grid.sync()  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
    grid.sync_aligned()  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]

    coop.adjacent_difference(warp, payload)  # pyright: ignore[reportArgumentType]
    coop.discontinuity(warp, payload)  # pyright: ignore[reportArgumentType]
    coop.shuffle(warp, payload)  # pyright: ignore[reportArgumentType]
    coop.radix_sort_keys(warp, payload)  # pyright: ignore[reportArgumentType]
    coop.radix_rank(warp, payload)  # pyright: ignore[reportArgumentType]
    coop.histogram(warp, payload, bins=8)  # pyright: ignore[reportArgumentType]
    coop.run_length_decode(
        warp,  # pyright: ignore[reportArgumentType]
        payload,
        payload,
        decoded_items_per_thread=2,
    )
    coop.topk_max_keys(warp, payload, 1)  # pyright: ignore[reportArgumentType]

    # A partitioned block is a group of warps, not a block collective group.
    block_warps = block.group_by(32)
    coop.reduce(block_warps, payload)  # pyright: ignore[reportCallIssue, reportArgumentType]
    coop.exchange(block_warps, payload)  # pyright: ignore[reportArgumentType]
    coop.histogram(block_warps, payload, bins=8)  # pyright: ignore[reportArgumentType]

    # Only physical warps and blocks can be partitioned by the portable API.
    thread.group_by(1)  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
    cluster.group_by(1)  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
    grid.group_by(1)  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
    warp.group_by(8).group_by(2)  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
    block_warps.group_by(2)  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]

    # Qualified constructors remain precise when passed to the portable root.
    coop.histogram(
        cutlass_coop.this_warp(),  # pyright: ignore[reportArgumentType]
        payload,
        bins=8,
    )
    coop.reduce(  # pyright: ignore[reportCallIssue]
        numba_coop.this_grid(),  # pyright: ignore[reportArgumentType]
        payload,
    )

    cutlass_coop.load(  # pyright: ignore[reportCallIssue]
        cutlass_coop.this_grid(),  # pyright: ignore[reportArgumentType]
        object(),
        cutlass_payload,
    )
    cutlass_coop.reduce(  # pyright: ignore[reportCallIssue]
        cutlass_coop.this_grid(),  # pyright: ignore[reportArgumentType]
        cutlass_payload,
    )
    cutlass_coop.histogram(
        cutlass_coop.this_warp(),  # pyright: ignore[reportArgumentType]
        cutlass_payload,
        bins=8,
    )
    cutlass_coop.scan(  # pyright: ignore[reportCallIssue]
        cutlass_coop.this_warp(),
        cutlass_payload,  # pyright: ignore[reportArgumentType]
    )
    cutlass_coop.exchange(  # pyright: ignore[reportCallIssue]
        cutlass_coop.this_cluster(),  # pyright: ignore[reportArgumentType]
        cutlass_payload,
    )
    cutlass_coop.exchange(  # pyright: ignore[reportCallIssue]
        cutlass_coop.this_warp(),
        cutlass_payload,
        mode="scatter_to_blocked",  # pyright: ignore[reportArgumentType]
    )
    cutlass_coop.exchange(  # pyright: ignore[reportCallIssue]
        cutlass_coop.this_warp(),
        cutlass_payload,
        valid_flags=cutlass_payload,  # pyright: ignore[reportArgumentType]
    )
    cutlass_coop.exchange(  # pyright: ignore[reportCallIssue]
        cutlass_coop.this_warp(),
        cutlass_payload,
        warp_time_slicing=True,  # pyright: ignore[reportArgumentType]
    )
    cutlass_coop.radix_sort_pairs(  # pyright: ignore[reportCallIssue]
        cutlass_coop.this_warp(),  # pyright: ignore[reportArgumentType]
        cutlass_payload,
        cutlass_payload,
    )
    cutlass_coop.this_cluster().group_by(1)  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]

    numba_coop.store(  # pyright: ignore[reportCallIssue]
        numba_coop.this_thread(),  # pyright: ignore[reportArgumentType]
        object(),
        payload,
    )
    numba_coop.reduce(  # pyright: ignore[reportCallIssue]
        numba_coop.this_grid(),  # pyright: ignore[reportArgumentType]
        payload,
    )
    numba_coop.this_grid().sync()  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
    numba_coop.this_grid().sync_aligned()  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
    numba_coop.scan(  # pyright: ignore[reportCallIssue]
        numba_coop.this_warp(),
        payload,  # pyright: ignore[reportArgumentType]
    )
    numba_coop.exchange(  # pyright: ignore[reportCallIssue]
        numba_coop.this_thread(),  # pyright: ignore[reportArgumentType]
        payload,
    )
    numba_coop.exchange(  # pyright: ignore[reportCallIssue]
        numba_coop.this_warp(),
        numba_payload,
        mode="scatter_to_blocked",  # pyright: ignore[reportArgumentType]
    )
    numba_coop.exchange(  # pyright: ignore[reportCallIssue]
        numba_coop.this_warp(),
        numba_payload,
        valid_flags=numba_payload,  # pyright: ignore[reportArgumentType]
    )
    numba_coop.exchange(  # pyright: ignore[reportCallIssue]
        numba_coop.this_warp(),
        numba_payload,
        warp_time_slicing=True,  # pyright: ignore[reportArgumentType]
    )
    numba_coop.topk_min_pairs(
        numba_coop.this_warp(),  # pyright: ignore[reportArgumentType]
        payload,
        payload,
        1,
    )
    numba_coop.this_thread().group_by(1)  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
