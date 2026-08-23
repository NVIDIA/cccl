# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Strict Pyright fixture for legal common-profile group combinations."""

# pyright: strict, reportUnnecessaryTypeIgnoreComment=error

from typing import TYPE_CHECKING, Literal

from typing_extensions import assert_type

if TYPE_CHECKING:
    import cuda.coop.cutlass as cutlass_coop
    import cuda.coop.numba_mlir as numba_coop
    from cuda import coop

    payload = coop.ThreadData(2, int)
    thread = coop.this_thread()
    warp = coop.this_warp()
    block = coop.this_block()
    cluster = coop.this_cluster()
    threads = warp.group_by(8)
    warps = block.group_by(32)

    assert_type(thread, coop.ThreadGroup[Literal["thread"]])
    assert_type(warp, coop.ThreadGroup[Literal["warp"]])
    assert_type(block, coop.ThreadGroup[Literal["block"]])
    assert_type(cluster, coop.ThreadGroup[Literal["cluster"]])
    assert_type(threads, coop.ThreadGroup[Literal["threads_within_warp"]])
    assert_type(warps, coop.ThreadGroup[Literal["warps_within_block"]])

    for synchronizable in (thread, warp, block, cluster, threads, warps):
        synchronizable.sync()
        synchronizable.sync_aligned()

    assert_type(coop.reduce(thread, 1), int)
    assert_type(coop.reduce(warp, payload), int)
    assert_type(coop.reduce(threads, payload), int)
    assert_type(coop.reduce(block, payload), int)
    assert_type(coop.reduce(cluster, payload), int)

    assert_type(coop.load(warp, object(), payload), coop.ThreadDataLike[int])
    assert_type(coop.load(block, object(), payload), coop.ThreadDataLike[int])
    assert_type(coop.scan(block, payload), coop.ThreadDataLike[int])
    warp_value: int = 1
    assert_type(coop.scan(warp, warp_value), int)
    assert_type(coop.exchange(block, payload), coop.ThreadDataLike[int])
    assert_type(
        coop.exchange(warp, payload, mode="blocked_to_striped"),
        coop.ThreadDataLike[int],
    )
    assert_type(coop.merge_sort_keys(warp, payload), coop.ThreadDataLike[int])
    assert_type(coop.histogram(block, payload, bins=8), coop.ThreadDataLike[int])

    # Backend group descriptors retain their concrete qualified type while also
    # satisfying the portable root's group-kind contract.
    cutlass_block = cutlass_coop.this_block()
    cutlass_warp = cutlass_coop.this_warp()
    cutlass_threads = cutlass_warp.group_by(8)
    cutlass_payload = cutlass_coop.ThreadData(2, int)
    numba_warp = numba_coop.this_warp()
    numba_thread = numba_coop.this_thread()
    numba_block = numba_coop.this_block()
    numba_cluster = numba_coop.this_cluster()
    numba_threads = numba_warp.group_by(8)
    numba_warps = numba_block.group_by(32)
    assert_type(
        cutlass_block,
        cutlass_coop.ThreadGroup[Literal["block"]],
    )
    assert_type(numba_warp, numba_coop.ThreadGroup[Literal["warp"]])
    for synchronizable in (
        numba_thread,
        numba_warp,
        numba_block,
        numba_cluster,
        numba_threads,
        numba_warps,
    ):
        synchronizable.sync()
        synchronizable.sync_aligned()
    cutlass_coop.this_grid().sync()
    cutlass_coop.this_grid().sync_aligned()
    assert_type(
        coop.histogram(cutlass_block, payload, bins=8),
        coop.ThreadDataLike[int],
    )
    assert_type(coop.scan(numba_warp, warp_value), int)
    assert_type(
        cutlass_block.group_by(32),
        cutlass_coop.ThreadGroup[Literal["warps_within_block"]],
    )
    assert_type(
        numba_warp.group_by(8),
        numba_coop.ThreadGroup[Literal["threads_within_warp"]],
    )
    assert_type(
        cutlass_coop.load(cutlass_coop.this_warp(), object(), cutlass_payload),
        cutlass_coop.ThreadData[int],
    )
    assert_type(
        cutlass_coop.exchange(cutlass_threads, cutlass_payload),
        cutlass_coop.ThreadData[int],
    )
    assert_type(
        cutlass_coop.exchange(
            cutlass_block,
            cutlass_payload,
            mode="scatter_to_blocked",
            ranks=cutlass_payload,
        ),
        cutlass_coop.ThreadData[int],
    )
    assert_type(
        cutlass_coop.exchange(
            cutlass_threads,
            cutlass_payload,
            mode="scatter_to_striped",
            ranks=cutlass_payload,
        ),
        cutlass_coop.ThreadData[int],
    )
    assert_type(
        cutlass_coop.reduce(cutlass_coop.this_cluster(), cutlass_payload),
        int,
    )
    assert_type(
        numba_coop.merge_sort_pairs(
            numba_coop.this_block(),
            payload,
            payload,
        ),
        tuple[coop.ThreadDataLike[int], coop.ThreadDataLike[int]],
    )
    numba_payload = numba_coop.ThreadData(2, int)
    assert_type(
        numba_coop.exchange(numba_threads, numba_payload),
        coop.ThreadDataLike[int],
    )
    assert_type(
        numba_coop.exchange(
            numba_block,
            numba_payload,
            mode="scatter_to_blocked",
            ranks=numba_payload,
        ),
        coop.ThreadDataLike[int],
    )
    assert_type(
        numba_coop.exchange(
            numba_threads,
            numba_payload,
            mode="scatter_to_striped",
            ranks=numba_payload,
        ),
        coop.ThreadDataLike[int],
    )
    assert_type(
        numba_coop.radix_sort_pairs(
            numba_coop.this_block(),
            payload,
            payload,
        ),
        tuple[coop.ThreadDataLike[int], coop.ThreadDataLike[int]],
    )
