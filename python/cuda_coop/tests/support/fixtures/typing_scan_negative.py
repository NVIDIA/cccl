# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Strict negative Pyright fixture for group-first Scan call families."""

# pyright: strict, reportUnnecessaryTypeIgnoreComment=error

from typing import TYPE_CHECKING


def _maximum(left: int, right: int) -> int:
    return max(left, right)


if TYPE_CHECKING:
    import cuda.coop.cutlass as cutlass_coop
    import cuda.coop.numba_mlir as numba_coop
    from cuda import coop

    common_block = coop.this_block()
    common_warp = coop.this_warp()
    common_payload = coop.ThreadData(2, int)
    coop.scan(common_block, common_payload, scan_op="max")  # pyright: ignore[reportCallIssue, reportArgumentType]
    coop.scan(common_warp, 1, scan_op="min", initial_value=None)  # pyright: ignore[reportCallIssue, reportArgumentType]
    coop.scan(common_block, common_payload, mode="inclusive", initial_value=0)  # pyright: ignore[reportCallIssue, reportArgumentType]
    coop.exclusive_scan(common_block, common_payload, scan_op="max")  # pyright: ignore[reportCallIssue, reportArgumentType]
    coop.inclusive_scan(common_block, common_payload, initial_value=0)  # pyright: ignore[reportCallIssue]

    cutlass_block = cutlass_coop.this_block()
    cutlass_warp = cutlass_coop.this_warp()
    cutlass_payload = cutlass_coop.ThreadData(2, int)
    cutlass_coop.scan(cutlass_block, cutlass_payload, valid_items=4)  # pyright: ignore[reportCallIssue, reportArgumentType]
    cutlass_coop.scan(cutlass_block, cutlass_payload, scan_op="max")  # pyright: ignore[reportCallIssue, reportArgumentType]
    cutlass_coop.scan(  # pyright: ignore[reportCallIssue]
        cutlass_block,
        cutlass_payload,
        mode="inclusive",
        initial_value=0,
        launch_metadata={"threads_per_block": 128},
    )
    cutlass_coop.exclusive_scan(cutlass_warp, 1, scan_op="min")  # pyright: ignore[reportCallIssue, reportArgumentType]
    cutlass_coop.inclusive_scan(  # pyright: ignore[reportCallIssue]
        cutlass_block,
        cutlass_payload,
        initial_value=0,
    )

    numba_block = numba_coop.this_block()
    numba_warp = numba_coop.this_warp()
    numba_payload = numba_coop.ThreadData(2, int)
    numba_coop.scan(numba_block, numba_payload, valid_items=4)  # pyright: ignore[reportCallIssue, reportArgumentType]
    numba_coop.scan(numba_block, numba_payload, scan_op="max")  # pyright: ignore[reportCallIssue, reportArgumentType]
    numba_coop.scan(numba_block, numba_payload, scan_op=_maximum)  # pyright: ignore[reportCallIssue, reportArgumentType]
    numba_coop.scan(  # pyright: ignore[reportCallIssue]
        numba_warp,
        1,
        scan_op=_maximum,
        initial_value=None,  # pyright: ignore[reportArgumentType]
    )
    numba_coop.scan(  # pyright: ignore[reportCallIssue]
        numba_block,
        numba_payload,
        mode="inclusive",
        scan_op=_maximum,
        initial_value=0,  # pyright: ignore[reportArgumentType]
    )
    numba_coop.exclusive_scan(  # pyright: ignore[reportCallIssue]
        numba_block,
        numba_payload,
        scan_op=_maximum,  # pyright: ignore[reportArgumentType]
    )
    numba_coop.inclusive_scan(  # pyright: ignore[reportCallIssue]
        numba_block,
        numba_payload,
        scan_op=_maximum,
        initial_value=0,
    )
