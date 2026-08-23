# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Strict negative typing fixture for group-first segmentation primitives."""

# pyright: strict, reportUnnecessaryTypeIgnoreComment=error

from typing import TYPE_CHECKING


def _subtract(left: int, right: int) -> int:
    return left - right


def _bad_difference(left: int, right: int) -> str:
    return f"{left - right}"


def _different(left: int, right: int) -> bool:
    return left != right


if TYPE_CHECKING:
    import cuda.coop.cutlass as cutlass_coop
    import cuda.coop.numba_mlir as numba_coop
    from cuda import coop

    common_group = coop.this_block()
    common_payload = coop.ThreadData(2, int)
    coop.adjacent_difference(common_group, common_payload, direction="center")  # pyright: ignore[reportCallIssue, reportArgumentType]
    coop.discontinuity(common_group, common_payload, mode="both")  # pyright: ignore[reportCallIssue, reportArgumentType]
    coop.discontinuity(common_group, common_payload, mode="heads_and_tails")  # pyright: ignore[reportCallIssue, reportArgumentType]
    coop.adjacent_difference(common_group, common_payload, difference_op=_subtract)  # pyright: ignore[reportCallIssue]
    coop.discontinuity(common_group, common_payload, flag_op=_different)  # pyright: ignore[reportCallIssue]
    coop.adjacent_difference(common_group, common_payload, common_payload)  # pyright: ignore[reportCallIssue]
    coop.discontinuity(common_group, common_payload, common_payload)  # pyright: ignore[reportCallIssue]
    coop.adjacent_difference(common_group, 1)  # pyright: ignore[reportArgumentType]
    coop.discontinuity(common_group, 1)  # pyright: ignore[reportArgumentType]
    coop.adjacent_difference(  # pyright: ignore[reportCallIssue]
        common_group,
        common_payload,
        direction="left",  # pyright: ignore[reportArgumentType]
        tile_successor_item=0,
    )
    coop.adjacent_difference(  # pyright: ignore[reportCallIssue]
        common_group,
        common_payload,
        direction="right",
        tile_predecessor_item=0,  # pyright: ignore[reportArgumentType]
    )
    coop.adjacent_difference(  # pyright: ignore[reportCallIssue]
        common_group,
        common_payload,
        direction="right",
        valid_items=63,
        tile_successor_item=0,  # pyright: ignore[reportArgumentType]
    )
    coop.adjacent_difference(
        common_group,
        common_payload,  # pyright: ignore[reportArgumentType]
        tile_predecessor_item=1.5,
    )
    coop.discontinuity(  # pyright: ignore[reportCallIssue]
        common_group,
        common_payload,
        mode="heads",  # pyright: ignore[reportArgumentType]
        tile_successor_item=0,
    )
    coop.discontinuity(  # pyright: ignore[reportCallIssue]
        common_group,
        common_payload,
        mode="tails",
        tile_predecessor_item=0,  # pyright: ignore[reportArgumentType]
    )
    coop.discontinuity(
        common_group,
        common_payload,  # pyright: ignore[reportArgumentType]
        tile_predecessor_item=1.5,
    )

    cutlass_group = cutlass_coop.this_block()
    cutlass_payload = cutlass_coop.ThreadData(2, int)
    cutlass_coop.adjacent_difference(  # pyright: ignore[reportCallIssue]
        cutlass_group,
        cutlass_payload,
        difference_op=_subtract,  # pyright: ignore[reportArgumentType]
    )
    cutlass_coop.discontinuity(  # pyright: ignore[reportCallIssue]
        cutlass_group,
        cutlass_payload,
        flag_op=_different,  # pyright: ignore[reportArgumentType]
    )
    cutlass_coop.discontinuity(
        cutlass_group,
        cutlass_payload,
        mode="heads_and_tails",
        invented=True,  # pyright: ignore[reportCallIssue]
    )
    cutlass_coop.adjacent_difference(  # pyright: ignore[reportCallIssue]
        cutlass_group,
        cutlass_payload,
        direction="right",
        valid_items=63,
        tile_successor_item=0,  # pyright: ignore[reportArgumentType]
    )
    cutlass_coop.adjacent_difference(  # pyright: ignore[reportCallIssue]
        cutlass_group,
        cutlass_payload,
        direction="left",  # pyright: ignore[reportArgumentType]
        tile_successor_item=0,
    )
    cutlass_coop.discontinuity(  # pyright: ignore[reportCallIssue]
        cutlass_group,
        cutlass_payload,
        mode="heads",  # pyright: ignore[reportArgumentType]
        tile_successor_item=0,
    )
    cutlass_coop.discontinuity(  # pyright: ignore[reportCallIssue]
        cutlass_group,
        cutlass_payload,
        mode="tails",
        tile_predecessor_item=0,  # pyright: ignore[reportArgumentType]
    )
    cutlass_coop.discontinuity(  # pyright: ignore[reportCallIssue]
        cutlass_group,
        cutlass_payload,  # pyright: ignore[reportArgumentType]
        tile_predecessor_item=1.5,
    )

    numba_group = numba_coop.this_block()
    numba_payload = numba_coop.ThreadData(2, int)
    numba_coop.adjacent_difference(  # pyright: ignore[reportCallIssue]
        numba_group,
        numba_payload,
        difference_op=_bad_difference,  # pyright: ignore[reportArgumentType]
    )
    numba_coop.adjacent_difference(  # pyright: ignore[reportCallIssue]
        numba_group,
        numba_payload,
        direction="center",  # pyright: ignore[reportArgumentType]
    )
    numba_coop.discontinuity(  # pyright: ignore[reportCallIssue]
        numba_group,
        numba_payload,
        mode="both",  # pyright: ignore[reportArgumentType]
    )
    numba_coop.discontinuity(
        numba_group,
        numba_payload,
        mode="heads_and_tails",
        launch_metadata={},  # pyright: ignore[reportCallIssue]
    )
    numba_coop.discontinuity(
        numba_group,
        numba_payload,
        numba_payload,  # pyright: ignore[reportCallIssue]
    )
    numba_coop.adjacent_difference(  # pyright: ignore[reportCallIssue]
        numba_group,
        numba_payload,
        direction="right",
        tile_predecessor_item=0,  # pyright: ignore[reportArgumentType]
    )
    numba_coop.adjacent_difference(  # pyright: ignore[reportCallIssue]
        numba_group,
        numba_payload,
        direction="left",  # pyright: ignore[reportArgumentType]
        tile_successor_item=0,
    )
    numba_coop.discontinuity(  # pyright: ignore[reportCallIssue]
        numba_group,
        numba_payload,
        mode="heads",  # pyright: ignore[reportArgumentType]
        tile_successor_item=0,
    )
    numba_coop.discontinuity(  # pyright: ignore[reportCallIssue]
        numba_group,
        numba_payload,
        mode="tails",
        tile_predecessor_item=0,  # pyright: ignore[reportArgumentType]
    )
    numba_coop.discontinuity(  # pyright: ignore[reportCallIssue]
        numba_group,
        numba_payload,
        tile_predecessor_item=1.5,  # pyright: ignore[reportArgumentType]
    )
