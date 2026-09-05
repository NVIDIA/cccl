# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Strict negative typing fixture for group-first Shuffle."""

# pyright: strict, reportUnnecessaryTypeIgnoreComment=error

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import cuda.coop.cutlass as cutlass_coop
    import cuda.coop.numba_mlir as numba_coop
    from cuda import coop

    common_group = coop.this_block()
    common_payload = coop.ThreadData(3, int)
    coop.shuffle(common_group, 7)  # pyright: ignore[reportArgumentType]
    coop.shuffle(common_group, common_payload, mode="offset")  # pyright: ignore[reportArgumentType]
    coop.shuffle(common_group, common_payload, mode="rotate")  # pyright: ignore[reportArgumentType]
    coop.shuffle(common_group, common_payload, distance=2)  # pyright: ignore[reportArgumentType]
    coop.shuffle(common_group, common_payload, block_prefix=common_payload)  # pyright: ignore[reportCallIssue]
    coop.shuffle(common_group, common_payload, common_payload)  # pyright: ignore[reportCallIssue]

    cutlass_group = cutlass_coop.this_block()
    cutlass_payload = cutlass_coop.ThreadData(3, int)
    cutlass_edge = cutlass_coop.ThreadData(1, int)
    cutlass_coop.shuffle(cutlass_group, cutlass_payload, mode="offset")  # pyright: ignore[reportCallIssue, reportArgumentType]
    cutlass_coop.shuffle(cutlass_group, cutlass_payload, distance=2)  # pyright: ignore[reportCallIssue, reportArgumentType]
    cutlass_coop.shuffle(cutlass_group, 7, mode="up")  # pyright: ignore[reportCallIssue, reportArgumentType]
    cutlass_coop.shuffle(  # pyright: ignore[reportCallIssue]
        cutlass_group,
        7,
        mode="rotate",
        block_suffix=cutlass_edge,  # pyright: ignore[reportArgumentType]
    )
    cutlass_coop.shuffle(  # pyright: ignore[reportCallIssue]
        cutlass_group,
        cutlass_payload,
        mode="up",
        block_prefix=cutlass_edge,  # pyright: ignore[reportArgumentType]
    )
    cutlass_coop.shuffle(  # pyright: ignore[reportCallIssue]
        cutlass_group,
        cutlass_payload,
        mode="down",  # pyright: ignore[reportArgumentType]
        block_suffix=cutlass_edge,
    )

    numba_group = numba_coop.this_block()
    numba_payload = numba_coop.ThreadData(3, int)
    numba_coop.shuffle(numba_group, numba_payload, mode="rotate")  # pyright: ignore[reportCallIssue, reportArgumentType]
    numba_coop.shuffle(numba_group, numba_payload, distance=2)  # pyright: ignore[reportCallIssue, reportArgumentType]
    numba_coop.shuffle(numba_group, 7, mode="diagonal")  # pyright: ignore[reportCallIssue, reportArgumentType]
    numba_coop.shuffle(  # pyright: ignore[reportCallIssue]
        numba_group,
        numba_payload,
        block_prefix=numba_payload,  # pyright: ignore[reportArgumentType]
    )
