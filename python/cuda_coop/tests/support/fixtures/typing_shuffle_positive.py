# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Strict positive typing fixture for group-first Shuffle."""

# pyright: strict, reportUnnecessaryTypeIgnoreComment=error

from typing import TYPE_CHECKING

from typing_extensions import assert_type

if TYPE_CHECKING:
    import cuda.coop.cutlass as cutlass_coop
    import cuda.coop.numba_mlir as numba_coop
    from cuda import coop

    common_group = coop.this_block()
    common_payload = coop.ThreadData(3, int)
    assert_type(
        coop.shuffle(common_group, common_payload),
        coop.ThreadDataLike[int],
    )
    assert_type(
        coop.shuffle(common_group, common_payload, mode="up", distance=1),
        coop.ThreadDataLike[int],
    )

    cutlass_group = cutlass_coop.this_block()
    cutlass_payload = cutlass_coop.ThreadData(3, int)
    cutlass_prefix = cutlass_coop.ThreadData(1, int)
    cutlass_suffix = cutlass_coop.ThreadData(1, int)
    assert_type(
        cutlass_coop.shuffle(
            cutlass_group,
            cutlass_payload,
            mode="down",
            block_prefix=cutlass_prefix,
        ),
        cutlass_coop.ThreadData[int],
    )
    assert_type(
        cutlass_coop.shuffle(
            cutlass_group,
            cutlass_payload,
            mode="up",
            block_suffix=cutlass_suffix,
        ),
        cutlass_coop.ThreadData[int],
    )
    assert_type(
        cutlass_coop.shuffle(cutlass_group, 7, mode="offset", distance=-2),
        int,
    )
    assert_type(
        cutlass_coop.shuffle(cutlass_group, 7, mode="rotate", distance=2),
        int,
    )

    numba_group = numba_coop.this_block()
    numba_payload = numba_coop.ThreadData(3, int)
    assert_type(
        numba_coop.shuffle(numba_group, numba_payload, mode="down"),
        coop.ThreadDataLike[int],
    )
    assert_type(
        numba_coop.shuffle(numba_group, numba_payload, mode="up", distance=1),
        coop.ThreadDataLike[int],
    )
    assert_type(
        numba_coop.shuffle(numba_group, 7, mode="offset", distance=-2),
        int,
    )
    assert_type(
        numba_coop.shuffle(numba_group, 7, mode="rotate", distance=2),
        int,
    )
    assert_type(numba_coop.shuffle(numba_group, 7), int)
    assert_type(numba_coop.shuffle(numba_group, 7, mode="up", distance=2), int)
    assert_type(numba_coop.shuffle(numba_group, 7, mode="down", distance=2), int)
