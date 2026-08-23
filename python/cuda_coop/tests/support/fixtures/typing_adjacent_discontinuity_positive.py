# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Strict positive typing fixture for group-first segmentation primitives."""

# pyright: strict, reportUnnecessaryTypeIgnoreComment=error

from typing import TYPE_CHECKING

from typing_extensions import assert_type


def _subtract(left: int, right: int) -> int:
    return left - right


def _different(left: int, right: int) -> bool:
    return left != right


if TYPE_CHECKING:
    import cuda.coop.cutlass as cutlass_coop
    import cuda.coop.numba_mlir as numba_coop
    from cuda import coop

    common_group = coop.this_block()
    common_payload = coop.ThreadData(2, int)
    common_scratch = coop.TempStorage()
    assert_type(
        coop.adjacent_difference(
            common_group,
            common_payload,
            direction="left",
            valid_items=63,
            tile_predecessor_item=0,
            temp_storage=common_scratch,
        ),
        coop.ThreadDataLike[int],
    )
    assert_type(
        coop.adjacent_difference(
            common_group,
            common_payload,
            direction="right",
            tile_successor_item=0,
        ),
        coop.ThreadDataLike[int],
    )
    assert_type(
        coop.adjacent_difference(
            common_group,
            common_payload,
            direction="right",
            valid_items=63,
        ),
        coop.ThreadDataLike[int],
    )
    assert_type(
        coop.discontinuity(
            common_group,
            common_payload,
            mode="heads",
            tile_predecessor_item=0,
            temp_storage=common_scratch,
        ),
        coop.ThreadDataLike[int],
    )
    assert_type(
        coop.discontinuity(
            common_group,
            common_payload,
            mode="tails",
            tile_successor_item=0,
        ),
        coop.ThreadDataLike[int],
    )

    cutlass_group = cutlass_coop.this_block()
    cutlass_payload = cutlass_coop.ThreadData(2, int)
    cutlass_scratch = cutlass_coop.TempStorage()
    assert_type(
        cutlass_coop.adjacent_difference(
            cutlass_group,
            cutlass_payload,
            direction="right",
            tile_successor_item=0,
            temp_storage=cutlass_scratch,
            difference_op="subtract",
        ),
        cutlass_coop.ThreadData[int],
    )
    assert_type(
        cutlass_coop.adjacent_difference(
            cutlass_group,
            cutlass_payload,
            direction="right",
            valid_items=63,
        ),
        cutlass_coop.ThreadData[int],
    )
    assert_type(
        cutlass_coop.discontinuity(
            cutlass_group,
            cutlass_payload,
            mode="heads",
            tile_predecessor_item=0,
            temp_storage=cutlass_scratch,
            flag_op="not_equal",
        ),
        cutlass_coop.ThreadData[int],
    )
    cutlass_head_flags, cutlass_tail_flags = cutlass_coop.discontinuity(
        cutlass_group,
        cutlass_payload,
        mode="heads_and_tails",
        tile_predecessor_item=0,
        tile_successor_item=0,
        flag_op="!=",
    )
    assert_type(cutlass_head_flags, cutlass_coop.ThreadData[int])
    assert_type(cutlass_tail_flags, cutlass_coop.ThreadData[int])

    cutlass_scalar = 7
    assert_type(
        cutlass_coop.adjacent_difference(
            cutlass_group,
            cutlass_scalar,
            direction="left",
            tile_predecessor_item=0,
            difference_op="subtract",
        ),
        int,
    )
    assert_type(
        cutlass_coop.discontinuity(
            cutlass_group,
            cutlass_scalar,
            mode="tails",
            tile_successor_item=0,
            flag_op="not_equal",
        ),
        int,
    )
    cutlass_scalar_head, cutlass_scalar_tail = cutlass_coop.discontinuity(
        cutlass_group,
        cutlass_scalar,
        mode="heads_and_tails",
        tile_predecessor_item=0,
        tile_successor_item=0,
        flag_op="!=",
    )
    assert_type(cutlass_scalar_head, int)
    assert_type(cutlass_scalar_tail, int)

    numba_group = numba_coop.this_block()
    numba_payload = numba_coop.ThreadData(2, int)
    numba_scratch = numba_coop.TempStorage()
    assert_type(
        numba_coop.adjacent_difference(
            numba_group,
            numba_payload,
            direction="left",
            valid_items=63,
            tile_predecessor_item=0,
            temp_storage=numba_scratch,
            difference_op=_subtract,
        ),
        coop.ThreadDataLike[int],
    )
    assert_type(
        numba_coop.adjacent_difference(
            numba_group,
            numba_payload,
            direction="right",
            valid_items=63,
        ),
        coop.ThreadDataLike[int],
    )
    assert_type(
        numba_coop.discontinuity(
            numba_group,
            numba_payload,
            mode="tails",
            tile_successor_item=0,
            temp_storage=numba_scratch,
            flag_op=_different,
        ),
        coop.ThreadDataLike[int],
    )
    numba_head_flags, numba_tail_flags = numba_coop.discontinuity(
        numba_group,
        numba_payload,
        mode="heads_and_tails",
        tile_predecessor_item=0,
        tile_successor_item=0,
        flag_op=_different,
    )
    assert_type(numba_head_flags, coop.ThreadDataLike[int])
    assert_type(numba_tail_flags, coop.ThreadDataLike[int])

    numba_scalar = 7
    assert_type(
        numba_coop.adjacent_difference(
            numba_group,
            numba_scalar,
            direction="left",
            tile_predecessor_item=0,
            difference_op=_subtract,
        ),
        int,
    )
    assert_type(
        numba_coop.discontinuity(
            numba_group,
            numba_scalar,
            mode="tails",
            tile_successor_item=0,
            flag_op=_different,
        ),
        int,
    )
    numba_scalar_head, numba_scalar_tail = numba_coop.discontinuity(
        numba_group,
        numba_scalar,
        mode="heads_and_tails",
        tile_predecessor_item=0,
        tile_successor_item=0,
        flag_op=_different,
    )
    assert_type(numba_scalar_head, int)
    assert_type(numba_scalar_tail, int)
