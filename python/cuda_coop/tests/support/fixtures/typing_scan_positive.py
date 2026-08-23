# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Strict positive Pyright fixture for group-first Scan call families."""

# pyright: strict, reportUnnecessaryTypeIgnoreComment=error

from typing import TYPE_CHECKING

from typing_extensions import assert_type


def _maximum(left: int, right: int) -> int:
    return max(left, right)


if TYPE_CHECKING:
    import cuda.coop.cutlass as cutlass_coop
    import cuda.coop.numba_mlir as numba_coop
    from cuda import coop

    common_block = coop.this_block()
    common_warp = coop.this_warp()
    common_payload = coop.ThreadData(2, int)
    assert_type(coop.scan(common_block, common_payload), coop.ThreadDataLike[int])
    assert_type(
        coop.scan(common_block, common_payload, scan_op="+", initial_value=0),
        coop.ThreadDataLike[int],
    )
    assert_type(
        coop.scan(common_block, common_payload, scan_op="max", initial_value=0),
        coop.ThreadDataLike[int],
    )
    assert_type(
        coop.scan(common_block, common_payload, mode="inclusive", scan_op="max"),
        coop.ThreadDataLike[int],
    )
    assert_type(coop.scan(common_warp, 1, initial_value=0), int)
    assert_type(
        coop.scan(common_warp, 1, scan_op="min", initial_value=0),
        int,
    )
    assert_type(coop.scan(common_warp, 1, mode="inclusive", scan_op="min"), int)
    assert_type(
        coop.exclusive_scan(common_block, common_payload, initial_value=0),
        coop.ThreadDataLike[int],
    )
    assert_type(
        coop.exclusive_scan(
            common_block,
            common_payload,
            scan_op="max",
            initial_value=0,
        ),
        coop.ThreadDataLike[int],
    )
    assert_type(
        coop.inclusive_scan(common_block, common_payload, scan_op="max"),
        coop.ThreadDataLike[int],
    )

    cutlass_block = cutlass_coop.this_block()
    cutlass_warp = cutlass_coop.this_warp()
    cutlass_threads = cutlass_warp.group_by(8)
    cutlass_payload = cutlass_coop.ThreadData(2, int)
    cutlass_storage = cutlass_coop.TempStorage()
    assert_type(
        cutlass_coop.scan(
            cutlass_block,
            cutlass_payload,
            scan_op="max",
            initial_value=0,
            algorithm="raking",
            temp_storage=cutlass_storage,
        ),
        cutlass_coop.ThreadData[int],
    )
    assert_type(
        cutlass_coop.scan(
            cutlass_block,
            cutlass_payload,
            mode="inclusive",
            scan_op="max",
        ),
        cutlass_coop.ThreadData[int],
    )
    assert_type(
        cutlass_coop.exclusive_scan(
            cutlass_warp,
            1,
            scan_op="min",
            initial_value=0,
        ),
        int,
    )
    assert_type(
        cutlass_coop.inclusive_scan(
            cutlass_warp,
            1,
            scan_op="min",
        ),
        int,
    )
    assert_type(
        cutlass_coop.scan(
            cutlass_threads,
            1,
            mode="inclusive",
            valid_items=7,
        ),
        int,
    )

    numba_block = numba_coop.this_block()
    numba_warp = numba_coop.this_warp()
    numba_threads = numba_warp.group_by(8)
    numba_payload = numba_coop.ThreadData(2, int)
    numba_storage = numba_coop.TempStorage()
    assert_type(
        numba_coop.scan(
            numba_block,
            numba_payload,
            scan_op="max",
            initial_value=0,
            algorithm=numba_coop.BlockScanAlgorithm.RAKING,
            temp_storage=numba_storage,
        ),
        coop.ThreadDataLike[int],
    )
    assert_type(
        numba_coop.scan(
            numba_block,
            numba_payload,
            scan_op=_maximum,
            initial_value=0,
        ),
        coop.ThreadDataLike[int],
    )
    assert_type(
        numba_coop.scan(
            numba_block,
            numba_payload,
            mode="inclusive",
            scan_op=_maximum,
        ),
        coop.ThreadDataLike[int],
    )
    assert_type(
        numba_coop.exclusive_scan(
            numba_block,
            numba_payload,
            scan_op=_maximum,
            initial_value=0,
        ),
        coop.ThreadDataLike[int],
    )
    assert_type(
        numba_coop.inclusive_scan(
            numba_block,
            numba_payload,
            scan_op=_maximum,
        ),
        coop.ThreadDataLike[int],
    )
    assert_type(
        numba_coop.scan(
            numba_warp,
            1,
            scan_op=_maximum,
            initial_value=0,
        ),
        int,
    )
    assert_type(
        numba_coop.scan(
            numba_threads,
            1,
            mode="inclusive",
            valid_items=7,
        ),
        int,
    )
    assert_type(
        numba_coop.scan(
            numba_warp,
            1,
            mode="inclusive",
            scan_op=_maximum,
        ),
        int,
    )
