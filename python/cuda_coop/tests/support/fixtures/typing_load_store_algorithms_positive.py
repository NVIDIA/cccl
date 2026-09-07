# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Strict positive typing fixture for group-specific Load/Store algorithms."""

# pyright: strict, reportUnnecessaryTypeIgnoreComment=error

from typing import TYPE_CHECKING

import numpy as np
from typing_extensions import assert_type

if TYPE_CHECKING:
    import cuda.coop.cutlass as cutlass_coop
    import cuda.coop.numba_mlir as numba_coop
    from cuda import coop

    common_block = coop.this_block()
    common_warp = coop.this_warp()
    common_payload = coop.ThreadData(2, int)

    assert_type(
        coop.load(
            common_block,
            object(),
            common_payload,
            algorithm="warp_transpose",
        ),
        coop.ThreadDataLike[int],
    )
    coop.store(
        common_block,
        object(),
        common_payload,
        algorithm="warp_transpose_timesliced",
    )
    assert_type(
        coop.load(common_warp, object(), common_payload, algorithm="transpose"),
        coop.ThreadDataLike[int],
    )
    coop.store(common_warp, object(), common_payload, algorithm="vectorize")
    assert_type(
        coop.load(
            common_block,
            object(),
            common_payload,
            valid_items=np.int32(1),
            oob_default=0,
            offset=np.int64(4),
        ),
        coop.ThreadDataLike[int],
    )
    coop.store(
        common_warp,
        object(),
        common_payload,
        valid_items=np.int32(1),
        offset=np.int64(4),
    )

    cutlass_block = cutlass_coop.this_block()
    cutlass_warp = cutlass_coop.this_warp()
    cutlass_payload = cutlass_coop.ThreadData(2, int)
    assert_type(
        cutlass_coop.load(
            cutlass_block,
            object(),
            cutlass_payload,
            algorithm="warp_transpose_timesliced",
        ),
        cutlass_coop.ThreadData[int],
    )
    cutlass_coop.store(
        cutlass_block,
        object(),
        cutlass_payload,
        algorithm="warp_transpose",
    )
    assert_type(
        cutlass_coop.load(
            cutlass_warp,
            object(),
            cutlass_payload,
            algorithm="transpose",
        ),
        cutlass_coop.ThreadData[int],
    )
    cutlass_coop.store(
        cutlass_warp,
        object(),
        cutlass_payload,
        algorithm="vectorize",
    )
    assert_type(
        cutlass_coop.load(
            cutlass_block,
            object(),
            cutlass_payload,
            valid_items=np.int32(1),
            oob_default=0,
            offset=np.int64(4),
        ),
        cutlass_coop.ThreadData[int],
    )

    numba_block = numba_coop.this_block()
    numba_warp = numba_coop.this_warp()
    numba_payload = numba_coop.ThreadData(2, int)
    assert_type(
        numba_coop.load(
            numba_block,
            object(),
            numba_payload,
            algorithm=numba_coop.BlockLoadAlgorithm.WARP_TRANSPOSE,
        ),
        coop.ThreadDataLike[int],
    )
    numba_coop.store(
        numba_block,
        object(),
        numba_payload,
        algorithm=numba_coop.BlockStoreAlgorithm.WARP_TRANSPOSE_TIMESLICED,
    )
    assert_type(
        numba_coop.load(
            numba_warp,
            object(),
            numba_payload,
            algorithm=numba_coop.WarpLoadAlgorithm.TRANSPOSE,
        ),
        coop.ThreadDataLike[int],
    )
    numba_coop.store(
        numba_warp,
        object(),
        numba_payload,
        algorithm=numba_coop.WarpStoreAlgorithm.VECTORIZE,
    )
    assert_type(
        numba_coop.load(
            numba_warp,
            object(),
            numba_payload,
            valid_items=np.int32(1),
            oob_default=0,
            offset=np.int64(4),
        ),
        coop.ThreadDataLike[int],
    )
    numba_coop.store(
        numba_block,
        object(),
        numba_payload,
        valid_items=np.int32(1),
        offset=np.int64(4),
    )
