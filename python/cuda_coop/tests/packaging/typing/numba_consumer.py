# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Strict consumer of the qualified Numba-CUDA-MLIR surface."""

from __future__ import annotations

from typing import Literal

import numpy as np
from typing_extensions import assert_type

import cuda.coop.numba_mlir as coop


def check_numba_surface(source: object, destination: object) -> None:
    """Exercise Numba declarations through their public package."""

    block = coop.this_block()
    warp = coop.this_warp()
    logical_warp = warp.group_by(8)
    byte_values = coop.ThreadData(1, np.int8)
    values = coop.ThreadData(2, np.uint16, alignas=16)
    ranks = coop.ThreadData(2, np.int32)
    flags = coop.ThreadData(2, np.uint8)
    storage = coop.TempStorage(alignment=16, sharing="shared")

    assert_type(block, coop.ThreadGroup[Literal["block"]])
    assert_type(warp, coop.ThreadGroup[Literal["warp"]])
    assert_type(
        logical_warp,
        coop.ThreadGroup[Literal["threads_within_warp"]],
    )
    assert_type(byte_values, coop.ThreadDataLike[np.int8])
    assert_type(values, coop.ThreadDataLike[np.uint16])
    assert_type(storage, coop.TempStorage)
    portable_storage: coop.TempStorageLike = storage
    assert_type(portable_storage, coop.TempStorageLike)
    assert_type(
        coop.load(
            block,
            source,
            values,
            algorithm="direct",
            temp_storage=storage,
        ),
        coop.ThreadDataLike[np.uint16],
    )
    assert_type(
        coop.load(block, source, byte_values),
        coop.ThreadDataLike[np.int8],
    )
    assert_type(
        coop.load(
            block,
            source,
            byte_values,
            valid_items=1,
            oob_default=0,
        ),
        coop.ThreadDataLike[np.int8],
    )
    assert_type(
        coop.load(block, source, values, algorithm="vectorize"),
        coop.ThreadDataLike[np.uint16],
    )
    assert_type(
        coop.load(
            block,
            source,
            values,
            algorithm="warp_transpose_timesliced",
        ),
        coop.ThreadDataLike[np.uint16],
    )
    assert_type(
        coop.store(
            block,
            destination,
            values,
            algorithm="direct",
            temp_storage=storage,
        ),
        None,
    )
    assert_type(
        coop.exchange(block, values, mode="blocked_to_warp_striped"),
        coop.ThreadDataLike[np.uint16],
    )
    assert_type(
        coop.exchange(
            block,
            values,
            mode="scatter_to_striped_flagged",
            ranks=ranks,
            valid_flags=flags,
        ),
        coop.ThreadDataLike[np.uint16],
    )
    assert_type(
        coop.exchange(
            logical_warp,
            values,
            mode="blocked_to_striped",
        ),
        coop.ThreadDataLike[np.uint16],
    )
    assert_type(
        coop.shuffle(block, values, mode="down"),
        coop.ThreadDataLike[np.uint16],
    )
    assert_type(coop.shuffle(block, np.int32(4), mode="rotate"), np.int32)
    assert_type(coop.store(block, destination, byte_values), None)
    assert_type(
        coop.load(
            warp,
            source,
            values,
            algorithm="transpose",
        ),
        coop.ThreadDataLike[np.uint16],
    )
    assert_type(
        coop.store(
            warp,
            destination,
            values,
            algorithm="striped",
        ),
        None,
    )
    assert_type(
        coop.load(
            logical_warp,
            source,
            values,
            algorithm="transpose",
        ),
        coop.ThreadDataLike[np.uint16],
    )
    assert_type(
        coop.store(
            logical_warp,
            destination,
            values,
            algorithm="striped",
        ),
        None,
    )
