# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Strict consumer of the portable Block Load and Store surface."""

from __future__ import annotations

from typing import Literal

import numpy as np
from typing_extensions import assert_type

import cuda.coop as coop


def check_portable_surface(source: object, destination: object) -> None:
    """Exercise public declarations without importing package internals."""

    block = coop.this_block()
    values = coop.ThreadData(2, np.int16)
    storage = coop.TempStorage(sharing="shared")

    assert_type(block, coop.ThreadGroup[Literal["block"]])
    assert_type(values, coop.ThreadDataLike[np.int16])
    assert_type(storage, coop.TempStorageLike)
    assert_type(
        coop.load(
            block,
            source,
            values,
            algorithm="direct",
            valid_items=15,
            oob_default=np.int16(0),
            offset=1,
            temp_storage=storage,
        ),
        coop.ThreadDataLike[np.int16],
    )
    assert_type(
        coop.load(block, source, values, algorithm="warp_transpose_timesliced"),
        coop.ThreadDataLike[np.int16],
    )
    assert_type(
        coop.store(
            block,
            destination,
            values,
            algorithm="direct",
            valid_items=15,
            offset=1,
            temp_storage=storage,
        ),
        None,
    )
