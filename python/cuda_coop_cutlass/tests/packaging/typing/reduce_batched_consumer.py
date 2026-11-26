# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Preserve batched reduction payload types across output layouts."""

from __future__ import annotations

import numpy as np
from typing_extensions import assert_type

import cuda.coop.cutlass as coop


def check_cutlass_reduce_batched_surface() -> None:
    values = coop.ThreadData(items_per_thread=2, dtype=np.int32)
    assert_type(
        coop.reduce_batched(coop.this_warp(), values), coop.ThreadData[np.int32]
    )
    assert_type(
        coop.reduce_batched(
            coop.this_warp().group_by(8), values, output_layout="blocked"
        ),
        coop.ThreadData[np.int32],
    )
