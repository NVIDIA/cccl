# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Strict consumer checks for exchange result types."""

from __future__ import annotations

from typing import Any

import numpy as np
from cutlass import (
    Int16,
    Uint64,
)
from typing_extensions import assert_type

import cuda.coop.cutlass as coop


def check_cutlass_exchange_surface() -> None:
    """Keep the value dtype separate from scatter rank and flag dtypes.

    Layout changes and scatters keep the value dtype. Exchange of a register
    tensor or ``TensorSSA`` value returns ``ThreadData[Any]`` because the
    stubs cannot see its CuTe element type; the compiler resolves the dtype
    while tracing.
    """
    block = coop.this_block()
    values = coop.ThreadData(items_per_thread=3, dtype=np.float32)
    ranks = coop.ThreadData(items_per_thread=3, dtype=Int16)
    flags = coop.ThreadData(items_per_thread=3, dtype=Uint64)
    for mode in (
        "striped_to_blocked",
        "blocked_to_striped",
        "warp_striped_to_blocked",
        "blocked_to_warp_striped",
    ):
        assert_type(
            coop.exchange(block, values, mode=mode), coop.ThreadData[np.float32]
        )
        assert_type(
            coop.exchange(block, values, mode=mode, warp_time_slicing=True),
            coop.ThreadData[np.float32],
        )
    for scatter_mode in ("scatter_to_blocked", "scatter_to_striped"):
        assert_type(
            coop.exchange(
                block,
                values,
                mode=scatter_mode,
                ranks=ranks,
                warp_time_slicing=True,
            ),
            coop.ThreadData[np.float32],
        )
    assert_type(
        coop.exchange(
            block, values, mode="scatter_to_striped_guarded", ranks=ranks
        ),
        coop.ThreadData[np.float32],
    )
    assert_type(
        coop.exchange(
            block,
            values,
            mode="scatter_to_striped_flagged",
            ranks=ranks,
            valid_flags=flags,
        ),
        coop.ThreadData[np.float32],
    )
    assert_type(
        coop.exchange(coop.this_warp(), values), coop.ThreadData[np.float32]
    )
    for width in (1, 2, 4, 8, 16, 32):
        logical = coop.this_warp().group_by(width)
        assert_type(coop.exchange(logical, values), coop.ThreadData[np.float32])
    assert_type(
        coop.exchange(block, values.to_tensor_ssa()), coop.ThreadData[Any]
    )
    assert_type(
        coop.exchange(
            block,
            values.to_register_tensor(),
            mode="scatter_to_striped_flagged",
            ranks=ranks.to_tensor_ssa(),
            valid_flags=flags.to_register_tensor(),
        ),
        coop.ThreadData[Any],
    )
    assert_type(
        coop.exchange(
            block,
            values,
            mode="scatter_to_blocked",
            ranks=coop.ThreadData(items_per_thread=3, dtype=np.int8),
        ),
        coop.ThreadData[np.float32],
    )
