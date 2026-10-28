# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Strict consumer checks for histogram result types."""

from __future__ import annotations

import numpy as np
from cutlass import (
    Int32,
    Uint64,
)
from typing_extensions import assert_type

import cuda.coop.cutlass as coop


def check_cutlass_histogram() -> None:
    """Infer counters from their selector independently of the sample dtype.

    NumPy and CuTe selectors retain their annotation type. The default and
    Python ``int`` both select CuTe Int32, even for register-payload samples.
    """
    block = coop.this_block()
    samples = coop.ThreadData(items_per_thread=3, dtype=np.uint8)
    storage = coop.TempStorage(alignment=16)
    assert_type(
        coop.histogram(
            block, samples, bins=65, bins_per_thread=2, temp_storage=storage
        ),
        coop.ThreadData[Int32],
    )
    assert_type(
        coop.histogram(
            block, samples, bins=65, bins_per_thread=2, counter_dtype=np.uint64
        ),
        coop.ThreadData[np.uint64],
    )
    assert_type(
        coop.histogram(
            block,
            samples,
            bins=65,
            bins_per_thread=2,
            counter_dtype=np.dtype(np.uint64),
        ),
        coop.ThreadData[np.uint64],
    )
    assert_type(
        coop.histogram(
            block, samples, bins=65, bins_per_thread=2, counter_dtype=Uint64
        ),
        coop.ThreadData[Uint64],
    )
    assert_type(
        coop.histogram(
            block,
            samples.to_register_tensor(),
            bins=65,
            bins_per_thread=2,
            counter_dtype=int,
        ),
        coop.ThreadData[Int32],
    )
    assert_type(
        coop.histogram(
            block,
            samples.to_tensor_ssa(),
            bins=65,
            bins_per_thread=2,
            algorithm="sort",
        ),
        coop.ThreadData[Int32],
    )
