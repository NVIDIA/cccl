# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Strict consumer checks for scan result types."""

from __future__ import annotations

import operator
from typing import Any

import numpy as np
from cutlass import (
    Float32,
    Float64,
    Int32,
    Uint32,
)
from typing_extensions import assert_type

import cuda.coop.cutlass as coop


def check_cutlass_scan_surface(scalar: Uint32) -> None:
    """Check scalar versus payload results and CUTLASS Scan controls.

    All five spellings preserve a ``ThreadData`` or scalar input dtype. Block
    Scan of a register tensor or ``TensorSSA`` input returns
    ``ThreadData[Any]`` because the stubs cannot see its CuTe element type;
    the compiler determines the dtype while tracing.
    """
    block = coop.this_block()
    logical = coop.this_warp().group_by(8)
    values = coop.ThreadData(items_per_thread=2, dtype=np.int32)
    aggregate = coop.ThreadData(items_per_thread=1, dtype=Uint32)
    assert_type(coop.scan(block, values), coop.ThreadData[np.int32])
    assert_type(coop.exclusive_scan(block, values), coop.ThreadData[np.int32])
    assert_type(coop.inclusive_scan(block, values), coop.ThreadData[np.int32])
    assert_type(coop.exclusive_sum(block, values), coop.ThreadData[np.int32])
    assert_type(coop.inclusive_sum(block, values), coop.ThreadData[np.int32])
    assert_type(
        coop.scan(block, values, scan_op="max", initial_value=np.int32(-8)),
        coop.ThreadData[np.int32],
    )
    assert_type(
        coop.scan(block, values, scan_op=np.add), coop.ThreadData[np.int32]
    )
    assert_type(
        coop.exclusive_scan(
            block, scalar, scan_op=operator.mul, initial_value=1
        ),
        Uint32,
    )
    assert_type(
        coop.inclusive_sum(block, values, algorithm="raking_memoize"),
        coop.ThreadData[np.int32],
    )
    storage = coop.TempStorage(sharing="exclusive")
    assert_type(
        coop.scan(block, values, algorithm="warp_scans", temp_storage=storage),
        coop.ThreadData[np.int32],
    )
    assert_type(
        coop.scan(logical, scalar, valid_items=7, aggregate_output=aggregate),
        Uint32,
    )
    assert_type(
        coop.exclusive_scan(
            logical, scalar, valid_items=7, aggregate_output=aggregate
        ),
        Uint32,
    )
    assert_type(
        coop.inclusive_scan(
            logical, scalar, valid_items=7, aggregate_output=aggregate
        ),
        Uint32,
    )
    assert_type(
        coop.exclusive_sum(
            logical, scalar, valid_items=7, aggregate_output=aggregate
        ),
        Uint32,
    )
    assert_type(
        coop.inclusive_sum(
            logical, scalar, valid_items=7, aggregate_output=aggregate
        ),
        Uint32,
    )
    assert_type(
        coop.inclusive_sum(block, values.to_register_tensor()),
        coop.ThreadData[Any],
    )
    assert_type(
        coop.exclusive_sum(block, values.to_tensor_ssa()), coop.ThreadData[Any]
    )


def check_cutlass_scan_seeds(integer_seed: int, floating_seed: float) -> None:
    """Accept matching initial-value types without changing the result dtype.

    NumPy and CuTe spellings can describe the same numeric dtype. Python int
    and float seeds take the input's dtype. These declarations check accepted
    type combinations; representability of a particular literal is a compiler
    validation, not a mypy assertion here.
    """
    block = coop.this_block()
    warp = coop.this_warp()
    numpy_values = coop.ThreadData(items_per_thread=2, dtype=np.int32)
    cute_values = coop.ThreadData(items_per_thread=2, dtype=Float32)
    assert_type(
        coop.exclusive_scan(block, Int32(4), initial_value=Int32(0)), Int32
    )
    assert_type(
        coop.scan(warp, Float32(4), mode="exclusive", initial_value=Float32(0)),
        Float32,
    )
    assert_type(
        coop.exclusive_scan(block, Int32(4), initial_value=np.int32(0)), Int32
    )
    assert_type(
        coop.scan(
            warp, Float32(4), mode="exclusive", initial_value=np.float32(0)
        ),
        Float32,
    )
    assert_type(
        coop.exclusive_scan(warp, Float64(4), initial_value=np.float64(0)),
        Float64,
    )
    assert_type(
        coop.scan(block, np.int32(4), mode="exclusive", initial_value=Int32(0)),
        np.int32,
    )
    assert_type(
        coop.exclusive_scan(warp, np.float32(4), initial_value=Float32(0)),
        np.float32,
    )
    assert_type(
        coop.scan(
            warp, np.float64(4), mode="exclusive", initial_value=Float64(0)
        ),
        np.float64,
    )
    assert_type(
        coop.scan(
            block, numpy_values, mode="exclusive", initial_value=Int32(0)
        ),
        coop.ThreadData[np.int32],
    )
    assert_type(
        coop.exclusive_scan(block, cute_values, initial_value=np.float32(0)),
        coop.ThreadData[Float32],
    )
    assert_type(
        coop.exclusive_scan(block, Int32(4), initial_value=integer_seed), Int32
    )
    assert_type(
        coop.scan(
            warp, Float32(4), mode="exclusive", initial_value=floating_seed
        ),
        Float32,
    )
