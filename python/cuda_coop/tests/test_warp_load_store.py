# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np
import pytest

from cuda.coop._core import ArgumentBinding
from cuda.coop._core.warp import (
    make_warp_load_specialization,
    make_warp_load_store_semantics,
    make_warp_store_specialization,
)


@pytest.mark.parametrize(
    "threads_in_warp", [True, 0, 3, 64, 8.0, np.int64(3), np.bool_(True)]
)
def test_warp_load_store_rejects_unsupported_widths(threads_in_warp):
    with pytest.raises(ValueError, match="threads_in_warp in"):
        make_warp_load_specialization(
            dtype="i32",
            items_per_thread=2,
            threads_in_warp=threads_in_warp,
            algorithm="direct",
        )


@pytest.mark.parametrize(
    "make_specialization",
    [make_warp_load_specialization, make_warp_store_specialization],
)
def test_numpy_integer_controls_share_the_same_specialization(
    make_specialization,
):
    def build(integer):
        return make_specialization(
            dtype="i32",
            items_per_thread=integer(2),
            threads_in_warp=integer(8),
            algorithm="direct",
        )

    assert build(int).semantic_key == build(np.int64).semantic_key


@pytest.mark.parametrize(
    "make_specialization",
    [make_warp_load_specialization, make_warp_store_specialization],
)
@pytest.mark.parametrize("valid_items", [-1, 17])
def test_warp_load_store_rejects_static_valid_items_outside_tile(
    make_specialization, valid_items
):
    with pytest.raises(ValueError, match="warp tile size"):
        make_specialization(
            dtype="i32",
            items_per_thread=2,
            threads_in_warp=8,
            algorithm="direct",
            valid_items=ArgumentBinding.static(valid_items),
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"algorithm": "warp_transpose"}, "unsupported WarpLoad algorithm"),
        (
            {"items_per_thread": True},
            "items_per_thread must be a positive integer",
        ),
        (
            {"items_per_thread": 2.0},
            "items_per_thread must be a positive integer",
        ),
        (
            {"kind": "store", "valid_items": True, "oob_default": True},
            "only valid for WarpLoad",
        ),
        ({"oob_default": True}, "requires a valid_items"),
    ],
)
def test_warp_load_store_rejects_invalid_options(overrides, message):
    options = {
        "kind": "load",
        "dtype": "i32",
        "items_per_thread": 1,
        "algorithm": "direct",
    }
    options.update(overrides)
    with pytest.raises(ValueError, match=message):
        make_warp_load_store_semantics(**options)
