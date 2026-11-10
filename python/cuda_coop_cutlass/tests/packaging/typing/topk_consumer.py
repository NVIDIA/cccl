# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Strict consumer checks for topk result types."""

from __future__ import annotations

from typing import Any

import numpy as np
from cutlass import (
    Int16,
    Int64,
    Uint32,
)
from typing_extensions import assert_type

import cuda.coop.cutlass as coop


class _ReadOnlyKeys:
    """Supply readable keys for static typing without mutable item access."""

    items_per_thread: int = 2
    dtype: object | None = np.int32

    def __len__(self) -> int:
        return self.items_per_thread

    def __getitem__(self, index: int, /) -> np.int32:
        return np.int32(index)


def check_cutlass_topk_surface() -> None:
    """Keep key and value dtypes through each minimum and maximum overload.

    Counts affect which result positions are defined, not their static type.
    Read-only payloads need no setter, and mixed register/payload pairs lose
    element detail only for the register operand whose annotation lacks it.
    """
    block = coop.this_block()
    keys = coop.ThreadData(items_per_thread=3, dtype=np.float32)
    values = coop.ThreadData(items_per_thread=3, dtype=Int16)
    storage = coop.TempStorage(alignment=32)
    assert_type(
        coop.topk_min_keys(block, keys, k=7), coop.ThreadData[np.float32]
    )
    assert_type(
        coop.topk_max_keys(block, keys, k=Int64(7), valid_items=Uint32(63)),
        coop.ThreadData[np.float32],
    )
    assert_type(
        coop.topk_min_pairs(block, keys, values, k=7, temp_storage=storage),
        tuple[coop.ThreadData[np.float32], coop.ThreadData[Int16]],
    )
    assert_type(
        coop.topk_max_pairs(
            block,
            keys,
            values,
            k=np.uint16(7),
            valid_items=31,
            temp_storage=storage,
        ),
        tuple[coop.ThreadData[np.float32], coop.ThreadData[Int16]],
    )
    assert_type(
        coop.topk_min_keys(block, _ReadOnlyKeys(), k=0),
        coop.ThreadData[np.int32],
    )
    assert_type(
        coop.topk_max_pairs(block, _ReadOnlyKeys(), _ReadOnlyKeys(), k=2),
        tuple[coop.ThreadData[np.int32], coop.ThreadData[np.int32]],
    )
    assert_type(
        coop.topk_min_keys(block, keys.to_register_tensor(), k=7),
        coop.ThreadData[Any],
    )
    assert_type(
        coop.topk_max_keys(block, keys.to_tensor_ssa(), k=7),
        coop.ThreadData[Any],
    )
    assert_type(
        coop.topk_min_pairs(block, keys.to_tensor_ssa(), values, k=7),
        tuple[coop.ThreadData[Any], coop.ThreadData[Int16]],
    )
    assert_type(
        coop.topk_max_pairs(block, keys, values.to_register_tensor(), k=7),
        tuple[coop.ThreadData[np.float32], coop.ThreadData[Any]],
    )
    assert_type(
        coop.topk_min_pairs(
            block, keys.to_register_tensor(), values.to_tensor_ssa(), k=7
        ),
        tuple[coop.ThreadData[Any], coop.ThreadData[Any]],
    )
