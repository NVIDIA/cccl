# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Strict consumer checks for radix result types."""

from __future__ import annotations

from typing import Any

import numpy as np
from cutlass import (
    Float32,
    Float64,
    Int16,
    Int32,
    Int64,
    Uint32,
    Uint64,
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


def check_cutlass_radix_surface(scalar: Float32, value: Int16) -> None:
    """Separate type-preserving Sort outputs from signed Int32 Rank outputs.

    Scalar and payload forms retain their shape. Register inputs can lose
    static element detail for Sort, but Rank has a fixed result dtype. Bin
    prefixes are a separate writable output with their own per-thread extent.
    """
    block = coop.this_block()
    keys = coop.ThreadData(items_per_thread=3, dtype=np.int32)
    values = coop.ThreadData(items_per_thread=3, dtype=np.float64)
    prefixes = coop.ThreadData(items_per_thread=1, dtype=Int32)
    storage = coop.TempStorage(alignment=32)
    assert_type(coop.radix_sort_keys(block, keys), coop.ThreadData[np.int32])
    assert_type(
        coop.radix_sort_pairs(
            block,
            keys,
            values,
            begin_bit=Int64(3),
            end_bit=Uint32(8),
            descending=True,
            temp_storage=storage,
            blocked_to_striped=True,
        ),
        tuple[coop.ThreadData[np.int32], coop.ThreadData[np.float64]],
    )
    assert_type(coop.radix_sort_keys(block, scalar), Float32)
    assert_type(
        coop.radix_sort_pairs(block, scalar, value), tuple[Float32, Int16]
    )
    assert_type(
        coop.radix_sort_keys(
            block, coop.ThreadData(items_per_thread=2, dtype=Float64)
        ),
        coop.ThreadData[Float64],
    )
    assert_type(
        coop.radix_sort_pairs(block, Uint64(1), np.uint8(2)),
        tuple[Uint64, np.uint8],
    )
    assert_type(
        coop.radix_sort_keys(block, _ReadOnlyKeys()), coop.ThreadData[np.int32]
    )
    assert_type(
        coop.radix_sort_pairs(block, _ReadOnlyKeys(), _ReadOnlyKeys()),
        tuple[coop.ThreadData[np.int32], coop.ThreadData[np.int32]],
    )
    assert_type(
        coop.radix_sort_keys(block, keys.to_tensor_ssa()), coop.ThreadData[Any]
    )
    assert_type(
        coop.radix_sort_pairs(block, keys.to_register_tensor(), values),
        tuple[coop.ThreadData[Any], coop.ThreadData[np.float64]],
    )
    assert_type(
        coop.radix_sort_pairs(block, keys, values.to_tensor_ssa()),
        tuple[coop.ThreadData[np.int32], coop.ThreadData[Any]],
    )
    assert_type(
        coop.radix_sort_pairs(
            block, keys.to_tensor_ssa(), values.to_register_tensor()
        ),
        tuple[coop.ThreadData[Any], coop.ThreadData[Any]],
    )
    assert_type(
        coop.radix_rank_keys(
            block,
            keys,
            begin_bit=4,
            radix_bits=4,
            descending=True,
            exclusive_digit_prefix=prefixes,
        ),
        coop.ThreadData[Int32],
    )
    assert_type(
        coop.radix_rank_keys(block, _ReadOnlyKeys()), coop.ThreadData[Int32]
    )
    assert_type(
        coop.radix_rank_keys(block, keys.to_register_tensor()),
        coop.ThreadData[Int32],
    )
    assert_type(
        coop.radix_rank_keys(block, keys.to_tensor_ssa()),
        coop.ThreadData[Int32],
    )
