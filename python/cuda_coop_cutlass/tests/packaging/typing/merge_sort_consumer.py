# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Strict consumer checks for merge sort result types."""

from __future__ import annotations

from typing import Any

import numpy as np
from cutlass import (
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


def check_cutlass_merge_sort_surface() -> None:
    """Keep key and value result types independent across sort overloads.

    Read-only inputs need no setter because sorting returns new payloads.
    Register inputs deliberately infer ``Any`` where CuTe annotations do not
    expose an element type; typed ``ThreadData`` operands retain their dtype.
    """
    block = coop.this_block()
    warp = coop.this_warp()
    logical = warp.group_by(8)
    keys = coop.ThreadData(items_per_thread=3, dtype=np.int32)
    values = coop.ThreadData(items_per_thread=3, dtype=np.float64)
    storage = coop.TempStorage(alignment=32)
    assert_type(coop.merge_sort_keys(block, keys), coop.ThreadData[np.int32])
    assert_type(
        coop.merge_sort_pairs(block, keys, values, temp_storage=storage),
        tuple[coop.ThreadData[np.int32], coop.ThreadData[np.float64]],
    )
    assert_type(
        coop.merge_sort_keys(
            block,
            keys,
            valid_items=Uint32(63),
            oob_default=1000,
            temp_storage=storage,
        ),
        coop.ThreadData[np.int32],
    )
    assert_type(
        coop.merge_sort_pairs(
            warp,
            keys,
            values,
            descending=True,
            valid_items=np.int64(31),
            oob_default=-1000,
        ),
        tuple[coop.ThreadData[np.int32], coop.ThreadData[np.float64]],
    )
    assert_type(
        coop.merge_sort_keys(
            logical, keys, valid_items=23, oob_default=np.int32(1000)
        ),
        coop.ThreadData[np.int32],
    )
    assert_type(
        coop.merge_sort_keys(warp, _ReadOnlyKeys()), coop.ThreadData[np.int32]
    )
    assert_type(
        coop.merge_sort_pairs(logical, _ReadOnlyKeys(), _ReadOnlyKeys()),
        tuple[coop.ThreadData[np.int32], coop.ThreadData[np.int32]],
    )
    assert_type(
        coop.merge_sort_keys(block, keys.to_register_tensor()),
        coop.ThreadData[Any],
    )
    assert_type(
        coop.merge_sort_pairs(warp, keys.to_tensor_ssa(), values),
        tuple[coop.ThreadData[Any], coop.ThreadData[np.float64]],
    )
    assert_type(
        coop.merge_sort_pairs(logical, keys, values.to_register_tensor()),
        tuple[coop.ThreadData[np.int32], coop.ThreadData[Any]],
    )
    assert_type(
        coop.merge_sort_pairs(
            block,
            keys.to_register_tensor(),
            values.to_tensor_ssa(),
            valid_items=95,
            oob_default=np.int32(1000),
        ),
        tuple[coop.ThreadData[Any], coop.ThreadData[Any]],
    )
