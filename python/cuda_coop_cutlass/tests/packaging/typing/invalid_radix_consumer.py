# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Reject unsupported radix calls through stubs."""

from __future__ import annotations

import numpy as np
from cutlass import Float32, Int32

import cuda.coop.cutlass as coop

block = coop.this_block()
warp = coop.this_warp()
logical = warp.group_by(8)
mapped = block.group_by(2)
scalar = Int32(1)
values = coop.ThreadData(items_per_thread=2, dtype=np.int32)


class _ReadOnlyPrefix:
    """Omit mutable item access so the checker rejects this output buffer."""

    items_per_thread: int = 1
    dtype: object | None = Int32

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int, /) -> Int32:
        return Int32(index)


coop.radix_sort_keys(warp, values)  # expected-error: [arg-type]
coop.radix_rank_keys(logical, values)  # expected-error: [arg-type]
coop.radix_sort_pairs(  # expected-error: [call-overload]
    block, scalar, values
)
coop.radix_sort_pairs(  # expected-error: [call-overload]
    block, values, scalar
)
coop.radix_sort_keys(  # expected-error: [type-var]
    block, coop.ThreadData(items_per_thread=2, dtype=np.int16)
)
coop.radix_sort_keys(  # expected-error: [type-var]
    block, coop.ThreadData(items_per_thread=2, dtype=np.complex64)
)
coop.radix_rank_keys(  # expected-error: [type-var]
    block, coop.ThreadData(items_per_thread=2, dtype=np.float32)
)
coop.radix_rank_keys(  # expected-error: [call-overload]
    block, Float32(1)
)
coop.radix_sort_keys(  # expected-error: [call-overload]
    block, values, valid_items=3
)
coop.radix_sort_pairs(  # expected-error: [call-overload]
    block, values, values, algorithm="radix"
)
coop.radix_sort_keys(
    block,
    values,
    begin_bit=np.uint64(0),  # expected-error: [arg-type]
)
coop.radix_rank_keys(  # expected-error: [call-overload]
    block, values, begin_bit=Int32(1)
)
coop.radix_rank_keys(  # expected-error: [call-overload]
    block, values, temp_storage=coop.TempStorage()
)
coop.radix_rank_keys(  # expected-error: [call-overload]
    block, values, blocked_to_striped=True
)
coop.radix_rank_keys(
    block,
    values,
    exclusive_digit_prefix=(
        coop.ThreadData(  # expected-error: [arg-type]
            items_per_thread=1, dtype=np.uint32
        )
    ),
)
coop.radix_rank_keys(  # expected-error: [call-overload]
    block, values, exclusive_digit_prefix=scalar
)
coop.radix_sort_keys(  # expected-error: [call-overload]
    group=block, keys=values
)
coop.radix_rank_keys(  # expected-error: [call-overload]
    block, values, 0, 4
)
coop.radix_rank_keys(  # expected-error: [call-overload]
    block, values, exclusive_digit_prefix=_ReadOnlyPrefix()
)
