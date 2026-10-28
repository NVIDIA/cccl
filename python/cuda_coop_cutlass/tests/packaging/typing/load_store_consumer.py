# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Strict consumer checks for load store result types."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from cutlass import (
    Float64,
    Int32,
    Uint32,
    Uint64,
)
from typing_extensions import assert_type

import cuda.coop.cutlass as coop


def check_cutlass_block_surface(source: object, destination: object) -> None:
    """Return payloads for one or more items and reuse block scratch."""
    block = coop.this_block()
    scratch = coop.TempStorage(auto_sync=True)
    assert_type(
        coop.load(block, source, items_per_thread=1), coop.ThreadData[Any]
    )
    for algorithm in (
        "direct",
        "striped",
        "vectorize",
        "transpose",
        "warp_transpose",
        "warp_transpose_timesliced",
    ):
        values = coop.load(
            block,
            source,
            items_per_thread=2,
            algorithm=algorithm,
            alignment=16,
            valid_items=61,
            oob_default=-1,
            offset=4,
            temp_storage=scratch,
        )
        assert_type(values, coop.ThreadData[Any])
        assert_type(
            coop.store(
                block,
                destination,
                values,
                algorithm=algorithm,
                valid_items=61,
                offset=4,
                temp_storage=scratch,
            ),
            None,
        )


def check_cutlass_dynamic_memory_controls(
    signed: Int32, unsigned: Uint32
) -> None:
    """Check CuTe counts, offsets and sentinels, plus scalar Store."""
    block = coop.this_block()
    values = coop.ThreadData(items_per_thread=2, dtype=Float64)
    assert_type(
        coop.load(
            block,
            object(),
            valid_items=signed,
            oob_default=Float64(0),
            offset=unsigned,
            items_per_thread=2,
        ),
        coop.ThreadData[Any],
    )
    assert_type(
        coop.store(
            block, object(), values, valid_items=unsigned, offset=signed
        ),
        None,
    )
    assert_type(coop.store(block, object(), Uint64(1)), None)


def check_cutlass_warp_surface(source: object, destination: object) -> None:
    """Check Warp Load/Store types with block and warp groups.

    With a physical warp and each of the four Warp algorithms, Load accepts
    ``valid_items``, ``oob_default``, and ``offset``, and Store accepts
    ``valid_items``. Group types preserve their supported operation
    contracts.
    """
    warp = coop.this_warp()
    assert_type(
        coop.load(warp, source, items_per_thread=1, alignment=16),
        coop.ThreadData[Any],
    )
    values = coop.ThreadData(items_per_thread=2, dtype=np.int32)
    assert_type(warp, coop.ThreadGroup[Literal["warp"]])
    assert_type(
        coop.load(warp, source, items_per_thread=2), coop.ThreadData[Any]
    )
    assert_type(coop.store(warp, destination, values), None)
    for algorithm in ("direct", "striped", "vectorize", "transpose"):
        coop.load(
            warp,
            source,
            algorithm=algorithm,
            valid_items=61,
            oob_default=-1,
            offset=4,
            items_per_thread=2,
        )
        coop.store(
            warp, destination, values, algorithm=algorithm, valid_items=61
        )


def check_cutlass_logical_warp_surface(
    source: object, destination: object
) -> None:
    """Check logical Warp group kinds and their Load/Store calls.

    Each width keeps the ``threads_within_warp`` kind. Qualified calls accept
    all four Warp algorithms. Mapped group kinds remain distinct.
    """
    values = coop.ThreadData(items_per_thread=2, dtype=np.int32)
    for width in (1, 2, 4, 8, 16, 32):
        group = coop.this_warp().group_by(width)
        assert_type(
            coop.load(
                group,
                source,
                items_per_thread=2,
                valid_items=width,
                oob_default=0,
            ),
            coop.ThreadData[Any],
        )
        assert_type(group, coop.ThreadGroup[Literal["threads_within_warp"]])
        for algorithm in ("direct", "striped", "vectorize", "transpose"):
            assert_type(
                coop.load(
                    group, source, algorithm=algorithm, items_per_thread=2
                ),
                coop.ThreadData[Any],
            )
            assert_type(
                coop.store(group, destination, values, algorithm=algorithm),
                None,
            )
    group = coop.this_warp().group_by(8, exhaustive=False)
    assert_type(group, coop.ThreadGroup[Literal["threads_within_warp"]])
    mapped = coop.this_block().group_by(2, exhaustive=False)
    assert_type(mapped, coop.ThreadGroup[Literal["warps_within_block"]])
