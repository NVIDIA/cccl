# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Strict consumer of qualified CUTLASS Block and Warp Load/Store."""

from __future__ import annotations

from typing import Literal

import numpy as np
from typing_extensions import assert_type

import cuda.coop.cutlass as cutlass_coop
from cuda import coop as common_coop


def register_cutlass() -> None:
    assert_type(common_coop.register("cutlass"), None)


def check_cutlass_surface(source: object, destination: object) -> None:
    block = cutlass_coop.this_block()
    values = cutlass_coop.ThreadData(2, np.int32, alignment=16)
    assert_type(block, cutlass_coop.ThreadGroup[Literal["block"]])
    assert_type(values, cutlass_coop.ThreadData[np.int32])
    assert_type(values[0], np.int32)
    assert_type(cutlass_coop.load(block, source, values), None)
    assert_type(cutlass_coop.store(block, destination, values), None)
    cutlass_coop.load(
        block,
        source,
        values,
        algorithm="direct",
        valid_items=31,
        oob_default=-1,
        offset=4,
    )
    cutlass_coop.store(block, destination, values, valid_items=31, offset=4)
    common_coop.load(common_coop.this_block(), source, values)
    common_coop.store(common_coop.this_block(), destination, values)

    storage = cutlass_coop.TempStorage(alignment=1, sharing="exclusive")
    assert_type(storage, cutlass_coop.TempStorage)
    assert_type(storage.auto_sync, bool)
    assert_type(storage.size_in_bytes, int | None)
    assert_type(storage.alignment, int | None)
    cutlass_coop.load(
        block, source, values, algorithm="transpose", temp_storage=storage
    )
    cutlass_coop.store(
        block, destination, values, algorithm="transpose", temp_storage=storage
    )
    manual = cutlass_coop.TempStorage(
        16384, alignment=32, auto_sync=False, sharing="shared"
    )
    common_coop.load(
        common_coop.this_block(),
        source,
        values,
        algorithm="transpose",
        temp_storage=manual,
    )
    assert_type(manual.sync(), None)
    common_coop.store(
        common_coop.this_block(),
        destination,
        values,
        algorithm="transpose",
        temp_storage=manual,
    )
    manual.sync()

    common_storage = common_coop.TempStorage(alignment=32)
    cutlass_coop.store(
        block, destination, values, algorithm="transpose", temp_storage=common_storage
    )

    copied = cutlass_coop.ThreadData.from_payload(values)
    assert_type(copied, cutlass_coop.ThreadData[np.int32])
    generated = cutlass_coop.ThreadData.from_fn(2, lambda index: np.int32(index))
    assert_type(generated, cutlass_coop.ThreadData[np.int32])
    initialized = cutlass_coop.ThreadData.from_values(np.int32(1), np.int32(2))
    assert_type(initialized, cutlass_coop.ThreadData[np.int32])
    fragment = values.to_register_tensor()
    restored = cutlass_coop.ThreadData.from_register_tensor(fragment, dtype=np.int32)
    assert_type(restored, cutlass_coop.ThreadData[np.int32])
    vector = values.to_tensor_ssa()
    restored_vector = cutlass_coop.ThreadData.from_vector(vector, dtype=np.int32)
    assert_type(restored_vector, cutlass_coop.ThreadData[np.int32])


def check_cutlass_warp_surface(source: object, destination: object) -> None:
    warp = cutlass_coop.this_warp()
    values = cutlass_coop.ThreadData(2, np.int32)
    assert_type(warp, cutlass_coop.ThreadGroup[Literal["warp"]])
    assert_type(cutlass_coop.load(warp, source, values), None)
    assert_type(cutlass_coop.store(warp, destination, values), None)
    for algorithm in ("direct", "striped", "vectorize", "transpose"):
        cutlass_coop.load(
            warp,
            source,
            values,
            algorithm=algorithm,
            valid_items=61,
            oob_default=-1,
            offset=4,
        )
        cutlass_coop.store(
            warp, destination, values, algorithm=algorithm, valid_items=61
        )
    common_coop.load(warp, source, values)
    common_coop.store(warp, destination, values)
    cutlass_coop.load(common_coop.this_warp(), source, values)
    cutlass_coop.store(common_coop.this_warp(), destination, values)


def check_cutlass_logical_warp_surface(source: object, destination: object) -> None:
    values = cutlass_coop.ThreadData(2, np.int32)
    for width in (1, 2, 4, 8, 16, 32):
        group = cutlass_coop.this_warp().group_by(width)
        assert_type(group, cutlass_coop.ThreadGroup[Literal["threads_within_warp"]])
        for algorithm in ("direct", "striped", "vectorize", "transpose"):
            assert_type(
                cutlass_coop.load(group, source, values, algorithm=algorithm), None
            )
            assert_type(
                cutlass_coop.store(group, destination, values, algorithm=algorithm),
                None,
            )
        common_coop.load(group, source, values)
        common_coop.store(group, destination, values)
    group = cutlass_coop.this_warp().group_by(8, exhaustive=False)
    assert_type(group, cutlass_coop.ThreadGroup[Literal["threads_within_warp"]])
    cutlass_coop.load(common_coop.this_warp().group_by(8), source, values)
    cutlass_coop.store(common_coop.this_warp().group_by(8), destination, values)

    mapped = cutlass_coop.this_block().group_by(2, exhaustive=False)
    assert_type(mapped, cutlass_coop.ThreadGroup[Literal["warps_within_block"]])
