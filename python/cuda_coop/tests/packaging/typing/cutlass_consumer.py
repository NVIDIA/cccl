# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Strict consumer of the qualified CUTLASS Block Load/Store surface."""

from __future__ import annotations

from typing import Literal

import numpy as np
from typing_extensions import assert_type

import cuda.coop.cutlass as cutlass_coop
from cuda import coop as common_coop


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
