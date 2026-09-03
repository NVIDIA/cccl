# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Strict consumer of the qualified Numba-CUDA-MLIR surface."""

from __future__ import annotations

from typing import Literal

import numpy as np
from typing_extensions import assert_type

import cuda.coop.numba_mlir as coop
from cuda.coop import ThreadDataLike


def check_numba_surface(source: object, destination: object) -> None:
    """Exercise Numba declarations through their public package."""

    block = coop.this_block()
    byte_values = coop.ThreadData(1, np.int8)
    values = coop.ThreadData(2, np.uint16, alignas=16)
    compatibility_values = coop.ThreadData(1, np.float32, alignment=8)
    storage = coop.TempStorage(alignment=16, sharing="shared")

    assert_type(block, coop.ThreadGroup[Literal["block"]])
    assert_type(byte_values, ThreadDataLike[np.int8])
    assert_type(values, ThreadDataLike[np.uint16])
    assert_type(compatibility_values, ThreadDataLike[np.float32])
    assert_type(storage, coop.TempStorage)
    assert_type(
        coop.load(
            block,
            source,
            values,
            algorithm=coop.BlockLoadAlgorithm.DIRECT,
            temp_storage=storage,
        ),
        ThreadDataLike[np.uint16],
    )
    assert_type(
        coop.load(block, source, byte_values),
        ThreadDataLike[np.int8],
    )
    assert_type(
        coop.load(block, source, values, algorithm="vectorize"),
        ThreadDataLike[np.uint16],
    )
    assert_type(
        coop.load(
            block,
            source,
            values,
            algorithm=coop.BlockLoadAlgorithm.WARP_TRANSPOSE_TIMESLICED,
        ),
        ThreadDataLike[np.uint16],
    )
    assert_type(
        coop.store(
            block,
            destination,
            values,
            algorithm=coop.BlockStoreAlgorithm.DIRECT,
            temp_storage=storage,
        ),
        None,
    )
    assert_type(coop.store(block, destination, byte_values), None)

    block_load_algorithms: tuple[coop.BlockLoadAlgorithm, ...] = (
        coop.BlockLoadAlgorithm.DIRECT,
        coop.BlockLoadAlgorithm.STRIPED,
        coop.BlockLoadAlgorithm.VECTORIZE,
        coop.BlockLoadAlgorithm.TRANSPOSE,
        coop.BlockLoadAlgorithm.WARP_TRANSPOSE,
        coop.BlockLoadAlgorithm.WARP_TRANSPOSE_TIMESLICED,
    )
    block_store_algorithms: tuple[coop.BlockStoreAlgorithm, ...] = (
        coop.BlockStoreAlgorithm.DIRECT,
        coop.BlockStoreAlgorithm.STRIPED,
        coop.BlockStoreAlgorithm.VECTORIZE,
        coop.BlockStoreAlgorithm.TRANSPOSE,
        coop.BlockStoreAlgorithm.WARP_TRANSPOSE,
        coop.BlockStoreAlgorithm.WARP_TRANSPOSE_TIMESLICED,
    )
    assert_type(block_load_algorithms, tuple[coop.BlockLoadAlgorithm, ...])
    assert_type(block_store_algorithms, tuple[coop.BlockStoreAlgorithm, ...])
