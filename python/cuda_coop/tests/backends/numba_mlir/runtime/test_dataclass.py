# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

from __future__ import annotations

import dataclasses
import gc
import weakref

import numpy as np
import pytest

cuda = pytest.importorskip("numba_cuda_mlir.cuda")
pytest.importorskip("numba_cuda_mlir")


import cuda.coop.numba_mlir as coop

from ..support.runtime import (
    ITEMS_PER_THREAD,
    THREADS,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore::numba_cuda_mlir.numba_cuda.core.errors.NumbaPerformanceWarning"
)


def test_gpu_dataclass_registry_releases_dead_instances():
    from cuda.coop.numba_mlir._dataclass import (
        _GPU_DATACLASS_SIGNATURE_REFCOUNTS,
        _GPU_DATACLASS_TYPES_BY_CLASS,
        _GPU_DATACLASS_TYPES_BY_SIGNATURE,
    )

    @dataclasses.dataclass
    class Traits:
        value: np.int32

    traits = coop.gpu_dataclass(Traits(np.int32(1)), compute_temp_storage=False)
    traits_id = id(traits)
    traits_ref = weakref.ref(traits)
    signature_key = next(
        key for key in _GPU_DATACLASS_TYPES_BY_SIGNATURE if key[0] is Traits
    )

    assert traits_id in _GPU_DATACLASS_TYPES_BY_CLASS[Traits]
    assert signature_key in _GPU_DATACLASS_TYPES_BY_SIGNATURE
    assert _GPU_DATACLASS_SIGNATURE_REFCOUNTS[signature_key] == 1

    del traits
    gc.collect()

    assert traits_ref() is None
    assert traits_id not in _GPU_DATACLASS_TYPES_BY_CLASS.get(Traits, {})
    assert signature_key not in _GPU_DATACLASS_TYPES_BY_SIGNATURE
    assert signature_key not in _GPU_DATACLASS_SIGNATURE_REFCOUNTS


def test_gpu_dataclass_primitive_temp_storage_metadata():
    @dataclasses.dataclass
    class Traits:
        load: object

    load = coop._block.load(
        dtype="int32",
        threads_per_block=THREADS,
        items_per_thread=ITEMS_PER_THREAD,
    )
    traits = coop.gpu_dataclass(Traits(load))

    assert traits.temp_storage_bytes_sum == load.temp_storage_bytes
    assert traits.temp_storage_bytes_max == load.temp_storage_bytes
    assert traits.temp_storage_alignment == load.temp_storage_alignment
