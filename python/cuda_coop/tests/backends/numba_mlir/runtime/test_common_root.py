# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

import numpy as np
import pytest

cuda = pytest.importorskip("numba_cuda_mlir.cuda")
pytest.importorskip("numba_cuda_mlir")

from numba_cuda_mlir import types

from cuda import coop
from cuda.coop import numba_mlir as _numba_mlir_coop  # noqa: F401

THREADS = 32
pytestmark = pytest.mark.filterwarnings(
    "ignore::numba_cuda_mlir.numba_cuda.core.errors.NumbaPerformanceWarning"
)


@cuda.jit
def _common_root_kernel(d_input, d_round_trip, d_total):
    tid = cuda.threadIdx.x
    group = coop.this_block()
    items = coop.ThreadData(1, dtype=types.int32)

    coop.load(group, d_input, items)
    coop.store(group, d_round_trip, items)
    d_total[tid] = coop.reduce(group, items[0])


def test_common_root_load_store_reduce_without_compiler_scope():
    values = np.arange(THREADS, dtype=np.int32)
    round_trip = np.zeros_like(values)
    totals = np.zeros_like(values)

    _common_root_kernel[1, THREADS](values, round_trip, totals)

    np.testing.assert_array_equal(round_trip, values)
    np.testing.assert_array_equal(totals, np.full_like(values, values.sum()))

    round_trip.fill(-1)
    totals.fill(-1)
    _common_root_kernel[1, THREADS](values, round_trip, totals)

    np.testing.assert_array_equal(round_trip, values)
    np.testing.assert_array_equal(totals, np.full_like(values, values.sum()))


def test_portable_root_sum_example_runtime(source_examples):
    del source_examples
    from examples.numba_mlir import portable_root_sum

    assert portable_root_sum.run_example() == sum(
        range(1, portable_root_sum.TILE_ITEMS + 1)
    )
