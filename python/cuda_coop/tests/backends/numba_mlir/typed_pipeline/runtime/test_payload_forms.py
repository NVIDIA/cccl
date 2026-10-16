# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""A ThreadData's ``items_per_thread``, and local-array payloads, run on the
GPU and checked against NumPy.

``items_per_thread`` is the payload's extent, usable as a loop bound before
the payload's dtype is known. A ``cuda.coop.numba_mlir`` load or store takes
a ``cuda.local.array`` payload of constant shape, or sized by a kernel
argument.
"""

from __future__ import annotations

import numpy as np
import pytest
from numba_cuda_mlir import cuda, types

import cuda.coop.numba_mlir as qualified_coop
from cuda import coop

from .._support import requires_typed_pipeline

pytestmark = requires_typed_pipeline

THREADS = 64


def _source(items, dtype=np.int32):
    return np.arange(THREADS * items, dtype=dtype) * 3 + 1


@pytest.mark.parametrize("items_per_thread", [1, 4])
@pytest.mark.parametrize(
    "dtype", [None, types.int32], ids=["inferred", "explicit"]
)
def test_items_per_thread_bounds_a_loop(dtype, items_per_thread):
    @cuda.jit
    def extent(source, output, extents, items_per_thread):
        items = coop.ThreadData(items_per_thread, dtype)
        coop.load(coop.this_block(), source, items)
        alias = items
        thread = cuda.threadIdx.x
        extents[thread] = alias.items_per_thread
        for i in range(alias.items_per_thread):
            output[thread * items_per_thread + i] = alias[i] * 2

    source = _source(items_per_thread)
    output = np.full_like(source, -1)
    extents = np.zeros(THREADS, np.int32)
    extent[1, THREADS](source, output, extents, items_per_thread)
    np.testing.assert_array_equal(output, source * 2)
    np.testing.assert_array_equal(extents, items_per_thread)


def test_items_per_thread_before_the_dtype_is_known():
    @cuda.jit
    def fill(output):
        items = coop.ThreadData(4)
        base = cuda.threadIdx.x * items.items_per_thread
        for i in range(items.items_per_thread):
            items[i] = np.float32(base + i)
        coop.store(coop.this_block(), output, items)

    output = np.zeros(THREADS * 4, np.float32)
    fill[1, THREADS](output)
    np.testing.assert_array_equal(output, np.arange(THREADS * 4))


@pytest.mark.parametrize("algorithm", ["direct", "transpose"])
def test_a_local_array_payload_sized_by_a_kernel_argument(algorithm):
    @cuda.jit
    def copy(source, output, items_per_thread):
        block = qualified_coop.this_block()
        items = cuda.local.array(shape=items_per_thread, dtype=types.int32)
        qualified_coop.load(block, source, items, algorithm=algorithm)
        for i in range(items_per_thread):
            items[i] += 7
        qualified_coop.store(block, output, items, algorithm=algorithm)

    for items_per_thread in (1, 4, 2):
        source = _source(items_per_thread)
        output = np.full_like(source, -1)
        copy[1, THREADS](source, output, items_per_thread)
        np.testing.assert_array_equal(output, source + 7)


def test_a_local_array_payload_of_constant_shape_with_controls():
    @cuda.jit
    def partial(source, output, n):
        block = qualified_coop.this_block()
        items = cuda.local.array(4, types.float32)
        qualified_coop.load(
            block, source, items, valid_items=n, oob_default=-2.0
        )
        qualified_coop.store(block, output, items, algorithm="striped")

    source = _source(4, np.float32)
    output = np.zeros_like(source)
    partial[1, THREADS](source, output, 99)
    expected = np.full_like(source, -2)
    expected[:99] = source[:99]
    striped = expected.reshape(THREADS, 4).T.ravel()
    np.testing.assert_array_equal(output, striped)
