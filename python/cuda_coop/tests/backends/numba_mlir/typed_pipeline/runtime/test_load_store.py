# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Block load and store through the typed pipeline, run on the GPU.

The block comes from the launch, so every kernel here is compiled at its
first launch. Results are checked against NumPy's view of the blocked and
striped arrangements.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from numba_cuda_mlir import cuda

import cuda.coop as coop_common
import cuda.coop.numba_mlir as coop_backend

from .._support import ALGORITHMS, requires_typed_pipeline, run_python

pytestmark = requires_typed_pipeline

THREADS = 128
ITEMS = 4


def _source(dtype=np.float32):
    return np.arange(THREADS * ITEMS, dtype=dtype) * 3 + 1


def test_helper_read_before_a_producer_uses_the_final_payload_dtype():
    @cuda.jit(device=True, inline="never")
    def read_first(items):
        return items[0]

    @cuda.jit
    def kernel(source, output):
        items = coop_common.ThreadData(1)
        items[0] = np.float64(1.25)
        observed = read_first(items)
        coop_common.load(coop_common.this_block(), source, items)
        result = (observed + np.float32(16777216)) - np.float32(16777216)
        output[cuda.threadIdx.x] = result + items[0]

    output = np.zeros(32, np.float64)
    kernel[1, 32](np.zeros(32, np.float32), output)
    np.testing.assert_array_equal(output, 2.0)


@pytest.mark.parametrize("aggregate", [False, True])
def test_dependent_writes_use_destination_dtype(aggregate):
    @cuda.jit
    def kernel(source, output):
        first = coop_common.ThreadData(1)
        second = coop_common.ThreadData(1)
        if aggregate:
            coop_backend.inclusive_sum(
                coop_backend.this_block(),
                source[cuda.threadIdx.x],
                aggregate_output=first,
            )
        else:
            first[0] = np.int32(7)
        second[0] = first[0]
        second[0] = second[0] + np.float64(0.5)
        coop_common.store(coop_common.this_block(), output, second)

    output = np.zeros(32, np.float64)
    kernel[1, 32](np.ones(32, np.int32), output)
    np.testing.assert_array_equal(output, 32.5 if aggregate else 7.5)


def test_read_modify_write_preserves_inferred_integer_dtype():
    @cuda.jit
    def kernel(output):
        items = coop_common.ThreadData(1)
        items[0] = np.int32(2147483647)
        items[0] = items[0] + np.int32(1)
        coop_common.store(coop_common.this_block(), output, items)

    output = np.zeros(32, np.int32)
    kernel[1, 32](output)
    np.testing.assert_array_equal(output, np.iinfo(np.int32).min)


@pytest.mark.parametrize(
    "coop", [coop_common, coop_backend], ids=["common", "backend"]
)
def test_k1_load_store_inferred(coop):
    @cuda.jit
    def k1(source, output):
        block = coop.this_block()
        items = coop.ThreadData(4)
        coop.load(block, source, items)
        coop.store(block, output, items)

    source = _source()
    d_out = cuda.device_array_like(source)
    k1[1, THREADS](cuda.to_device(source), d_out)
    np.testing.assert_array_equal(d_out.copy_to_host(), source)


@pytest.mark.parametrize("algorithm", ["direct", "transpose"])
def test_branch_joins_constructors(algorithm):
    coop = coop_common

    @cuda.jit
    def joined(source, output, first):
        block = coop.this_block()
        a = coop.ThreadData(ITEMS)
        b = coop.ThreadData(ITEMS)
        if first[0]:
            items = a
        else:
            items = b
        coop.load(block, source, items, algorithm=algorithm)
        for i in range(ITEMS):
            items[i] = items[i] + first[0]
        coop.store(block, output, items, algorithm=algorithm)

    source = _source(np.int32)
    for first in (0, 1):
        output = np.zeros_like(source)
        joined[1, THREADS](source, output, np.array([first], np.int32))
        np.testing.assert_array_equal(output, source + first)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_round_trip(algorithm):
    coop = coop_common

    @cuda.jit
    def round_trip(source, output):
        block = coop.this_block()
        items = coop.ThreadData(4)
        coop.load(block, source, items, algorithm=algorithm)
        coop.store(block, output, items, algorithm=algorithm)

    source = _source(np.int32)
    d_out = cuda.device_array_like(source)
    round_trip[1, THREADS](cuda.to_device(source), d_out)
    np.testing.assert_array_equal(d_out.copy_to_host(), source)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_load_arrangement(algorithm):
    coop = coop_common

    @cuda.jit
    def arrangement(source, output):
        block = coop.this_block()
        items = coop.ThreadData(4)
        coop.load(block, source, items, algorithm=algorithm)
        base = cuda.threadIdx.x * 4
        for i in range(4):
            output[base + i] = items[i]

    source = _source(np.float64)
    d_out = cuda.device_array_like(source)
    arrangement[1, THREADS](cuda.to_device(source), d_out)
    if algorithm == "striped":
        expected = source.reshape(ITEMS, THREADS).T.ravel()
    else:
        expected = source
    np.testing.assert_array_equal(d_out.copy_to_host(), expected)


def test_extent_from_a_kernel_argument():
    coop = coop_common

    @cuda.jit
    def extent(source, output, items_per_thread):
        block = coop.this_block()
        items = coop.ThreadData(items_per_thread)
        coop.load(block, source, items, algorithm="transpose")
        coop.store(block, output, items, algorithm="transpose")

    source = _source()
    d_out = cuda.device_array_like(source)
    extent[1, THREADS](cuda.to_device(source), d_out, ITEMS)
    np.testing.assert_array_equal(d_out.copy_to_host(), source)


def test_one_nvrtc_compile_cold_and_none_warm():
    result = run_python(
        """
        import json

        import numpy as np
        from cuda.bindings import nvrtc
        from numba_cuda_mlir import cuda

        from cuda import coop

        calls = []
        compile_program = nvrtc.nvrtcCompileProgram

        def counting(*args, **kwargs):
            calls.append(1)
            return compile_program(*args, **kwargs)

        nvrtc.nvrtcCompileProgram = counting

        ITEMS_PER_THREAD = 4

        def load_store_inferred_kernel(source, output):
            block = coop.this_block()
            items = coop.ThreadData(ITEMS_PER_THREAD)
            coop.load(block, source, items)
            for i in range(ITEMS_PER_THREAD):
                items[i] = items[i] * 2
            coop.store(block, output, items)

        source = np.arange(512, dtype=np.float32)
        d_source = cuda.to_device(source)
        counts = []
        for _ in range(2):
            before = len(calls)
            d_out = cuda.device_array_like(source)
            kernel = cuda.jit(load_store_inferred_kernel)
            kernel[1, 128](d_source, d_out)
            assert (d_out.copy_to_host() == source * 2).all()
            counts.append(len(calls) - before)
        print(json.dumps(counts))
        """,
        "typed",
    )
    assert result.returncode == 0, result.stderr
    cold, warm = json.loads(result.stdout.strip().splitlines()[-1])
    assert (cold, warm) == (1, 0)


@pytest.mark.parametrize("algorithm", ["direct", "striped", "vectorize"])
def test_storage_free_operations_preserve_user_dynamic_shared_memory(algorithm):
    coop = coop_common

    @cuda.jit
    def round_trip(source, output, preserved):
        own = cuda.shared.array(0, np.int32)
        thread = cuda.threadIdx.x
        own[thread] = 100000 + thread
        cuda.syncthreads()
        items = coop.ThreadData(4)
        coop.load(coop.this_block(), source, items, algorithm=algorithm)
        coop.store(coop.this_block(), output, items, algorithm=algorithm)
        cuda.syncthreads()
        preserved[thread] = own[thread]

    source = _source(np.int32)
    output = np.zeros_like(source)
    preserved = np.zeros(THREADS, np.int32)
    round_trip[1, THREADS, 0, THREADS * 4](source, output, preserved)
    np.testing.assert_array_equal(output, source)
    np.testing.assert_array_equal(preserved, 100000 + np.arange(THREADS))
