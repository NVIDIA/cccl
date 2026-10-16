# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Partial tiles, offsets and scalar stores through the typed pipeline, run
on the GPU and checked against NumPy.

A load with ``valid_items=n`` reads the first ``n`` items of the tile and
leaves the rest as they were, or sets them to ``oob_default``; a store with
``valid_items=n`` writes only the first ``n``. ``offset=k`` moves the tile to
start at item ``k``.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from numba_cuda_mlir import cuda

from cuda import coop

from .._support import requires_typed_pipeline, run_python

pytestmark = requires_typed_pipeline

THREADS = 128
ITEMS = 4
TILE = THREADS * ITEMS
ALGORITHMS = ("direct", "transpose")
UNSET = -1.0


def _source(size=TILE, dtype=np.float32):
    return np.arange(size, dtype=dtype) * 3 + 1


def _blank(size=TILE, dtype=np.float32):
    return cuda.to_device(np.full(size, UNSET, dtype=dtype))


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_partial_load_preserves_tail(algorithm):
    @cuda.jit
    def partial_load(source, output, n):
        block = coop.this_block()
        items = coop.ThreadData(4, dtype=source.dtype)
        for i in range(4):
            items[i] = -1.0
        coop.load(block, source, items, algorithm=algorithm, valid_items=n)
        base = cuda.threadIdx.x * 4
        for i in range(4):
            output[base + i] = items[i]

    source, out = _source(), _blank()
    partial_load[1, THREADS](cuda.to_device(source), out, 300)
    expected = np.full(TILE, UNSET, np.float32)
    expected[:300] = source[:300]
    np.testing.assert_array_equal(out.copy_to_host(), expected)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_load_a_partial_tile_with_oob_default(algorithm):
    @cuda.jit
    def padded_load(source, output, n, pad):
        block = coop.this_block()
        items = coop.ThreadData(4)
        coop.load(
            block,
            source,
            items,
            algorithm=algorithm,
            valid_items=n,
            oob_default=pad,
        )
        coop.store(block, output, items, algorithm=algorithm)

    source, out = _source(), _blank()
    padded_load[1, THREADS](cuda.to_device(source), out, 77, np.float32(-9))
    expected = np.full(TILE, -9, np.float32)
    expected[:77] = source[:77]
    np.testing.assert_array_equal(out.copy_to_host(), expected)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_store_a_partial_tile(algorithm):
    @cuda.jit
    def partial_store(source, output, n):
        block = coop.this_block()
        items = coop.ThreadData(4)
        coop.load(block, source, items, algorithm=algorithm)
        coop.store(block, output, items, algorithm=algorithm, valid_items=n)

    source, out = _source(), _blank()
    partial_store[1, THREADS](cuda.to_device(source), out, np.uint32(200))
    expected = np.full(TILE, UNSET, np.float32)
    expected[:200] = source[:200]
    np.testing.assert_array_equal(out.copy_to_host(), expected)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_runtime_offsets(algorithm):
    @cuda.jit
    def shifted(source, output, load_at, store_at):
        block = coop.this_block()
        items = coop.ThreadData(4)
        coop.load(block, source, items, algorithm=algorithm, offset=load_at)
        coop.store(block, output, items, algorithm=algorithm, offset=store_at)

    source, out = _source(3 * TILE), _blank(3 * TILE)
    shifted[1, THREADS](
        cuda.to_device(source), out, np.int32(TILE + 5), 2 * TILE
    )
    expected = np.full(3 * TILE, UNSET, np.float32)
    expected[2 * TILE :] = source[TILE + 5 : 2 * TILE + 5]
    np.testing.assert_array_equal(out.copy_to_host(), expected)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_static_valid_items_and_offset(algorithm):
    @cuda.jit
    def fixed(source, output):
        block = coop.this_block()
        items = coop.ThreadData(4)
        coop.load(
            block,
            source,
            items,
            algorithm=algorithm,
            valid_items=100,
            oob_default=0,
            offset=8,
        )
        coop.store(block, output, items, algorithm=algorithm, valid_items=450)

    source, out = _source(TILE + 8), _blank()
    fixed[1, THREADS](cuda.to_device(source), out)
    expected = np.full(TILE, UNSET, np.float32)
    expected[:450] = 0
    expected[:100] = source[8:108]
    np.testing.assert_array_equal(out.copy_to_host(), expected)


def test_a_partial_tile_of_a_multi_block_launch():
    @cuda.jit
    def tail(source, output, size):
        block = coop.this_block()
        start = cuda.blockIdx.x * 512
        n = min(size - start, 512)
        items = coop.ThreadData(4)
        coop.load(block, source, items, valid_items=n, offset=start)
        coop.store(block, output, items, valid_items=n, offset=start)

    size = 2 * TILE + 123
    source, out = _source(size), _blank(size)
    tail[3, THREADS](cuda.to_device(source), out, size)
    np.testing.assert_array_equal(out.copy_to_host(), source)


def test_scalar_store_runtime_and_static():
    @cuda.jit
    def fill(output, value):
        coop.store(coop.this_block(), output, value)
        coop.store(coop.this_block(), output, 7, offset=THREADS)
        coop.store(coop.this_block(), output, value, valid_items=5, offset=256)

    out = _blank(3 * THREADS, np.int32)
    fill[1, THREADS](out, np.int32(42))
    expected = np.full(3 * THREADS, UNSET, np.int32)
    expected[:THREADS] = 42
    expected[THREADS : 2 * THREADS] = 7
    expected[256:261] = 42
    np.testing.assert_array_equal(out.copy_to_host(), expected)


def test_runtime_values_share_a_specialization_and_literals_do_not():
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

        def runtime_a(source, output, n):
            block = coop.this_block()
            items = coop.ThreadData(4)
            coop.load(block, source, items, valid_items=n, oob_default=0.0)
            coop.store(block, output, items)

        def runtime_b(source, output, n):
            block = coop.this_block()
            items = coop.ThreadData(4)
            coop.load(block, source, items, valid_items=n, oob_default=0.0)
            coop.store(block, output, items)

        def literal_100(source, output, n):
            block = coop.this_block()
            items = coop.ThreadData(4)
            coop.load(block, source, items, valid_items=100, oob_default=0.0)
            coop.store(block, output, items)

        def literal_200(source, output, n):
            block = coop.this_block()
            items = coop.ThreadData(4)
            coop.load(block, source, items, valid_items=200, oob_default=0.0)
            coop.store(block, output, items)

        source = np.arange(512, dtype=np.float32) + 1
        d_source = cuda.to_device(source)
        counts = {}
        for function, n in [
            (runtime_a, 100),
            (runtime_b, 200),
            (literal_100, 0),
            (literal_200, 0),
        ]:
            valid = n or int(function.__name__.split("_")[1])
            before = len(calls)
            d_out = cuda.device_array_like(source)
            cuda.jit(function)[1, 128](d_source, d_out, n)
            expected = np.where(np.arange(512) < valid, source, 0)
            assert (d_out.copy_to_host() == expected).all(), function.__name__
            counts[function.__name__] = len(calls) - before
        print(json.dumps(counts))
        """,
        "typed",
    )
    assert result.returncode == 0, result.stderr
    counts = json.loads(result.stdout.strip().splitlines()[-1])
    assert counts == {
        "runtime_a": 1,
        "runtime_b": 0,
        "literal_100": 1,
        "literal_200": 1,
    }
