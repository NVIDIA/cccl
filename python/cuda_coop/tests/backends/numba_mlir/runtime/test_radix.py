# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

"""Stable radix ordering against independent host oracles."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

cuda = pytest.importorskip("numba_cuda_mlir.cuda")
if not cuda.is_available():
    pytest.skip("requires a CUDA-capable runtime", allow_module_level=True)
from numba_cuda_mlir import types

import cuda.coop.numba_mlir as numba_coop
from cuda import coop

pytestmark = [
    pytest.mark.backend_numba_mlir,
    pytest.mark.runtime,
    pytest.mark.gpu,
    pytest.mark.filterwarnings(
        "ignore::numba_cuda_mlir.numba_cuda.core.errors.NumbaPerformanceWarning"
    ),
]

_THREADS = 64
_ITEMS = 3


def _ordered_bits(keys):
    unsigned = np.dtype(f"uint{keys.dtype.itemsize * 8}")
    bits = keys.view(unsigned).copy()
    sign = np.array(1 << (keys.dtype.itemsize * 8 - 1), dtype=unsigned)
    if keys.dtype.kind == "i":
        bits ^= sign
    elif keys.dtype.kind == "f":
        bits = np.where(bits & sign, ~bits, bits ^ sign)
        # CUB treats negative and positive zero as equivalent.
        bits[keys == 0] = sign
    return bits


def _permutation(keys, begin, end, descending):
    digits = (_ordered_bits(keys) >> begin) & np.array(
        (1 << (end - begin)) - 1, dtype=f"uint{keys.dtype.itemsize * 8}"
    )
    return np.argsort(~digits if descending else digits, kind="stable")


@pytest.mark.parametrize(
    "dtype", [np.int32, np.uint32, np.int64, np.uint64, np.float32, np.float64]
)
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("partial", [False, True])
def test_pairs_preserve_stability_association_and_inputs(dtype, descending, partial):
    compiler_dtype = getattr(types, np.dtype(dtype).name)
    qualified = np.dtype(dtype).kind == "f"
    source = ((np.arange(_THREADS * _ITEMS, dtype=np.int64) * 17) % 43 - 21).astype(
        dtype
    )
    if qualified:
        source[0:6] = [np.inf, -np.inf, -0.0, 0.0, -1.5, 1.5]
    payload = np.arange(source.size, dtype=np.float64) + 0.25
    begin = 2 if partial else 0
    end = 6 if partial else source.dtype.itemsize * 8

    @cuda.jit
    def kernel(keys_in, values_in, keys_out, values_out, preserved, begin_bit, end_bit):
        t = cuda.threadIdx.x
        keys = coop.ThreadData(_ITEMS, dtype=compiler_dtype)
        values = coop.ThreadData(_ITEMS, dtype=types.float64)
        for i in range(_ITEMS):
            keys[i] = keys_in[t * _ITEMS + i]
            values[i] = values_in[t * _ITEMS + i]
        if qualified:
            sorted_keys, sorted_values = numba_coop.radix_sort_pairs(
                numba_coop.this_block(),
                keys,
                values,
                begin_bit=begin_bit,
                end_bit=end_bit,
                descending=descending,
            )
        else:
            sorted_keys, sorted_values = coop.radix_sort_pairs(
                coop.this_block(),
                keys,
                values,
                begin_bit=begin_bit,
                end_bit=end_bit,
                descending=descending,
                temp_storage=coop.TempStorage(),
            )
        for i in range(_ITEMS):
            keys_out[t * _ITEMS + i] = sorted_keys[i]
            values_out[t * _ITEMS + i] = sorted_values[i]
            preserved[t * _ITEMS + i] = keys[i]

    keys_out = np.empty_like(source)
    values_out = np.empty_like(payload)
    preserved = np.empty_like(source)
    kernel[1, _THREADS](source, payload, keys_out, values_out, preserved, begin, end)
    order = _permutation(source, begin, end, descending)
    np.testing.assert_array_equal(keys_out, source[order])
    np.testing.assert_array_equal(values_out, payload[order])
    np.testing.assert_array_equal(
        preserved.view(f"uint{source.dtype.itemsize * 8}"),
        source.view(f"uint{source.dtype.itemsize * 8}"),
    )


@pytest.mark.parametrize("dtype", [np.int32, np.uint32, np.int64, np.uint64])
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("sign_window", [False, True])
def test_rank_is_stable_signed_int32_and_composes(dtype, descending, sign_window):
    compiler_dtype = getattr(types, np.dtype(dtype).name)
    source = ((np.arange(_THREADS * _ITEMS, dtype=np.int64) * 17) % 43 - 21).astype(
        dtype
    )
    begin = source.dtype.itemsize * 8 - 4 if sign_window else 0

    @cuda.jit
    def kernel(keys_in, output, sorted_ranks, preserved):
        t = cuda.threadIdx.x
        keys = coop.ThreadData(_ITEMS, dtype=compiler_dtype)
        for i in range(_ITEMS):
            keys[i] = keys_in[t * _ITEMS + i]
        ranks = coop.radix_rank(
            coop.this_block(), keys, begin_bit=begin, descending=descending
        )
        ordered = coop.radix_sort_keys(coop.this_block(), ranks)
        for i in range(_ITEMS):
            output[t * _ITEMS + i] = ranks[i]
            sorted_ranks[t * _ITEMS + i] = ordered[i]
            preserved[t * _ITEMS + i] = keys[i]

    output = np.empty(source.size, dtype=np.int32)
    ordered = np.empty_like(output)
    preserved = np.empty_like(source)
    kernel[1, _THREADS](source, output, ordered, preserved)
    permutation = _permutation(source, begin, begin + 4, descending)
    expected = np.empty_like(output)
    expected[permutation] = np.arange(source.size, dtype=np.int32)
    np.testing.assert_array_equal(output, expected)
    np.testing.assert_array_equal(ordered, np.arange(source.size, dtype=np.int32))
    np.testing.assert_array_equal(preserved, source)


@pytest.mark.parametrize("descending", [False, True])
def test_qualified_striped_keys_and_exclusive_prefix(descending):
    source = (np.arange(_THREADS * _ITEMS, dtype=np.int32) * 17) % 97

    @cuda.jit
    def kernel(keys_in, output, ranks_out, prefixes):
        t = cuda.threadIdx.x
        keys = cuda.local.array(_ITEMS, dtype=types.int32)
        for i in range(_ITEMS):
            keys[i] = keys_in[t * _ITEMS + i]
        prefix = numba_coop.ThreadData(2, dtype=types.int32)
        ranks = numba_coop.radix_rank(
            numba_coop.this_block(),
            keys,
            radix_bits=7,
            descending=descending,
            exclusive_digit_prefix=prefix,
        )
        result = numba_coop.radix_sort_keys(
            numba_coop.this_block(),
            keys,
            blocked_to_striped=True,
            descending=descending,
        )
        for i in range(_ITEMS):
            output[i * _THREADS + t] = result[i]
            ranks_out[t * _ITEMS + i] = ranks[i]
        for i in range(2):
            prefixes[t * 2 + i] = prefix[i]

    output = np.empty_like(source)
    ranks = np.empty_like(source)
    prefixes = np.empty(_THREADS * 2, dtype=np.int32)
    kernel[1, _THREADS](source, output, ranks, prefixes)
    permutation = _permutation(source, 0, 7, descending)
    np.testing.assert_array_equal(output, source[permutation])
    expected_ranks = np.empty_like(ranks)
    expected_ranks[permutation] = np.arange(source.size)
    np.testing.assert_array_equal(ranks, expected_ranks)
    counts = np.bincount(source, minlength=128)
    expected_prefix = (
        (source.size - np.cumsum(counts))
        if descending
        else np.r_[0, np.cumsum(counts[:-1])]
    )
    # CUB keeps ascending bin indices; descending prefixes count greater digits.
    np.testing.assert_array_equal(prefixes, expected_prefix)


def test_qualified_scalar_sort_and_rank():
    source = np.arange(_THREADS, dtype=np.int32)[::-1].copy()

    @cuda.jit
    def kernel(source, sorted_out, rank_out):
        t = cuda.threadIdx.x
        sorted_out[t] = numba_coop.radix_sort_keys(numba_coop.this_block(), source[t])
        rank_out[t] = numba_coop.radix_rank(
            numba_coop.this_block(), source[t], radix_bits=6
        )

    sorted_out = np.empty_like(source)
    rank_out = np.empty_like(source)
    kernel[1, _THREADS](source, sorted_out, rank_out)
    np.testing.assert_array_equal(sorted_out, np.sort(source))
    np.testing.assert_array_equal(rank_out, source)


@pytest.mark.parametrize("pairs", [False, True])
@pytest.mark.parametrize(
    "begin,end,unsigned",
    [
        (-1, 32, False),
        (0, 33, False),
        (8, 4, False),
        (4, 4, False),
        (2**32, 2**32 + 8, False),
        (0, 2**32 - 1, True),
    ],
)
def test_invalid_runtime_bit_intervals_trap_before_narrowing(
    pairs, begin, end, unsigned
):
    # Device traps poison their context; keep each invalid launch in a child.
    script = f"""
import numpy as np
from pathlib import Path
from numba_cuda_mlir import cuda, types
import cuda.coop.numba_mlir as numba_coop
from cuda import coop
assert Path(numba_coop.__file__).resolve() == Path({str(Path(numba_coop.__file__).resolve())!r})
@cuda.jit
def kernel(source, output, begin, end):
    keys = coop.ThreadData(2, dtype=types.int32)
    keys[0] = source[cuda.threadIdx.x * 2]
    keys[1] = source[cuda.threadIdx.x * 2 + 1]
    if {pairs!r}:
        result, values = coop.radix_sort_pairs(coop.this_block(), keys, keys, begin_bit=begin, end_bit=end)
    else:
        result = coop.radix_sort_keys(coop.this_block(), keys, begin_bit=begin, end_bit=end)
    output[cuda.threadIdx.x] = result[0]
source = np.arange(128, dtype=np.int32)
output = np.empty(64, dtype=np.int32)
dtype = np.uint32 if {unsigned!r} else np.int64
kernel[1, 64](source, output, dtype({begin}), dtype({end}))
cuda.synchronize()
raise AssertionError("invalid radix interval did not trap")
"""
    result = subprocess.run(
        [
            sys.executable,
            "-P" if sys.version_info >= (3, 11) else "-I",
            "-B",
            "-c",
            script,
        ],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    output = result.stdout + result.stderr
    assert result.returncode != 0, output
    assert any(
        error in output
        for error in ("CUDA_ERROR_ILLEGAL_INSTRUCTION", "CUDA_ERROR_LAUNCH_FAILED")
    ), output


@pytest.mark.parametrize("qualified", [False, True])
@pytest.mark.parametrize("pairs", [False, True])
def test_chained_sorts_and_ranks_infer_dtypes_from_indexed_writes(qualified, pairs):
    api = numba_coop if qualified else coop

    @cuda.jit
    def kernel(source, payload, output, associated, ordered_ranks, preserved):
        block = api.this_block()
        keys = api.ThreadData(_ITEMS)
        values = api.ThreadData(_ITEMS)
        for item in range(_ITEMS):
            index = cuda.threadIdx.x * _ITEMS + item
            keys[item] = source[index]
            values[item] = payload[index]
        if pairs:
            first_keys, first_values = api.radix_sort_pairs(
                block, keys, values, descending=True
            )
            chosen, chosen_values = api.radix_sort_pairs(
                block, first_keys, first_values
            )
            api.store(block, associated, chosen_values)
        else:
            first = api.radix_sort_keys(block, keys, descending=True)
            chosen = api.radix_sort_keys(block, first)
        ranks = api.radix_rank(block, chosen, radix_bits=8)
        sorted_ranks = api.radix_sort_keys(block, ranks)
        api.store(block, ordered_ranks, sorted_ranks)
        api.store(block, output, chosen)
        api.store(block, preserved, keys)

    source = ((np.arange(_THREADS * _ITEMS) * 17) % 43 - 21).astype(np.int64)
    payload = np.arange(source.size, dtype=np.float32) + np.float32(0.25)
    output = np.empty_like(source)
    associated = np.empty_like(payload)
    ordered_ranks = np.empty(source.size, dtype=np.int32)
    preserved = np.empty_like(source)
    kernel[1, _THREADS](source, payload, output, associated, ordered_ranks, preserved)
    cuda.synchronize()
    permutation = np.argsort(source, kind="stable")
    np.testing.assert_array_equal(output, source[permutation])
    np.testing.assert_array_equal(preserved, source)
    np.testing.assert_array_equal(ordered_ranks, np.arange(source.size, dtype=np.int32))
    if pairs:
        np.testing.assert_array_equal(associated, payload[permutation])
