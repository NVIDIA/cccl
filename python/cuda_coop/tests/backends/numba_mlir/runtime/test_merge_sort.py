# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

"""Merge Sort runtime contracts, including tuple results and input preservation."""

import numpy as np
import pytest

cuda = pytest.importorskip("numba_cuda_mlir.cuda")
if not cuda.is_available():
    pytest.skip("requires a CUDA-capable runtime", allow_module_level=True)

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


def _run(
    *,
    width=64,
    dtype=np.int32,
    value_dtype=np.float64,
    pairs=True,
    partial=False,
    descending=False,
    qualified=False,
    custom=False,
    scratch=False,
    block_dim=64,
    valid_count=None,
):
    api = numba_coop if qualified else coop
    dtype = np.dtype(dtype)
    group_kind = "block" if width == 64 else "warp" if width == 32 else "logical"

    def compare(left, right):
        return left > right

    @cuda.jit
    def kernel(
        source,
        values,
        output,
        associations,
        preserved,
        preserved_values,
        count,
        sentinel,
    ):
        thread = cuda.threadIdx.x + cuda.blockDim.x * (
            cuda.threadIdx.y + cuda.blockDim.y * cuda.threadIdx.z
        )
        keys = api.ThreadData(3)
        payload = api.ThreadData(3)
        for item in range(3):
            keys[item] = source[thread * 3 + item]
            payload[item] = values[thread * 3 + item]
        if group_kind == "block":
            group = api.this_block()
        elif group_kind == "warp":
            group = api.this_warp()
        else:
            group = api.this_warp().group_by(width)
        if pairs:
            if partial:
                result, result_values = api.merge_sort_pairs(
                    group,
                    keys,
                    payload,
                    descending=descending,
                    valid_items=count,
                    oob_default=sentinel,
                )
            elif custom:
                result, result_values = numba_coop.merge_sort_pairs(
                    group, keys, payload, compare_op=compare
                )
            elif scratch:
                storage = api.TempStorage()
                intermediate, inter_values = api.merge_sort_pairs(
                    group, keys, payload, temp_storage=storage
                )
                result, result_values = api.merge_sort_pairs(
                    group,
                    intermediate,
                    inter_values,
                    temp_storage=storage,
                    descending=descending,
                )
            else:
                result, result_values = api.merge_sort_pairs(
                    group, keys, payload, descending=descending
                )
            for item in range(3):
                associations[thread * 3 + item] = result_values[item]
        else:
            if partial:
                result = api.merge_sort_keys(
                    group,
                    keys,
                    descending=descending,
                    valid_items=count,
                    oob_default=sentinel,
                )
            else:
                result = api.merge_sort_keys(group, keys, descending=descending)
        for item in range(3):
            output[thread * 3 + item] = result[item]
            preserved[thread * 3 + item] = keys[item]
            preserved_values[thread * 3 + item] = payload[item]

    # Duplicate keys exercise associations without assuming stable tie ordering.
    source = ((np.arange(192) * 17 + 9) % 47).astype(dtype)
    values = (np.arange(192) + 0.25).astype(value_dtype)
    output = np.zeros_like(source)
    associations = np.zeros_like(values)
    preserved = np.zeros_like(source)
    preserved_values = np.zeros_like(values)
    count = width * 3 - 2 if valid_count is None else valid_count
    limit = np.finfo(dtype).max if dtype.kind == "f" else np.iinfo(dtype).max
    sentinel = dtype.type(0 if descending else limit)
    kernel[1, block_dim](
        source,
        values,
        output,
        associations,
        preserved,
        preserved_values,
        np.int64(count),
        sentinel,
    )
    np.testing.assert_array_equal(preserved, source)
    np.testing.assert_array_equal(preserved_values, values)
    for start in range(0, 192, width * 3):
        size = count if partial else width * 3
        expected = np.sort(source[start : start + size])
        if descending or custom:
            expected = expected[::-1]
        np.testing.assert_array_equal(output[start : start + size], expected)
        if pairs:
            actual = sorted(
                zip(
                    output[start : start + size].tolist(),
                    associations[start : start + size].tolist(),
                )
            )
            original = sorted(
                zip(
                    source[start : start + size].tolist(),
                    values[start : start + size].tolist(),
                )
            )
            assert actual == original


@pytest.mark.parametrize(
    "dtype",
    [
        np.int8,
        np.uint8,
        np.int16,
        np.uint16,
        np.int32,
        np.uint32,
        np.int64,
        np.uint64,
        np.float32,
        np.float64,
    ],
)
@pytest.mark.parametrize("pairs", [False, True])
def test_numeric_payloads(dtype, pairs):
    _run(dtype=dtype, pairs=pairs)


@pytest.mark.parametrize("width", [1, 2, 4, 8, 16, 32])
@pytest.mark.parametrize("pairs,partial", [(False, False), (True, True)])
def test_independent_warp_groups(width, pairs, partial):
    _run(width=width, pairs=pairs, partial=partial, descending=True)


@pytest.mark.parametrize("width", [64, 8, 32])
@pytest.mark.parametrize("valid_count", [0, 1])
def test_partial_empty_and_single_item(width, valid_count):
    _run(width=width, partial=True, valid_count=valid_count)


@pytest.mark.parametrize("dtype", [np.int8, np.uint16, np.float32, np.float64])
def test_partial_key_types(dtype):
    _run(dtype=dtype, partial=True)


def test_multidimensional_block_and_reused_scratch():
    _run(block_dim=(8, 4, 2), scratch=True, descending=True)


@pytest.mark.parametrize("width", [64, 8, 32])
def test_qualified_custom_comparator(width):
    _run(width=width, qualified=True, custom=True)


def test_qualified_comparator_with_nested_device_helper():
    @cuda.jit(device=True)
    def less(left, right):
        return left < right

    def compare(left, right):
        return less(left, right)

    @cuda.jit
    def kernel(source, observed):
        thread = cuda.threadIdx.x
        keys = numba_coop.ThreadData(1)
        keys[0] = source[thread]
        result = numba_coop.merge_sort_keys(
            numba_coop.this_block(), keys, compare_op=compare
        )
        observed[thread] = result[0]

    source = ((np.arange(64, dtype=np.int32) * 7) % 41) - 20
    observed = np.full_like(source, -1)
    kernel[1, 64](source, observed)
    np.testing.assert_array_equal(observed, np.sort(source))


def test_qualified_local_arrays_and_device_helper():
    from numba_cuda_mlir import types

    @cuda.jit(device=True, inline=True)
    def helper(keys):
        return numba_coop.merge_sort_keys(
            numba_coop.this_block(), keys, descending=True
        )

    @cuda.jit
    def kernel(source, observed, preserved):
        keys = cuda.local.array(2, types.float32)
        thread = cuda.threadIdx.x
        for item in range(2):
            keys[item] = source[thread * 2 + item]
        result = helper(keys)
        for item in range(2):
            observed[thread * 2 + item] = result[item]
            preserved[thread * 2 + item] = keys[item]

    source = np.arange(128, dtype=np.float32)
    observed = np.zeros_like(source)
    preserved = np.zeros_like(source)
    kernel[1, 64](source, observed, preserved)
    np.testing.assert_array_equal(observed, source[::-1])
    np.testing.assert_array_equal(preserved, source)


@pytest.mark.parametrize("count", [-1, 129, 1 << 32])
def test_invalid_runtime_count_traps(count):
    import subprocess
    import sys
    from pathlib import Path

    script = f"""
import numpy as np
from numba_cuda_mlir import cuda
from cuda import coop
from pathlib import Path
assert Path(coop.__file__).resolve() == Path({str(Path(coop.__file__).resolve())!r})
@cuda.jit
def kernel(source, output, count):
    keys = coop.ThreadData(2)
    thread = cuda.threadIdx.x
    for item in range(2):
        keys[item] = source[thread * 2 + item]
    result = coop.merge_sort_keys(coop.this_block(), keys, valid_items=count, oob_default=1000)
    for item in range(2):
        output[thread * 2 + item] = result[item]
source = np.arange(128, dtype=np.int32)
output = np.zeros_like(source)
kernel[1, 64](source, output, np.int64({count}))
cuda.synchronize()
raise AssertionError('invalid count did not trap')
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
        check=False,
        timeout=180,
    )
    output = result.stdout + result.stderr
    assert result.returncode != 0, output
    assert any(
        error in output
        for error in ("CUDA_ERROR_ILLEGAL_INSTRUCTION", "CUDA_ERROR_LAUNCH_FAILED")
    ), output


def test_static_partial_default_and_load_inference():
    @cuda.jit
    def kernel(source, observed):
        keys = coop.ThreadData(2)
        coop.load(coop.this_block(), source, keys)
        result = coop.merge_sort_keys(
            coop.this_block(), keys, valid_items=125, oob_default=127
        )
        coop.store(coop.this_block(), observed, result)

    source = np.arange(128, dtype=np.int16)[::-1].copy()
    observed = np.zeros_like(source)
    kernel[1, 64](source, observed)
    np.testing.assert_array_equal(observed[:125], np.sort(source[:125]))


@pytest.mark.parametrize("width", [64, 8, 32])
def test_partial_full_capacity(width):
    _run(width=width, partial=True, valid_count=width * 3)


@pytest.mark.parametrize("value_dtype", [np.int8, np.uint16, np.float32])
def test_pair_value_types(value_dtype):
    _run(value_dtype=value_dtype, partial=True)
