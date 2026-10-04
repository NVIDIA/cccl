# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check reduction values, input preservation, and group participation.

Host references keep the payload dtype and group boundaries explicit.
Root-only results are observed only at the group root. Callback tests also
check that captured values affect compiled-code reuse. Out-of-range
``valid_items`` probes isolate their device traps in child processes.
"""

from __future__ import annotations

import math
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
from cuda import coop as root_coop

assert numba_coop.__file__ is not None
_QUALIFIED_COOP_ORIGIN = Path(numba_coop.__file__).resolve()
_SAFE_PATH_FLAG = "-P" if sys.version_info >= (3, 11) else "-I"

pytestmark = [
    pytest.mark.backend_numba_mlir,
    pytest.mark.runtime,
    pytest.mark.gpu,
    pytest.mark.filterwarnings(
        "ignore::numba_cuda_mlir.numba_cuda.core.errors.NumbaPerformanceWarning"
    ),
]

_BLOCK_THREADS = 128
_WARP_THREADS = 32
_LOGICAL_WARP_THREADS = 8
_ITEMS_PER_THREAD = 2
_STATIC_BLOCK_VALID = 73
_RUNTIME_WARP_VALID = 19
_RUNTIME_LOGICAL_VALID = 5
_INTEGER_DTYPES = (
    np.int8,
    np.uint8,
    np.int16,
    np.uint16,
    np.int32,
    np.uint32,
    np.int64,
    np.uint64,
)
_DTYPES = (*_INTEGER_DTYPES, np.float32, np.float64)


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("qualified", [False, True])
@pytest.mark.parametrize("width", [8, 32, 128])
def test_default_scalar_reductions_return_group_root_values(
    dtype, qualified, width
):
    module = numba_coop if qualified else root_coop

    @cuda.jit
    def kernel(source, sums, maxima):
        thread = cuda.threadIdx.x
        if width == 128:
            group = module.this_block()
        elif width == 32:
            group = module.this_warp()
        else:
            group = module.this_warp().group_by(width)
        total = module.sum(group, source[thread])
        maximum = module.reduce(group, source[thread], binary_op="max")
        if thread % width == 0:
            sums[thread // width] = total
            maxima[thread // width] = maximum

    source = _dtype_values(dtype, _BLOCK_THREADS)
    sums = np.zeros(_BLOCK_THREADS // width, dtype=dtype)
    maxima = np.zeros_like(sums)
    kernel[1, _BLOCK_THREADS](source, sums, maxima)
    grouped = source.reshape(-1, width)
    np.testing.assert_array_equal(sums, grouped.sum(axis=1, dtype=dtype))
    np.testing.assert_array_equal(maxima, grouped.max(axis=1))


@pytest.mark.parametrize("items_per_thread", [1, 4])
@pytest.mark.parametrize("qualified", [False, True])
@pytest.mark.parametrize("width", [1, 8, 17, 31, 32])
@pytest.mark.parametrize("dtype", [np.int32, np.float64])
def test_warp_arrays_reduce_all_items_and_reuse_scratch(
    items_per_thread, qualified, width, dtype
):
    module = numba_coop if qualified else root_coop
    groups_per_warp = 32 // width
    groups_per_block = (_BLOCK_THREADS // 32) * groups_per_warp

    @cuda.jit
    def kernel(source, sums, maxima, preserved, items_per_thread):
        thread = cuda.threadIdx.x
        if width == 32:
            group = module.this_warp()
        else:
            group = module.this_warp().group_by(width, exhaustive=False)
        payload = module.ThreadData(items_per_thread)
        for tile in range(3):
            offset = (tile * cuda.blockDim.x + thread) * items_per_thread
            for item in range(items_per_thread):
                payload[item] = source[offset + item]
            if group.is_member():
                total = module.sum(group, payload)
                maximum = module.reduce(group, payload, binary_op="max")
                if group.rank() == 0:
                    instance = (thread // 32) * groups_per_warp + (
                        thread % 32
                    ) // width
                    sums[tile, instance] = total
                    maxima[tile, instance] = maximum
            for item in range(items_per_thread):
                preserved[offset + item] = payload[item]

    source = _dtype_values(dtype, 3 * _BLOCK_THREADS * items_per_thread)
    sums = np.zeros((3, groups_per_block), dtype=dtype)
    maxima = np.zeros_like(sums)
    preserved = np.zeros_like(source)
    kernel[1, _BLOCK_THREADS](source, sums, maxima, preserved, items_per_thread)
    physical_warps = source.reshape(3, -1, 32, items_per_thread)
    participating = physical_warps[:, :, : groups_per_warp * width, :]
    grouped = participating.reshape(
        3, groups_per_block, width * items_per_thread
    )
    np.testing.assert_array_equal(sums, grouped.sum(axis=2, dtype=dtype))
    np.testing.assert_array_equal(maxima, grouped.max(axis=2))
    np.testing.assert_array_equal(preserved, source)


@pytest.mark.parametrize("width", [17, 31])
@pytest.mark.parametrize("qualified", [False, True])
def test_nonexhaustive_warp_scalar_prefixes(width, qualified):
    module = numba_coop if qualified else root_coop
    groups_per_warp = 32 // width
    groups_per_block = (_BLOCK_THREADS // 32) * groups_per_warp

    @cuda.jit
    def kernel(source, sums, maxima, valid_items):
        thread = cuda.threadIdx.x
        group = module.this_warp().group_by(width, exhaustive=False)
        if group.is_member():
            for tile in range(3):
                value = source[tile * cuda.blockDim.x + thread]
                total = module.sum(group, value, valid_items=valid_items)
                maximum = module.reduce(
                    group, value, binary_op="max", valid_items=width - 1
                )
                if group.rank() == 0:
                    instance = (thread // 32) * groups_per_warp + (
                        thread % 32
                    ) // width
                    sums[tile, instance] = total
                    maxima[tile, instance] = maximum

    source = np.arange(3 * _BLOCK_THREADS, dtype=np.int32) % 37 - 17
    sums = np.zeros((3, groups_per_block), dtype=np.int32)
    maxima = np.zeros_like(sums)
    kernel[1, _BLOCK_THREADS](source, sums, maxima, np.int64(width - 1))
    physical_warps = source.reshape(3, -1, 32)
    participating = physical_warps[:, :, : groups_per_warp * width]
    grouped = participating.reshape(3, groups_per_block, width)
    np.testing.assert_array_equal(
        sums, grouped[:, :, :-1].sum(axis=2, dtype=np.int32)
    )
    np.testing.assert_array_equal(maxima, grouped[:, :, :-1].max(axis=2))


@pytest.mark.parametrize("items_per_thread", [1, 4])
@pytest.mark.parametrize("qualified", [False, True])
@pytest.mark.parametrize(
    ("sharing", "auto_sync", "size_in_bytes"),
    [
        ("shared", True, None),
        ("shared", False, None),
        ("exclusive", False, None),
        ("shared", True, 64 * 1024),
    ],
    ids=["shared-auto", "shared-manual", "exclusive-manual", "dynamic"],
)
def test_reductions_reuse_explicit_scratch_with_load(
    items_per_thread, qualified, sharing, auto_sync, size_in_bytes
):
    module = numba_coop if qualified else root_coop

    @cuda.jit
    def kernel(source, sums, maxima, preserved, items_per_thread):
        block = module.this_block()
        scratch = module.TempStorage(
            size_in_bytes, sharing=sharing, auto_sync=auto_sync
        )
        items = module.ThreadData(items_per_thread)
        for tile in range(2):
            module.load(
                block,
                source,
                items,
                offset=tile * cuda.blockDim.x * items_per_thread,
                algorithm="transpose",
                temp_storage=scratch,
            )
            if not auto_sync:
                block.sync()
            total = module.sum(block, items, temp_storage=scratch)
            if not auto_sync:
                block.sync()
            maximum = module.reduce(
                block, items, binary_op="max", temp_storage=scratch
            )
            if not auto_sync:
                block.sync()
            module.store(
                block,
                preserved,
                items,
                offset=tile * cuda.blockDim.x * items_per_thread,
                algorithm="transpose",
                temp_storage=scratch,
            )
            if not auto_sync:
                block.sync()
            if cuda.threadIdx.x == 0:
                sums[tile] = total
                maxima[tile] = maximum

    source = (
        np.arange(2 * _BLOCK_THREADS * items_per_thread, dtype=np.int32) % 17
    )
    sums = np.zeros(2, dtype=np.int32)
    maxima = np.zeros_like(sums)
    preserved = np.zeros_like(source)
    kernel[1, _BLOCK_THREADS](source, sums, maxima, preserved, items_per_thread)
    grouped = source.reshape(2, -1)
    np.testing.assert_array_equal(sums, grouped.sum(axis=1, dtype=np.int32))
    np.testing.assert_array_equal(maxima, grouped.max(axis=1))
    np.testing.assert_array_equal(preserved, source)
    if size_in_bytes:
        compiled = next(iter(kernel._launch_config_overloads.values()))
        assert (
            compiled.metadata["required_dynamic_shared_memory"] >= size_in_bytes
        )


@pytest.mark.parametrize("dtype", [np.int32, np.float32])
@pytest.mark.parametrize("scope", ["warp", "block"])
@pytest.mark.parametrize("collectives", [1, 2])
def test_sum_infers_loop_carried_scalar_dtype(dtype, scope, collectives):
    @cuda.jit
    def kernel(source, observed, repetitions):
        thread = cuda.threadIdx.x
        value = source[thread]
        if scope == "warp":
            group = root_coop.this_warp()
            leader = thread % 32 == 0
        else:
            group = root_coop.this_block()
            leader = thread == 0
        for _ in range(repetitions):
            total = root_coop.sum(group, value)
            cuda.syncthreads()
            if leader:
                value = total
            if collectives == 2:
                total = root_coop.sum(group, value)
                cuda.syncthreads()
                if leader:
                    value = total
        observed[thread] = value

    width = 32 if scope == "warp" else 64
    source = np.arange(1, 65, dtype=dtype)
    observed = cuda.to_device(np.zeros_like(source))
    repetitions = 5
    kernel[1, 64](cuda.to_device(source), observed, np.int32(repetitions))
    expected = source.copy().reshape(-1, width)
    expected[:, 0] += repetitions * collectives * expected[:, 1:].sum(axis=1)
    np.testing.assert_array_equal(observed.copy_to_host(), expected.ravel())


def _dtype_values(dtype, size: int) -> np.ndarray:
    """Choose patterns that reveal lost payload bits during reduction.

    Float64 values include a fraction lost in float32. Wider integer patterns
    use bits beyond the next smaller width. Host references explicitly reduce
    in the payload dtype, including its integer wraparound behavior.
    """

    indices = np.arange(size, dtype=np.int64)
    if np.dtype(dtype).kind == "u":
        values = (indices % 3 == 0).astype(dtype)
    elif np.dtype(dtype).kind == "f":
        values = ((indices % 5) - 2).astype(dtype) * dtype(0.25)
    else:
        values = ((indices % 3) - 1).astype(dtype)
    if np.dtype(dtype).kind == "f":
        if np.dtype(dtype).itemsize == 8:
            values += dtype(2**-30)
    elif np.dtype(dtype).itemsize > 1:
        scale = {2: 257, 4: 65537, 8: 2**33 + 1}[np.dtype(dtype).itemsize]
        values *= dtype(scale)
    return values


@cuda.jit
def _mixed_thread_data_builtins(source, observed, preserved, items_per_thread):
    thread = cuda.threadIdx.x
    payload = root_coop.ThreadData(items_per_thread)
    for item in range(items_per_thread):
        payload[item] = source[thread * items_per_thread + item]

    common_sum = root_coop.sum(root_coop.this_block(), payload)
    qualified_maximum = numba_coop.reduce(
        numba_coop.this_block(), payload, binary_op="max"
    )
    qualified_xor = numba_coop.reduce(
        numba_coop.this_block(), payload, binary_op="bit_xor"
    )
    qualified_or = numba_coop.reduce(
        numba_coop.this_block(), payload, binary_op="bit_or"
    )
    qualified_scalar_minimum = numba_coop.reduce(
        numba_coop.this_block(), source[thread], binary_op="min"
    )

    if thread == 0:
        observed[0] = common_sum
        observed[1] = qualified_maximum
        observed[2] = qualified_xor
        observed[3] = qualified_or
        observed[4] = qualified_scalar_minimum
    for item in range(items_per_thread):
        preserved[thread * items_per_thread + item] = payload[item]


@pytest.mark.parametrize("items_per_thread", [1, 4])
@pytest.mark.parametrize("dtype", _INTEGER_DTYPES)
def test_consecutive_common_and_qualified_builtins_preserve_thread_data(
    dtype, *, items_per_thread
):
    source = _dtype_values(dtype, _BLOCK_THREADS * items_per_thread)
    observed = np.full(5, 127, dtype=dtype)
    preserved = np.full_like(source, 127)

    _mixed_thread_data_builtins[1, _BLOCK_THREADS](
        source, observed, preserved, items_per_thread
    )

    expected = np.array(
        [
            source.sum(dtype=dtype),
            source.max(),
            np.bitwise_xor.reduce(source, dtype=dtype),
            np.bitwise_or.reduce(source, dtype=dtype),
            source[:_BLOCK_THREADS].min(),
        ],
        dtype=dtype,
    )
    np.testing.assert_array_equal(observed, expected)
    np.testing.assert_array_equal(preserved, source)


@cuda.jit
def _qualified_local_array_root_sum(
    source, output, preserved, items_per_thread
):
    thread = cuda.threadIdx.x
    payload = cuda.local.array(items_per_thread, dtype=source.dtype)
    for item in range(items_per_thread):
        payload[item] = source[thread * items_per_thread + item]

    total = numba_coop.sum(numba_coop.this_block(), payload)
    if thread == 0:
        output[0] = total
    for item in range(items_per_thread):
        preserved[thread * items_per_thread + item] = payload[item]


@pytest.mark.parametrize("items_per_thread", [1, 4])
@pytest.mark.parametrize("dtype", _DTYPES)
def test_qualified_local_array_reduction_returns_only_at_the_block_root(
    dtype, *, items_per_thread
):
    source = _dtype_values(dtype, _BLOCK_THREADS * items_per_thread)
    output = np.full(1, 127, dtype=dtype)
    preserved = np.full_like(source, 127)

    _qualified_local_array_root_sum[1, _BLOCK_THREADS](
        source, output, preserved, items_per_thread
    )

    assert output[0] == source.sum(dtype=dtype)
    np.testing.assert_array_equal(preserved, source)


@cuda.jit
def _cub_valid_prefixes(
    source,
    block_output,
    warp_output,
    logical_output,
    warp_valid_items,
    logical_valid_items,
):
    thread = cuda.threadIdx.x
    block_total = root_coop.sum(
        root_coop.this_block(),
        source[thread],
        valid_items=_STATIC_BLOCK_VALID,
    )
    if thread == 0:
        block_output[0] = block_total

    warp_maximum = numba_coop.reduce(
        numba_coop.this_warp(),
        source[thread],
        binary_op="max",
        valid_items=warp_valid_items,
    )
    if thread % _WARP_THREADS == 0:
        warp_output[thread // _WARP_THREADS] = warp_maximum

    logical_total = root_coop.sum(
        root_coop.this_warp().group_by(_LOGICAL_WARP_THREADS),
        source[thread],
        valid_items=logical_valid_items,
    )
    if thread % _LOGICAL_WARP_THREADS == 0:
        logical_output[thread // _LOGICAL_WARP_THREADS] = logical_total


@pytest.mark.parametrize("dtype", _DTYPES)
def test_cub_static_and_runtime_prefixes_reduce_the_first_group_members(dtype):
    source = _dtype_values(dtype, _BLOCK_THREADS)
    if dtype is np.int32:
        source = ((np.arange(_BLOCK_THREADS, dtype=np.int32) * 11) % 101) - 47
    block_output = np.full(1, 127, dtype=dtype)
    warp_output = np.full(_BLOCK_THREADS // _WARP_THREADS, 127, dtype=dtype)
    logical_output = np.full(
        _BLOCK_THREADS // _LOGICAL_WARP_THREADS,
        127,
        dtype=dtype,
    )

    _cub_valid_prefixes[1, _BLOCK_THREADS](
        source,
        block_output,
        warp_output,
        logical_output,
        np.uint32(_RUNTIME_WARP_VALID),
        np.int64(_RUNTIME_LOGICAL_VALID),
    )

    assert block_output[0] == source[:_STATIC_BLOCK_VALID].sum(dtype=dtype)
    expected_warp = np.asarray(
        [
            values[:_RUNTIME_WARP_VALID].max()
            for values in source.reshape(-1, _WARP_THREADS)
        ],
        dtype=dtype,
    )
    expected_logical = np.asarray(
        [
            values[:_RUNTIME_LOGICAL_VALID].sum(dtype=dtype)
            for values in source.reshape(-1, _LOGICAL_WARP_THREADS)
        ],
        dtype=dtype,
    )
    np.testing.assert_array_equal(warp_output, expected_warp)
    np.testing.assert_array_equal(logical_output, expected_logical)


@cuda.jit
def _cub_deterministic_algorithms(source, output, preserved, items_per_thread):
    thread = cuda.threadIdx.x
    payload = root_coop.ThreadData(items_per_thread, dtype=types.int32)
    for item in range(items_per_thread):
        payload[item] = source[thread * items_per_thread + item]

    raking_sum = root_coop.sum(
        root_coop.this_block(),
        payload,
        algorithm="raking",
    )
    warp_reductions_maximum = numba_coop.reduce(
        numba_coop.this_block(),
        source[thread],
        binary_op="max",
        algorithm="warp_reductions",
    )
    commutative_xor = root_coop.reduce(
        root_coop.this_block(),
        source[thread],
        binary_op="bit_xor",
        algorithm="raking_commutative_only",
    )

    if thread == 0:
        output[0] = raking_sum
        output[1] = warp_reductions_maximum
        output[2] = commutative_xor
    for item in range(items_per_thread):
        preserved[thread * items_per_thread + item] = payload[item]


@pytest.mark.parametrize("items_per_thread", [1, 4])
def test_each_deterministic_block_algorithm_matches_an_independent_oracle(
    *, items_per_thread
):
    source = (
        (np.arange(_BLOCK_THREADS * items_per_thread, dtype=np.int32) * 17)
        % 257
    ) - 121
    output = np.full(3, -1, dtype=np.int32)
    preserved = np.full_like(source, -1)

    _cub_deterministic_algorithms[1, _BLOCK_THREADS](
        source, output, preserved, items_per_thread
    )

    expected = np.asarray(
        (
            source.sum(dtype=np.int32),
            source[:_BLOCK_THREADS].max(),
            np.bitwise_xor.reduce(source[:_BLOCK_THREADS]),
        ),
        dtype=np.int32,
    )
    np.testing.assert_array_equal(output, expected)
    np.testing.assert_array_equal(preserved, source)


def _maximum(left, right):
    return max(right, left)


_device_maximum = cuda.jit(device=True)(_maximum)


def test_distinct_constant_arrays_do_not_reuse_a_cached_callback():
    def make_kernel(offset):
        table = np.zeros(2000, dtype=np.int32)
        table[1000] = offset

        @cuda.jit(device=True)
        def add(left, right):
            return left + right + table[1000]

        @cuda.jit
        def kernel(source, observed):
            thread = cuda.threadIdx.x
            result = numba_coop.reduce(
                numba_coop.this_block(),
                source[thread],
                binary_op=add,
            )
            if thread == 0:
                observed[0] = result

        return kernel

    source = np.ones(_BLOCK_THREADS, dtype=np.int32)
    # Both callbacks compile in one process. Their arrays have identical
    # truncated representations, but differ in a captured constant element.
    for offset in (0, 1):
        observed = np.full(1, -1, dtype=np.int32)
        make_kernel(offset)[1, _BLOCK_THREADS](source, observed)
        expected = _BLOCK_THREADS + (_BLOCK_THREADS - 1) * offset
        np.testing.assert_array_equal(
            observed, np.full_like(observed, expected)
        )


def test_numpy_scalar_nan_sign_does_not_reuse_a_cached_callback():
    def make_kernel(captured):
        @cuda.jit(device=True)
        def add(left, right):
            return left + right + math.copysign(1.0, captured)

        @cuda.jit
        def kernel(source, observed):
            thread = cuda.threadIdx.x
            result = numba_coop.reduce(
                numba_coop.this_block(),
                source[thread],
                binary_op=add,
            )
            if thread == 0:
                observed[0] = result

        return kernel

    source = np.ones(_BLOCK_THREADS, dtype=np.float64)
    for sign in (1, -1):
        captured = np.copysign(np.float32("nan"), np.float32(sign))
        observed = np.full(1, -1, dtype=np.float64)
        make_kernel(captured)[1, _BLOCK_THREADS](source, observed)
        expected = _BLOCK_THREADS + (_BLOCK_THREADS - 1) * sign
        np.testing.assert_array_equal(
            observed, np.full_like(observed, expected)
        )


@pytest.mark.parametrize("inline", [True, False])
def test_qualified_reduce_accepts_a_callback_with_a_nested_device_helper(
    inline,
):
    helper = cuda.jit(device=True, inline=inline)(_maximum)

    @cuda.jit(device=True)
    def maximum(left, right):
        return helper(left, right)

    @cuda.jit
    def kernel(source, observed):
        thread = cuda.threadIdx.x
        value = source[thread]
        result = numba_coop.reduce(
            numba_coop.this_block(),
            value,
            binary_op=maximum,
        )
        if thread == 0:
            observed[0] = result

    source = ((np.arange(_BLOCK_THREADS, dtype=np.int32) * 7) % 41) - 20
    observed = np.full(1, -1, dtype=np.int32)
    kernel[1, _BLOCK_THREADS](source, observed)
    np.testing.assert_array_equal(
        observed, np.full_like(observed, source.max())
    )


@cuda.jit
def _stateless_callback_reductions(
    source,
    block_output,
    warp_output,
    logical_output,
    preserved,
    items_per_thread,
):
    thread = cuda.threadIdx.x
    payload = cuda.local.array(items_per_thread, dtype=types.int32)
    for item in range(items_per_thread):
        payload[item] = source[thread * items_per_thread + item]

    block_maximum = numba_coop.reduce(
        numba_coop.this_block(),
        payload,
        binary_op=_device_maximum,
    )
    if thread == 0:
        block_output[0] = block_maximum

    warp_maximum = numba_coop.reduce(
        numba_coop.this_warp(), payload, binary_op=_device_maximum
    )
    if thread % _WARP_THREADS == 0:
        warp_output[thread // _WARP_THREADS] = warp_maximum

    logical_maximum = numba_coop.reduce(
        numba_coop.this_warp().group_by(_LOGICAL_WARP_THREADS),
        source[thread * items_per_thread],
        binary_op=_device_maximum,
        valid_items=_RUNTIME_LOGICAL_VALID,
    )
    if thread % _LOGICAL_WARP_THREADS == 0:
        logical_output[thread // _LOGICAL_WARP_THREADS] = logical_maximum

    for item in range(items_per_thread):
        preserved[thread * items_per_thread + item] = payload[item]


@pytest.mark.parametrize("items_per_thread", [1, 4])
def test_qualified_callbacks_cover_group_arrays_and_logical_warp_prefixes(
    *, items_per_thread
):
    source = (
        (np.arange(_BLOCK_THREADS * items_per_thread, dtype=np.int32) * 29)
        % 313
    ) - 173
    block_output = np.full(1, -1, dtype=np.int32)
    warp_output = np.full(_BLOCK_THREADS // _WARP_THREADS, -1, dtype=np.int32)
    logical_output = np.full(
        _BLOCK_THREADS // _LOGICAL_WARP_THREADS,
        -1,
        dtype=np.int32,
    )
    preserved = np.full_like(source, -1)

    _stateless_callback_reductions[1, _BLOCK_THREADS](
        source,
        block_output,
        warp_output,
        logical_output,
        preserved,
        items_per_thread,
    )

    logical_input = source[::items_per_thread].reshape(
        -1,
        _LOGICAL_WARP_THREADS,
    )
    expected_logical = logical_input[:, :_RUNTIME_LOGICAL_VALID].max(axis=1)
    assert block_output[0] == source.max()
    np.testing.assert_array_equal(
        warp_output,
        source.reshape(-1, _WARP_THREADS * items_per_thread).max(axis=1),
    )
    np.testing.assert_array_equal(logical_output, expected_logical)
    np.testing.assert_array_equal(preserved, source)


def _run_invalid_runtime_prefix_probe(
    group: str,
    valid_items: int,
) -> subprocess.CompletedProcess[str]:
    """Run an out-of-range ``valid_items`` count in a separate CUDA context.

    A device trap poisons its context, so the invalid call must run outside
    the pytest worker. The child checks its package origin against the
    parent's imported source. Preserve the parent's import path without
    running editable-install startup hooks that could select another checkout.
    The caller inspects the failure text to require a device trap, then runs a
    valid reduction in the parent to check that its context remains usable.
    """

    group_expression = {
        "block": "root_coop.this_block()",
        "logical_warp": (
            f"root_coop.this_warp().group_by({_LOGICAL_WARP_THREADS})"
        ),
    }[group]
    script = f"""\
import sys
sys.path[:] = {sys.path!r}
import numpy as np
import numba_cuda_mlir.cuda as cuda
from pathlib import Path

import cuda.coop.numba_mlir as numba_coop
from cuda import coop as root_coop

expected_origin = Path({str(_QUALIFIED_COOP_ORIGIN)!r})
actual_origin = Path(numba_coop.__file__).resolve()
if actual_origin != expected_origin:
    raise RuntimeError(
        f"trap probe imported cuda.coop from {{actual_origin}}, "
        f"expected {{expected_origin}}"
    )

_THREADS = {_BLOCK_THREADS}

@cuda.jit
def kernel(source, output, valid_items):
    thread = cuda.threadIdx.x
    total = root_coop.sum(
        {group_expression},
        source[thread],
        valid_items=valid_items,
    )
    if thread == 0:
        output[0] = total

source = np.arange(_THREADS, dtype=np.int32)
output = np.full(1, -1, dtype=np.int32)
kernel[1, _THREADS](source, output, np.int64({valid_items}))
cuda.synchronize()
raise AssertionError("invalid Reduce valid_items did not trap")
"""
    return subprocess.run(
        [sys.executable, _SAFE_PATH_FLAG, "-S", "-B", "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )


@pytest.mark.parametrize(
    ("group", "valid_items"),
    (
        pytest.param("block", -1, id="block-negative"),
        pytest.param(
            "logical_warp",
            _LOGICAL_WARP_THREADS + 1,
            id="logical-warp-beyond-width",
        ),
    ),
)
def test_invalid_runtime_prefix_traps_in_an_isolated_context(
    group: str,
    valid_items: int,
) -> None:
    result = _run_invalid_runtime_prefix_probe(group, valid_items)
    output = result.stdout + result.stderr

    assert result.returncode != 0, output
    assert any(
        error in output
        for error in (
            "CUDA_ERROR_ILLEGAL_INSTRUCTION",
            "CUDA_ERROR_LAUNCH_FAILED",
        )
    ), output

    # Prove that the pytest worker's independent context remains usable.
    @cuda.jit
    def valid_sum(source, observed):
        total = root_coop.sum(root_coop.this_block(), source[cuda.threadIdx.x])
        if cuda.threadIdx.x == 0:
            observed[0] = total

    source = np.arange(_BLOCK_THREADS, dtype=np.int32)
    observed = np.full(1, -1, dtype=np.int32)
    valid_sum[1, _BLOCK_THREADS](source, observed)
    assert observed[0] == source.sum(dtype=np.int32)
