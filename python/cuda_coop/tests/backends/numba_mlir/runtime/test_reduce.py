# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

"""Hierarchy Reduce and Sum runtime qualification for Numba-CUDA-MLIR."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

cuda = pytest.importorskip("numba_cuda_mlir.cuda")
if not cuda.is_available():
    pytest.skip("requires a CUDA-capable runtime", allow_module_level=True)

from numba_cuda_mlir import types

import cuda.coop.numba_mlir as qualified_coop
from cuda import coop as root_coop

assert qualified_coop.__file__ is not None
_QUALIFIED_COOP_ORIGIN = Path(qualified_coop.__file__).resolve()
_SAFE_PATH_FLAG = "-P" if sys.version_info >= (3, 11) else "-I"
_COMPUTE_CAPABILITY = cuda.get_current_device().compute_capability
_HAS_THREAD_BLOCK_CLUSTERS = int(_COMPUTE_CAPABILITY[0]) >= 9

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
_WARPS_PER_MAPPED_GROUP = 2
_MAPPED_GROUP_THREADS = _WARPS_PER_MAPPED_GROUP * _WARP_THREADS
_ITEMS_PER_THREAD = 2
_STATIC_BLOCK_VALID = 73
_RUNTIME_WARP_VALID = 19
_RUNTIME_LOGICAL_VALID = 5
_CLUSTER_BLOCKS = 2
_CLUSTER_BLOCK_THREADS = 32
_HIERARCHY_RESULT_ROWS = 10


def _broadcast_grouped_sum(values: np.ndarray, width: int) -> np.ndarray:
    totals = values.reshape(-1, width).sum(axis=1, dtype=np.int32)
    return np.repeat(totals, width)


def _broadcast_grouped_maximum(values: np.ndarray, width: int) -> np.ndarray:
    maxima = values.reshape(-1, width).max(axis=1)
    return np.repeat(maxima, width)


@cuda.jit
def _hierarchy_scalar_reductions(source, observed):
    thread = cuda.threadIdx.x
    value = source[thread]
    logical_warp = root_coop.this_warp().group_by(_LOGICAL_WARP_THREADS)
    mapped_warps = root_coop.this_block().group_by(_WARPS_PER_MAPPED_GROUP)

    observed[0 * _BLOCK_THREADS + thread] = root_coop.sum(
        root_coop.this_thread(), value
    )
    observed[1 * _BLOCK_THREADS + thread] = qualified_coop.sum(
        qualified_coop.this_thread(), value
    )
    observed[2 * _BLOCK_THREADS + thread] = root_coop.reduce(
        root_coop.this_warp(), value, binary_op="max"
    )
    observed[3 * _BLOCK_THREADS + thread] = qualified_coop.reduce(
        qualified_coop.this_warp(), value, binary_op="max"
    )
    observed[4 * _BLOCK_THREADS + thread] = root_coop.sum(logical_warp, value)
    observed[5 * _BLOCK_THREADS + thread] = qualified_coop.sum(
        qualified_coop.this_warp().group_by(_LOGICAL_WARP_THREADS), value
    )
    observed[6 * _BLOCK_THREADS + thread] = root_coop.sum(root_coop.this_block(), value)
    observed[7 * _BLOCK_THREADS + thread] = qualified_coop.sum(
        qualified_coop.this_block(), value
    )
    observed[8 * _BLOCK_THREADS + thread] = root_coop.sum(mapped_warps, value)
    observed[9 * _BLOCK_THREADS + thread] = qualified_coop.sum(
        qualified_coop.this_block().group_by(_WARPS_PER_MAPPED_GROUP), value
    )


def test_both_namespaces_cover_thread_warp_block_and_mapped_scalar_reductions():
    source = ((np.arange(_BLOCK_THREADS, dtype=np.int32) * 7) % 41) - 20
    observed = np.full(
        _HIERARCHY_RESULT_ROWS * _BLOCK_THREADS,
        -1,
        dtype=np.int32,
    )

    _hierarchy_scalar_reductions[1, _BLOCK_THREADS](source, observed)

    expected = np.stack(
        (
            source,
            source,
            _broadcast_grouped_maximum(source, _WARP_THREADS),
            _broadcast_grouped_maximum(source, _WARP_THREADS),
            _broadcast_grouped_sum(source, _LOGICAL_WARP_THREADS),
            _broadcast_grouped_sum(source, _LOGICAL_WARP_THREADS),
            np.full(
                _BLOCK_THREADS,
                source.sum(dtype=np.int32),
                dtype=np.int32,
            ),
            np.full(
                _BLOCK_THREADS,
                source.sum(dtype=np.int32),
                dtype=np.int32,
            ),
            _broadcast_grouped_sum(source, _MAPPED_GROUP_THREADS),
            _broadcast_grouped_sum(source, _MAPPED_GROUP_THREADS),
        )
    )
    np.testing.assert_array_equal(
        observed.reshape(_HIERARCHY_RESULT_ROWS, _BLOCK_THREADS),
        expected,
    )


@cuda.jit
def _nonexhaustive_mapped_sum(source, observed, continued):
    thread = cuda.threadIdx.x
    mapped_warps = root_coop.this_block().group_by(3, exhaustive=False)
    observed[thread] = root_coop.sum(mapped_warps, source[thread])
    continued[thread] = source[thread] + 1


def test_nonexhaustive_mapped_reduce_guards_nonmembers_and_returns_normally():
    participating_threads = 3 * _WARP_THREADS
    source = np.arange(1, _BLOCK_THREADS + 1, dtype=np.int32)
    observed = np.full_like(source, -1)
    continued = np.full_like(source, -1)

    _nonexhaustive_mapped_sum[1, _BLOCK_THREADS](source, observed, continued)

    expected = np.zeros_like(source)
    expected[:participating_threads] = source[:participating_threads].sum(
        dtype=np.int32
    )
    np.testing.assert_array_equal(observed, expected)
    np.testing.assert_array_equal(continued, source + 1)


@cuda.jit
def _mixed_thread_data_builtins(source, observed, preserved):
    thread = cuda.threadIdx.x
    payload = root_coop.ThreadData(_ITEMS_PER_THREAD, dtype=types.int32)
    for item in range(_ITEMS_PER_THREAD):
        payload[item] = source[thread * _ITEMS_PER_THREAD + item]

    portable_sum = root_coop.sum(root_coop.this_block(), payload)
    qualified_maximum = qualified_coop.reduce(
        qualified_coop.this_block(), payload, binary_op="max"
    )
    qualified_xor = qualified_coop.reduce(
        qualified_coop.this_block(), payload, binary_op="bit_xor"
    )
    qualified_or = qualified_coop.reduce(
        qualified_coop.this_block(), payload, binary_op="bit_or"
    )
    qualified_scalar_minimum = qualified_coop.reduce(
        qualified_coop.this_block(), source[thread], binary_op="min"
    )

    observed[0 * _BLOCK_THREADS + thread] = portable_sum
    observed[1 * _BLOCK_THREADS + thread] = qualified_maximum
    observed[2 * _BLOCK_THREADS + thread] = qualified_xor
    observed[3 * _BLOCK_THREADS + thread] = qualified_or
    observed[4 * _BLOCK_THREADS + thread] = qualified_scalar_minimum
    for item in range(_ITEMS_PER_THREAD):
        preserved[thread * _ITEMS_PER_THREAD + item] = payload[item]


def test_consecutive_portable_and_qualified_builtins_preserve_thread_data():
    source = (
        (np.arange(_BLOCK_THREADS * _ITEMS_PER_THREAD, dtype=np.int32) * 13) % 251
    ) - 117
    observed = np.full(5 * _BLOCK_THREADS, -1, dtype=np.int32)
    preserved = np.full_like(source, -1)

    _mixed_thread_data_builtins[1, _BLOCK_THREADS](source, observed, preserved)

    expected = np.stack(
        (
            np.full(
                _BLOCK_THREADS,
                source.sum(dtype=np.int32),
                dtype=np.int32,
            ),
            np.full(_BLOCK_THREADS, source.max(), dtype=np.int32),
            np.full(
                _BLOCK_THREADS,
                np.bitwise_xor.reduce(source),
                dtype=np.int32,
            ),
            np.full(
                _BLOCK_THREADS,
                np.bitwise_or.reduce(source),
                dtype=np.int32,
            ),
            np.full(
                _BLOCK_THREADS,
                source[:_BLOCK_THREADS].min(),
                dtype=np.int32,
            ),
        )
    )
    np.testing.assert_array_equal(observed.reshape(5, _BLOCK_THREADS), expected)
    np.testing.assert_array_equal(preserved, source)


@cuda.jit
def _qualified_local_array_root_sum(source, output, preserved):
    thread = cuda.threadIdx.x
    payload = cuda.local.array(_ITEMS_PER_THREAD, dtype=types.int32)
    for item in range(_ITEMS_PER_THREAD):
        payload[item] = source[thread * _ITEMS_PER_THREAD + item]

    total = qualified_coop.sum(qualified_coop.this_block(), payload, broadcast=False)
    if thread == 0:
        output[0] = total
    for item in range(_ITEMS_PER_THREAD):
        preserved[thread * _ITEMS_PER_THREAD + item] = payload[item]


def test_qualified_local_array_reduction_returns_only_at_the_block_root():
    source = np.arange(
        1,
        _BLOCK_THREADS * _ITEMS_PER_THREAD + 1,
        dtype=np.int32,
    )
    output = np.full(1, -1, dtype=np.int32)
    preserved = np.full_like(source, -1)

    _qualified_local_array_root_sum[1, _BLOCK_THREADS](
        source,
        output,
        preserved,
    )

    assert output[0] == source.sum(dtype=np.int32)
    np.testing.assert_array_equal(preserved, source)


@cuda.jit
def _cluster_reductions(source, observed):
    thread = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    observed[thread] = root_coop.sum(root_coop.this_cluster(), source[thread])
    observed[source.size + thread] = qualified_coop.reduce(
        qualified_coop.this_cluster(), source[thread], binary_op="max"
    )


@pytest.mark.skipif(
    not _HAS_THREAD_BLOCK_CLUSTERS,
    reason="thread-block clusters require compute capability 9.0 or newer",
)
def test_both_namespaces_reduce_across_a_two_block_cluster():
    cluster_threads = _CLUSTER_BLOCKS * _CLUSTER_BLOCK_THREADS
    source = np.arange(1, cluster_threads + 1, dtype=np.int32)
    observed = np.full(2 * cluster_threads, -1, dtype=np.int32)

    configured = _cluster_reductions.configure(
        (_CLUSTER_BLOCKS, 1, 1),
        (_CLUSTER_BLOCK_THREADS, 1, 1),
        cluster=(_CLUSTER_BLOCKS, 1, 1),
    )
    configured(source, observed)

    expected = np.stack(
        (
            np.full_like(source, source.sum(dtype=np.int32)),
            np.full_like(source, source.max()),
        )
    )
    np.testing.assert_array_equal(observed.reshape(2, cluster_threads), expected)


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
        broadcast=False,
        valid_items=_STATIC_BLOCK_VALID,
    )
    if thread == 0:
        block_output[0] = block_total

    warp_maximum = qualified_coop.reduce(
        qualified_coop.this_warp(),
        source[thread],
        binary_op="max",
        broadcast=False,
        valid_items=warp_valid_items,
    )
    if thread % _WARP_THREADS == 0:
        warp_output[thread // _WARP_THREADS] = warp_maximum

    logical_total = root_coop.sum(
        root_coop.this_warp().group_by(_LOGICAL_WARP_THREADS),
        source[thread],
        broadcast=False,
        valid_items=logical_valid_items,
    )
    if thread % _LOGICAL_WARP_THREADS == 0:
        logical_output[thread // _LOGICAL_WARP_THREADS] = logical_total


def test_cub_static_and_runtime_prefixes_reduce_the_first_group_members():
    source = ((np.arange(_BLOCK_THREADS, dtype=np.int32) * 11) % 101) - 47
    block_output = np.full(1, -1, dtype=np.int32)
    warp_output = np.full(_BLOCK_THREADS // _WARP_THREADS, -1, dtype=np.int32)
    logical_output = np.full(
        _BLOCK_THREADS // _LOGICAL_WARP_THREADS,
        -1,
        dtype=np.int32,
    )

    _cub_valid_prefixes[1, _BLOCK_THREADS](
        source,
        block_output,
        warp_output,
        logical_output,
        np.uint32(_RUNTIME_WARP_VALID),
        np.int64(_RUNTIME_LOGICAL_VALID),
    )

    assert block_output[0] == source[:_STATIC_BLOCK_VALID].sum(dtype=np.int32)
    expected_warp = np.asarray(
        [
            values[:_RUNTIME_WARP_VALID].max()
            for values in source.reshape(-1, _WARP_THREADS)
        ],
        dtype=np.int32,
    )
    expected_logical = np.asarray(
        [
            values[:_RUNTIME_LOGICAL_VALID].sum(dtype=np.int32)
            for values in source.reshape(-1, _LOGICAL_WARP_THREADS)
        ],
        dtype=np.int32,
    )
    np.testing.assert_array_equal(warp_output, expected_warp)
    np.testing.assert_array_equal(logical_output, expected_logical)


@cuda.jit
def _cub_deterministic_algorithms(source, output, preserved):
    thread = cuda.threadIdx.x
    payload = root_coop.ThreadData(_ITEMS_PER_THREAD, dtype=types.int32)
    for item in range(_ITEMS_PER_THREAD):
        payload[item] = source[thread * _ITEMS_PER_THREAD + item]

    raking_sum = root_coop.sum(
        root_coop.this_block(),
        payload,
        broadcast=False,
        algorithm="raking",
    )
    warp_reductions_maximum = qualified_coop.reduce(
        qualified_coop.this_block(),
        source[thread],
        binary_op="max",
        broadcast=False,
        algorithm="warp_reductions",
    )
    commutative_xor = root_coop.reduce(
        root_coop.this_block(),
        source[thread],
        binary_op="bit_xor",
        broadcast=False,
        algorithm="raking_commutative_only",
    )

    if thread == 0:
        output[0] = raking_sum
        output[1] = warp_reductions_maximum
        output[2] = commutative_xor
    for item in range(_ITEMS_PER_THREAD):
        preserved[thread * _ITEMS_PER_THREAD + item] = payload[item]


def test_each_deterministic_block_algorithm_matches_an_independent_oracle():
    source = (
        (np.arange(_BLOCK_THREADS * _ITEMS_PER_THREAD, dtype=np.int32) * 17) % 257
    ) - 121
    output = np.full(3, -1, dtype=np.int32)
    preserved = np.full_like(source, -1)

    _cub_deterministic_algorithms[1, _BLOCK_THREADS](source, output, preserved)

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
    return left if left > right else right


_device_maximum = cuda.jit(device=True)(_maximum)


@cuda.jit
def _stateless_callback_reductions(
    source,
    block_output,
    logical_output,
    preserved,
):
    thread = cuda.threadIdx.x
    payload = cuda.local.array(_ITEMS_PER_THREAD, dtype=types.int32)
    for item in range(_ITEMS_PER_THREAD):
        payload[item] = source[thread * _ITEMS_PER_THREAD + item]

    block_maximum = qualified_coop.reduce(
        qualified_coop.this_block(),
        payload,
        binary_op=_device_maximum,
        broadcast=False,
    )
    if thread == 0:
        block_output[0] = block_maximum

    logical_maximum = qualified_coop.reduce(
        qualified_coop.this_warp().group_by(_LOGICAL_WARP_THREADS),
        source[thread * _ITEMS_PER_THREAD],
        binary_op=_device_maximum,
        broadcast=False,
        valid_items=_RUNTIME_LOGICAL_VALID,
    )
    if thread % _LOGICAL_WARP_THREADS == 0:
        logical_output[thread // _LOGICAL_WARP_THREADS] = logical_maximum

    for item in range(_ITEMS_PER_THREAD):
        preserved[thread * _ITEMS_PER_THREAD + item] = payload[item]


def test_qualified_callbacks_cover_block_arrays_and_logical_warp_prefixes():
    source = (
        (np.arange(_BLOCK_THREADS * _ITEMS_PER_THREAD, dtype=np.int32) * 29) % 313
    ) - 173
    block_output = np.full(1, -1, dtype=np.int32)
    logical_output = np.full(
        _BLOCK_THREADS // _LOGICAL_WARP_THREADS,
        -1,
        dtype=np.int32,
    )
    preserved = np.full_like(source, -1)

    _stateless_callback_reductions[1, _BLOCK_THREADS](
        source,
        block_output,
        logical_output,
        preserved,
    )

    logical_input = source[::_ITEMS_PER_THREAD].reshape(
        -1,
        _LOGICAL_WARP_THREADS,
    )
    expected_logical = logical_input[:, :_RUNTIME_LOGICAL_VALID].max(axis=1)
    assert block_output[0] == source.max()
    np.testing.assert_array_equal(logical_output, expected_logical)
    np.testing.assert_array_equal(preserved, source)


def _run_invalid_runtime_prefix_probe(
    group: str,
    valid_items: int,
) -> subprocess.CompletedProcess[str]:
    # A device trap poisons its CUDA context, so invalid launches must run in
    # disposable child processes rather than the pytest worker.
    group_expression = {
        "block": "root_coop.this_block()",
        "logical_warp": (f"root_coop.this_warp().group_by({_LOGICAL_WARP_THREADS})"),
    }[group]
    script = f"""\
import numpy as np
import numba_cuda_mlir.cuda as cuda
from pathlib import Path

import cuda.coop.numba_mlir as _coop_numba_mlir
from cuda import coop as root_coop

expected_origin = Path({str(_QUALIFIED_COOP_ORIGIN)!r})
actual_origin = Path(_coop_numba_mlir.__file__).resolve()
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
        broadcast=False,
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
        [sys.executable, _SAFE_PATH_FLAG, "-B", "-c", script],
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
    source = np.arange(_BLOCK_THREADS, dtype=np.int32)
    observed = np.full(
        _HIERARCHY_RESULT_ROWS * _BLOCK_THREADS,
        -1,
        dtype=np.int32,
    )
    _hierarchy_scalar_reductions[1, _BLOCK_THREADS](source, observed)
    assert observed[6 * _BLOCK_THREADS] == source.sum(dtype=np.int32)
