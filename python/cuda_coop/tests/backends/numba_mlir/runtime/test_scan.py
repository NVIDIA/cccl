# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

"""Block and Warp Scan runtime qualification for Numba-CUDA-MLIR."""

from __future__ import annotations

import operator
import subprocess
import sys
from functools import cache
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

pytestmark = [
    pytest.mark.backend_numba_mlir,
    pytest.mark.runtime,
    pytest.mark.gpu,
    pytest.mark.filterwarnings(
        "ignore::numba_cuda_mlir.numba_cuda.core.errors.NumbaPerformanceWarning"
    ),
]

_BLOCK_THREADS = 64
_WARP_THREADS = 32
_LOGICAL_WARP_THREADS = 8
_ITEMS_PER_THREAD = 2
_TILE_ITEMS = _BLOCK_THREADS * _ITEMS_PER_THREAD
_RUNTIME_VALID_ITEMS = 5
_PREFIX_TILE_COUNT = 4
_PREFIX_INITIAL_STATE = 17
_DYNAMIC_STORAGE_BYTES = 64 * 1024


def _exclusive_sum(values: np.ndarray, initial: int = 0) -> np.ndarray:
    result = np.empty_like(values)
    result[0] = initial
    result[1:] = initial + np.cumsum(values[:-1], dtype=values.dtype)
    return result


@cuda.jit
def _five_scan_spellings(source, output, aggregates, initial):
    thread = cuda.threadIdx.x
    value = source[thread]
    aggregate = qualified_coop.ThreadData(1)

    output[0 * _BLOCK_THREADS + thread] = root_coop.scan(root_coop.this_block(), value)
    output[1 * _BLOCK_THREADS + thread] = qualified_coop.exclusive_scan(
        qualified_coop.this_block(),
        value,
        initial_value=initial,
        aggregate_output=aggregate,
    )
    output[2 * _BLOCK_THREADS + thread] = qualified_coop.inclusive_scan(
        qualified_coop.this_block(), value, scan_op="max"
    )
    output[3 * _BLOCK_THREADS + thread] = root_coop.exclusive_sum(
        root_coop.this_block(), value
    )
    output[4 * _BLOCK_THREADS + thread] = root_coop.inclusive_sum(
        root_coop.this_block(), value
    )
    aggregates[thread] = aggregate[0]


def test_all_five_spellings_preserve_mode_initial_and_aggregate_semantics():
    source = ((np.arange(_BLOCK_THREADS, dtype=np.int32) * 7) % 29) + 1
    output = np.full(5 * _BLOCK_THREADS, -1, dtype=np.int32)
    aggregates = np.full(_BLOCK_THREADS, -1, dtype=np.int32)
    initial = np.int32(11)

    _five_scan_spellings[1, _BLOCK_THREADS](source, output, aggregates, initial)

    expected_exclusive = _exclusive_sum(source)
    expected = np.stack(
        (
            expected_exclusive,
            _exclusive_sum(source, int(initial)),
            np.maximum.accumulate(source),
            expected_exclusive,
            np.cumsum(source, dtype=np.int32),
        )
    )
    np.testing.assert_array_equal(output.reshape(5, _BLOCK_THREADS), expected)
    np.testing.assert_array_equal(
        aggregates,
        np.full(_BLOCK_THREADS, source.sum(dtype=np.int32), dtype=np.int32),
    )


@cache
def _thread_data_algorithm_kernel(algorithm: str):
    @cuda.jit
    def kernel(source, output, preserved):
        thread = cuda.threadIdx.x
        value = root_coop.ThreadData(_ITEMS_PER_THREAD)
        for item in range(_ITEMS_PER_THREAD):
            index = thread * _ITEMS_PER_THREAD + item
            value[item] = source[index]
        scanned = root_coop.inclusive_sum(
            root_coop.this_block(), value, algorithm=algorithm
        )
        root_coop.store(root_coop.this_block(), output, scanned)
        for item in range(_ITEMS_PER_THREAD):
            index = thread * _ITEMS_PER_THREAD + item
            preserved[index] = value[item]

    return kernel


@pytest.mark.parametrize("algorithm", ("raking", "raking_memoize", "warp_scans"))
def test_block_algorithms_scan_thread_data_out_of_place(algorithm: str):
    source = ((np.arange(_TILE_ITEMS, dtype=np.int32) * 5) % 37) - 11
    output = np.full_like(source, -1)
    preserved = np.full_like(source, -1)

    _thread_data_algorithm_kernel(algorithm)[1, _BLOCK_THREADS](
        source, output, preserved
    )

    np.testing.assert_array_equal(output, np.cumsum(source, dtype=np.int32))
    np.testing.assert_array_equal(preserved, source)


@cuda.jit
def _local_array_numpy_scan(source, output, preserved, aggregates):
    thread = cuda.threadIdx.x
    value = cuda.local.array(_ITEMS_PER_THREAD, dtype=types.int32)
    aggregate = cuda.local.array(1, dtype=types.int32)
    for item in range(_ITEMS_PER_THREAD):
        index = thread * _ITEMS_PER_THREAD + item
        value[item] = source[index]
    scanned = qualified_coop.inclusive_scan(
        qualified_coop.this_block(),
        value,
        scan_op=np.maximum,
        algorithm="raking_memoize",
        aggregate_output=aggregate,
    )
    for item in range(_ITEMS_PER_THREAD):
        index = thread * _ITEMS_PER_THREAD + item
        output[index] = scanned[item]
        preserved[index] = value[item]
    aggregates[thread] = aggregate[0]


def test_qualified_local_array_and_numpy_ufunc_preserve_input_and_aggregate():
    source = ((np.arange(_TILE_ITEMS, dtype=np.int32) * 17) % 113) - 51
    output = np.full_like(source, -1)
    preserved = np.full_like(source, -1)
    aggregates = np.full(_BLOCK_THREADS, -1, dtype=np.int32)

    _local_array_numpy_scan[1, _BLOCK_THREADS](source, output, preserved, aggregates)

    np.testing.assert_array_equal(output, np.maximum.accumulate(source))
    np.testing.assert_array_equal(preserved, source)
    np.testing.assert_array_equal(
        aggregates,
        np.full(_BLOCK_THREADS, source.max(), dtype=np.int32),
    )


def _maximum(left, right):
    return left if left > right else right  # noqa: FURB136


_device_maximum = cuda.jit(device=True)(_maximum)


@cuda.jit(device=True)
def _prefix_after_block_aggregate(block_aggregate):
    return block_aggregate + 7


@cuda.jit(device=True)
def _running_prefix_int64(state, block_aggregate):
    previous = state[0]
    state[0] = previous + block_aggregate
    return previous


_RUNNING_PREFIX_INT64 = qualified_coop.StatefulFunction(
    _running_prefix_int64,
    types.int64,
    name="cuda_coop_test_running_prefix_int64",
)


@cuda.jit
def _block_scan_prefix_aliases(source, canonical, compatibility):
    thread = cuda.threadIdx.x
    values = cuda.local.array(_ITEMS_PER_THREAD, dtype=types.int32)
    for item in range(_ITEMS_PER_THREAD):
        index = thread * _ITEMS_PER_THREAD + item
        values[item] = source[index]

    canonical_values = qualified_coop.exclusive_scan(
        qualified_coop.this_block(),
        values,
        scan_op=_device_maximum,
        prefix_op=_prefix_after_block_aggregate,
        algorithm="raking_memoize",
    )
    compatibility_values = qualified_coop.exclusive_scan(
        qualified_coop.this_block(),
        values,
        scan_op=_device_maximum,
        block_prefix_callback_op=_prefix_after_block_aggregate,
        algorithm="raking_memoize",
    )
    for item in range(_ITEMS_PER_THREAD):
        index = thread * _ITEMS_PER_THREAD + item
        canonical[index] = canonical_values[item]
        compatibility[index] = compatibility_values[item]


def test_stateless_prefix_aliases_match_for_custom_array_scan_without_initial():
    source = ((np.arange(_TILE_ITEMS, dtype=np.int32) * 19) % 101) - 37
    canonical = np.full_like(source, -1)
    compatibility = np.full_like(source, -1)

    _block_scan_prefix_aliases[1, _BLOCK_THREADS](
        source,
        canonical,
        compatibility,
    )

    expected = np.full_like(source, source.max() + 7)
    np.testing.assert_array_equal(canonical, expected)
    np.testing.assert_array_equal(compatibility, expected)


@cache
def _stateful_prefix_kernel(algorithm: str, storage_mode: str):
    if storage_mode == "caller":

        @cuda.jit
        def kernel(source, output, final_state):
            thread = cuda.threadIdx.x
            state = qualified_coop.ThreadData(1, dtype=types.int64)
            state[0] = _PREFIX_INITIAL_STATE
            storage = qualified_coop.TempStorage(sharing="shared")
            for tile in range(_PREFIX_TILE_COUNT):
                index = tile * _BLOCK_THREADS + thread
                output[index] = qualified_coop.inclusive_sum(
                    qualified_coop.this_block(),
                    source[index],
                    state,
                    prefix_op=_RUNNING_PREFIX_INT64,
                    algorithm=algorithm,
                    temp_storage=storage,
                )
            if thread == 0:
                final_state[0] = state[0]

    elif storage_mode == "dynamic":

        @cuda.jit
        def kernel(source, output, final_state):
            thread = cuda.threadIdx.x
            state = cuda.local.array(1, dtype=types.int64)
            state[0] = _PREFIX_INITIAL_STATE
            storage = qualified_coop.TempStorage(
                _DYNAMIC_STORAGE_BYTES,
                alignment=16,
            )
            for tile in range(_PREFIX_TILE_COUNT):
                index = tile * _BLOCK_THREADS + thread
                output[index] = qualified_coop.exclusive_sum(
                    qualified_coop.this_block(),
                    source[index],
                    state,
                    prefix_op=_RUNNING_PREFIX_INT64,
                    algorithm=algorithm,
                    temp_storage=storage,
                )
            if thread == 0:
                final_state[0] = state[0]

    else:

        @cuda.jit
        def kernel(source, output, final_state):
            thread = cuda.threadIdx.x
            state = qualified_coop.ThreadData(1, dtype=types.int64)
            state[0] = _PREFIX_INITIAL_STATE
            storage = qualified_coop.TempStorage(
                sharing="shared",
                auto_sync=False,
            )
            for tile in range(_PREFIX_TILE_COUNT):
                index = tile * _BLOCK_THREADS + thread
                output[index] = qualified_coop.exclusive_sum(
                    qualified_coop.this_block(),
                    source[index],
                    state,
                    prefix_op=_RUNNING_PREFIX_INT64,
                    algorithm=algorithm,
                    temp_storage=storage,
                )
                cuda.syncthreads()
            if thread == 0:
                final_state[0] = state[0]

    return kernel


@pytest.mark.parametrize(
    ("algorithm", "storage_mode"),
    (
        pytest.param("raking", "caller", id="raking-caller-storage"),
        pytest.param(
            "raking_memoize",
            "dynamic",
            id="raking-memoize-dynamic-storage",
        ),
        pytest.param(
            "warp_scans",
            "manual-sync",
            id="warp-scans-auto-sync-false",
        ),
    ),
)
def test_stateful_prefix_tracks_repeated_scans_across_modes_and_storage(
    algorithm: str,
    storage_mode: str,
):
    source = (
        (np.arange(_PREFIX_TILE_COUNT * _BLOCK_THREADS, dtype=np.int32) * 7) % 23
    ) + 1
    output = np.full_like(source, -1)
    final_state = np.full(1, -1, dtype=np.int64)
    dispatcher = _stateful_prefix_kernel(algorithm, storage_mode)

    dispatcher[1, _BLOCK_THREADS](source, output, final_state)

    expected = (
        _PREFIX_INITIAL_STATE + np.cumsum(source, dtype=np.int32)
        if storage_mode == "caller"
        else _exclusive_sum(source, _PREFIX_INITIAL_STATE)
    )
    np.testing.assert_array_equal(output, expected)
    assert final_state[0] == _PREFIX_INITIAL_STATE + source.sum(dtype=np.int64)
    if storage_mode == "dynamic":
        compiled = next(iter(dispatcher._launch_config_overloads.values()))
        assert (
            compiled.metadata["required_dynamic_shared_memory"]
            == _DYNAMIC_STORAGE_BYTES
        )


@cuda.jit
def _warp_scans(source, operator_output, callback_output, partial, aggregates, valid):
    thread = cuda.threadIdx.x
    value = source[thread]
    logical_warp = qualified_coop.this_warp().group_by(_LOGICAL_WARP_THREADS)
    aggregate = qualified_coop.ThreadData(1, dtype=types.int32)

    operator_output[thread] = qualified_coop.inclusive_scan(
        qualified_coop.this_warp(), value, scan_op=operator.add
    )
    callback_output[thread] = qualified_coop.inclusive_scan(
        qualified_coop.this_warp(), value, scan_op=_device_maximum
    )
    partial[thread] = qualified_coop.exclusive_sum(
        logical_warp,
        value,
        valid_items=valid,
        aggregate_output=aggregate,
    )
    aggregates[thread] = aggregate[0]


def test_physical_and_logical_warp_forms_cover_alias_callback_and_valid_prefix():
    source = ((np.arange(_BLOCK_THREADS, dtype=np.int32) * 13) % 47) + 1
    operator_output = np.full_like(source, -1)
    callback_output = np.full_like(source, -1)
    partial = np.full_like(source, -1)
    aggregates = np.full_like(source, -1)

    _warp_scans[1, _BLOCK_THREADS](
        source,
        operator_output,
        callback_output,
        partial,
        aggregates,
        np.int64(_RUNTIME_VALID_ITEMS),
    )

    for start in range(0, _BLOCK_THREADS, _WARP_THREADS):
        warp = source[start : start + _WARP_THREADS]
        np.testing.assert_array_equal(
            operator_output[start : start + _WARP_THREADS],
            np.cumsum(warp, dtype=np.int32),
        )
        np.testing.assert_array_equal(
            callback_output[start : start + _WARP_THREADS],
            np.maximum.accumulate(warp),
        )
    for start in range(0, _BLOCK_THREADS, _LOGICAL_WARP_THREADS):
        valid_values = source[start : start + _RUNTIME_VALID_ITEMS]
        np.testing.assert_array_equal(
            partial[start : start + _RUNTIME_VALID_ITEMS],
            _exclusive_sum(valid_values),
        )
        np.testing.assert_array_equal(
            aggregates[start : start + _LOGICAL_WARP_THREADS],
            np.full(
                _LOGICAL_WARP_THREADS,
                valid_values.sum(dtype=np.int32),
                dtype=np.int32,
            ),
        )


@cuda.jit
def _warp_scan_combined_runtime_abi(source, output, aggregates, initial, valid):
    thread = cuda.threadIdx.x
    aggregate = qualified_coop.ThreadData(1, dtype=types.int32)
    output[thread] = qualified_coop.exclusive_scan(
        qualified_coop.this_warp().group_by(_LOGICAL_WARP_THREADS),
        source[thread],
        scan_op="max",
        initial_value=initial,
        valid_items=valid,
        aggregate_output=aggregate,
    )
    aggregates[thread] = aggregate[0]


def test_warp_scan_combines_runtime_initial_prefix_and_aggregate_abi():
    source = ((np.arange(_BLOCK_THREADS, dtype=np.int32) * 11) % 43) + 1
    output = np.full_like(source, -99)
    aggregates = np.full_like(source, -99)
    initial = np.int32(-17)

    _warp_scan_combined_runtime_abi[1, _BLOCK_THREADS](
        source,
        output,
        aggregates,
        initial,
        np.int64(_RUNTIME_VALID_ITEMS),
    )

    for start in range(0, _BLOCK_THREADS, _LOGICAL_WARP_THREADS):
        valid_values = source[start : start + _RUNTIME_VALID_ITEMS]
        expected = np.empty(_RUNTIME_VALID_ITEMS, dtype=np.int32)
        expected[0] = initial
        expected[1:] = np.maximum.accumulate(valid_values[:-1])
        np.testing.assert_array_equal(
            output[start : start + _RUNTIME_VALID_ITEMS],
            expected,
        )
        np.testing.assert_array_equal(
            aggregates[start : start + _LOGICAL_WARP_THREADS],
            np.full(
                _LOGICAL_WARP_THREADS,
                valid_values.max(),
                dtype=np.int32,
            ),
        )


@cache
def _storage_scan_kernel(storage_mode: str):
    if storage_mode == "implicit":

        @cuda.jit
        def kernel(source, output):
            thread = cuda.threadIdx.x
            output[thread] = qualified_coop.inclusive_sum(
                qualified_coop.this_block(), source[thread]
            )

    elif storage_mode == "caller":

        @cuda.jit
        def kernel(source, output):
            thread = cuda.threadIdx.x
            storage = qualified_coop.TempStorage(sharing="shared")
            output[thread] = qualified_coop.inclusive_sum(
                qualified_coop.this_block(),
                source[thread],
                temp_storage=storage,
            )

    else:

        @cuda.jit
        def kernel(source, output):
            thread = cuda.threadIdx.x
            storage = qualified_coop.TempStorage(64 * 1024, alignment=16)
            output[thread] = qualified_coop.inclusive_sum(
                qualified_coop.this_block(),
                source[thread],
                temp_storage=storage,
            )

    return kernel


@pytest.mark.parametrize("storage_mode", ("implicit", "caller", "dynamic"))
def test_block_scan_accepts_implicit_caller_and_dynamic_storage(storage_mode: str):
    source = np.arange(1, _BLOCK_THREADS + 1, dtype=np.int32)
    output = np.full_like(source, -1)
    dispatcher = _storage_scan_kernel(storage_mode)

    dispatcher[1, _BLOCK_THREADS](source, output)

    np.testing.assert_array_equal(output, np.cumsum(source, dtype=np.int32))
    if storage_mode == "dynamic":
        compiled = next(iter(dispatcher._launch_config_overloads.values()))
        assert compiled.metadata["required_dynamic_shared_memory"] == 64 * 1024


@cuda.jit
def _reuse_scan_storage(source, exclusive, inclusive, preserved):
    thread = cuda.threadIdx.x
    storage = qualified_coop.TempStorage(sharing="shared")
    value = source[thread]
    exclusive[thread] = qualified_coop.exclusive_sum(
        qualified_coop.this_block(), value, temp_storage=storage
    )
    inclusive[thread] = qualified_coop.inclusive_sum(
        qualified_coop.this_block(), value, temp_storage=storage
    )
    preserved[thread] = value


def test_reused_caller_storage_keeps_calls_ordered_and_input_unchanged():
    source = np.arange(1, _BLOCK_THREADS + 1, dtype=np.int32)
    exclusive = np.full_like(source, -1)
    inclusive = np.full_like(source, -1)
    preserved = np.full_like(source, -1)

    _reuse_scan_storage[1, _BLOCK_THREADS](source, exclusive, inclusive, preserved)

    np.testing.assert_array_equal(exclusive, _exclusive_sum(source))
    np.testing.assert_array_equal(inclusive, np.cumsum(source, dtype=np.int32))
    np.testing.assert_array_equal(preserved, source)


def _run_invalid_runtime_prefix_probe(
    valid_items: int,
) -> subprocess.CompletedProcess[str]:
    # A device trap poisons its CUDA context, so invalid launches must run in a
    # disposable child process rather than the pytest worker.
    script = f"""\
import numpy as np
import numba_cuda_mlir.cuda as cuda
from pathlib import Path

import cuda.coop.numba_mlir as coop

expected_origin = Path({str(_QUALIFIED_COOP_ORIGIN)!r})
actual_origin = Path(coop.__file__).resolve()
if actual_origin != expected_origin:
    raise RuntimeError(
        f"trap probe imported cuda.coop from {{actual_origin}}, "
        f"expected {{expected_origin}}"
    )

_THREADS = {_BLOCK_THREADS}
_WIDTH = {_LOGICAL_WARP_THREADS}

@cuda.jit
def kernel(source, output, valid):
    thread = cuda.threadIdx.x
    output[thread] = coop.inclusive_sum(
        coop.this_warp().group_by(_WIDTH),
        source[thread],
        valid_items=valid,
    )

source = np.arange(_THREADS, dtype=np.int32)
output = np.full_like(source, -1)
kernel[1, _THREADS](source, output, np.int64({valid_items}))
cuda.synchronize()
raise AssertionError("invalid Scan valid_items did not trap")
"""
    return subprocess.run(
        [sys.executable, _SAFE_PATH_FLAG, "-B", "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )


@pytest.mark.parametrize(
    "valid_items",
    (-1, 0, _LOGICAL_WARP_THREADS + 1),
    ids=("negative", "zero", "beyond-logical-warp"),
)
def test_invalid_runtime_valid_items_traps_in_an_isolated_process(valid_items: int):
    result = _run_invalid_runtime_prefix_probe(valid_items)
    output = result.stdout + result.stderr

    assert result.returncode != 0, output
    assert any(
        error in output
        for error in (
            "CUDA_ERROR_ILLEGAL_INSTRUCTION",
            "CUDA_ERROR_LAUNCH_FAILED",
        )
    ), output

    # Prove the pytest worker's independent CUDA context still works.
    source = np.arange(1, _BLOCK_THREADS + 1, dtype=np.int32)
    observed = np.full_like(source, -1)
    _storage_scan_kernel("implicit")[1, _BLOCK_THREADS](source, observed)
    assert observed[-1] == source.sum(dtype=np.int32)
