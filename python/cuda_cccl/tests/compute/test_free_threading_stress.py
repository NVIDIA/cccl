# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import concurrent.futures
import sys
import sysconfig
import threading
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pytest
from _utils.device_array import DeviceArray

from cuda.core import Device

pytestmark = [
    pytest.mark.free_threading,
    pytest.mark.no_numba,
    pytest.mark.no_verify_sass(
        reason="Free-threading stress tests intentionally run concurrent workers."
    ),
]

STRESS_ITERATIONS = 10
# Four workers give the single-flight build path multiple simultaneous waiters
# and richer interleavings than a single winner/waiter pair.
STRESS_THREADS = 4
TRANSFORM_NATIVE_CACHE_THREADS = 4
# Each iteration compiles one distinct specialization per worker, so iterations
# are capped well below STRESS_ITERATIONS to bound native-compile time.
DISTINCT_KEY_STORM_ITERATIONS = 3


def _is_free_threaded_build() -> bool:
    return sysconfig.get_config_var("Py_GIL_DISABLED") in (1, "1")


def _assert_gil_disabled(where: str) -> None:
    is_gil_enabled = getattr(sys, "_is_gil_enabled", None)
    if is_gil_enabled is not None and is_gil_enabled():
        pytest.fail(f"the GIL is enabled {where}")


def _require_free_threaded_python() -> None:
    if not _is_free_threaded_build():
        pytest.skip("requires a free-threaded CPython build")
    _assert_gil_disabled("at free-threaded test start")


def _require_serialization_backend() -> None:
    from cuda.compute._build_info import USING_V2

    if USING_V2:
        pytest.skip("serialization is not supported by the C Parallel v2 backend")


@pytest.fixture
def compute_module():
    _require_free_threaded_python()

    import cuda.compute as cc

    _assert_gil_disabled("after importing cuda.compute")
    cc.clear_all_caches()
    try:
        yield cc
    finally:
        cc.clear_all_caches()


def _make_stream():
    # cuda.core streams implement __cuda_stream__, so one object serves both
    # as the allocation/synchronization handle and the per-call stream arg.
    device = Device()
    device.set_current()
    return device.create_stream()


def _run_threaded(workers: list[Callable[[threading.Barrier], None]]) -> None:
    # The default timeout turns a worker that dies before reaching the barrier
    # into a BrokenBarrierError in its peers instead of hanging the CI job.
    barrier = threading.Barrier(len(workers), timeout=60)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(workers)) as executor:
        futures = [executor.submit(worker, barrier) for worker in workers]
        errors = []
        for future in futures:
            try:
                future.result()
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
        if errors:
            # Surface the root-cause worker failure, not the barrier breakage
            # it caused in the other workers.
            raise next(
                (
                    error
                    for error in errors
                    if not isinstance(error, threading.BrokenBarrierError)
                ),
                errors[0],
            )
    _assert_gil_disabled("after concurrent cuda.compute operations")


def _call_with_temp(algorithm, **kwargs):
    temp_storage_bytes = algorithm(temp_storage=None, **kwargs)
    temp_storage = DeviceArray.empty(temp_storage_bytes, dtype=np.uint8)
    return algorithm(temp_storage=temp_storage, **kwargs)


def _get_build_result(algorithm):
    if hasattr(algorithm, "build_results"):
        assert len(algorithm.build_results) == 1
        return next(iter(algorithm.build_results.values()))
    if hasattr(algorithm, "build_result"):
        return algorithm.build_result
    if hasattr(algorithm, "partitioner"):
        return _get_build_result(algorithm.partitioner)
    raise AssertionError(f"{type(algorithm).__name__} does not expose a build result")


@dataclass(frozen=True)
class _AlgorithmCase:
    name: str
    make_shared: Callable
    make_worker: Callable
    run: Callable
    check: Callable

    def __str__(self):
        return self.name


def _run_thread_local_algorithm_case(cc, case: _AlgorithmCase) -> None:
    warm_algorithm = case.make_shared(cc)

    warm_worker = case.make_worker(cc, worker_id=0, iteration=-1)
    case.run(cc, warm_algorithm, warm_worker)
    case.check(cc, warm_worker)

    for iteration in range(STRESS_ITERATIONS):
        worker_state = [
            case.make_worker(cc, worker_id=worker_id, iteration=iteration)
            for worker_id in range(STRESS_THREADS)
        ]
        returned_algorithms = [None] * STRESS_THREADS

        def make_thread(worker_id, worker):
            def thread(barrier):
                barrier.wait()
                algorithm = case.make_shared(cc)
                returned_algorithms[worker_id] = algorithm
                case.run(cc, algorithm, worker)
                case.check(cc, worker)

            return thread

        _run_threaded(
            [
                make_thread(worker_id, worker)
                for worker_id, worker in enumerate(worker_state)
            ]
        )

        assert len({id(algorithm) for algorithm in returned_algorithms}) == len(
            returned_algorithms
        )
        assert (
            len({id(_get_build_result(algorithm)) for algorithm in returned_algorithms})
            == 1
        )


def _make_reduce_worker(cc, worker_id, iteration):
    stream = _make_stream()
    h_in = np.arange(64, dtype=np.int32) + worker_id * 101 + iteration
    h_init = np.array([7 + worker_id], dtype=np.int32)
    d_in = DeviceArray.from_numpy(h_in)
    d_out = DeviceArray.empty(1, dtype=np.int32)
    return {
        "stream": stream,
        "h_in": h_in,
        "d_in": d_in,
        "d_out": d_out,
        "h_init": h_init,
    }


def _make_reduce_shared(cc):
    worker = _make_reduce_worker(cc, 0, -1)
    return cc.make_reduce_into(
        d_in=worker["d_in"],
        d_out=worker["d_out"],
        op=cc.OpKind.PLUS,
        h_init=worker["h_init"],
    )


def _run_reduce(cc, reducer, worker):
    _call_with_temp(
        reducer,
        d_in=worker["d_in"],
        d_out=worker["d_out"],
        op=cc.OpKind.PLUS,
        h_init=worker["h_init"],
        num_items=worker["h_in"].size,
        stream=worker["stream"],
    )


def _check_reduce(cc, worker):
    worker["stream"].sync()
    expected = worker["h_in"].sum(dtype=np.int64) + int(worker["h_init"][0])
    assert int(worker["d_out"].copy_to_host()[0]) == int(expected)


def _make_unary_worker(cc, worker_id, iteration):
    stream = _make_stream()
    h_in = np.arange(32, dtype=np.int32) + worker_id * 17 + iteration
    d_in = DeviceArray.from_numpy(h_in)
    d_out = DeviceArray.empty(h_in.shape, h_in.dtype)
    return {
        "stream": stream,
        "h_in": h_in,
        "d_in": d_in,
        "d_out": d_out,
    }


def _make_unary_shared(cc):
    worker = _make_unary_worker(cc, 0, -1)
    return cc.make_unary_transform(
        d_in=worker["d_in"], d_out=worker["d_out"], op=cc.OpKind.NEGATE
    )


def _make_unary_for_worker(cc, worker):
    return cc.make_unary_transform(
        d_in=worker["d_in"], d_out=worker["d_out"], op=cc.OpKind.NEGATE
    )


def _run_unary(cc, transformer, worker):
    transformer(
        d_in=worker["d_in"],
        d_out=worker["d_out"],
        op=cc.OpKind.NEGATE,
        num_items=worker["h_in"].size,
        stream=worker["stream"],
    )


def _run_unary_empty(cc, transformer, worker):
    transformer(
        d_in=worker["d_in"],
        d_out=worker["d_out"],
        op=cc.OpKind.NEGATE,
        num_items=0,
        stream=worker["stream"],
    )


def _check_unary(cc, worker):
    worker["stream"].sync()
    np.testing.assert_array_equal(worker["d_out"].copy_to_host(), -worker["h_in"])


def _make_binary_worker(cc, worker_id, iteration):
    stream = _make_stream()
    h_in1 = np.arange(32, dtype=np.int32) + worker_id * 13
    h_in2 = np.arange(32, dtype=np.int32) + iteration * 7
    d_in1 = DeviceArray.from_numpy(h_in1)
    d_in2 = DeviceArray.from_numpy(h_in2)
    d_out = DeviceArray.empty(h_in1.shape, h_in1.dtype)
    return {
        "stream": stream,
        "h_in1": h_in1,
        "h_in2": h_in2,
        "d_in1": d_in1,
        "d_in2": d_in2,
        "d_out": d_out,
    }


def _make_binary_for_worker(cc, worker):
    return cc.make_binary_transform(
        d_in1=worker["d_in1"],
        d_in2=worker["d_in2"],
        d_out=worker["d_out"],
        op=cc.OpKind.PLUS,
    )


def _run_binary(cc, transformer, worker):
    transformer(
        d_in1=worker["d_in1"],
        d_in2=worker["d_in2"],
        d_out=worker["d_out"],
        op=cc.OpKind.PLUS,
        num_items=worker["h_in1"].size,
        stream=worker["stream"],
    )


def _run_binary_empty(cc, transformer, worker):
    transformer(
        d_in1=worker["d_in1"],
        d_in2=worker["d_in2"],
        d_out=worker["d_out"],
        op=cc.OpKind.PLUS,
        num_items=0,
        stream=worker["stream"],
    )


def _check_binary(cc, worker):
    worker["stream"].sync()
    np.testing.assert_array_equal(
        worker["d_out"].copy_to_host(), worker["h_in1"] + worker["h_in2"]
    )


def _make_scan_worker(cc, worker_id, iteration):
    stream = _make_stream()
    h_in = np.arange(1, 33, dtype=np.int32) + worker_id + iteration
    h_init = np.array([3 + worker_id], dtype=np.int32)
    d_in = DeviceArray.from_numpy(h_in)
    d_out = DeviceArray.empty(h_in.shape, h_in.dtype)
    return {
        "stream": stream,
        "h_in": h_in,
        "h_init": h_init,
        "d_in": d_in,
        "d_out": d_out,
    }


def _make_exclusive_scan_shared(cc):
    worker = _make_scan_worker(cc, 0, -1)
    return cc.make_exclusive_scan(
        d_in=worker["d_in"],
        d_out=worker["d_out"],
        op=cc.OpKind.PLUS,
        init_value=worker["h_init"],
    )


def _run_scan(cc, scanner, worker):
    _call_with_temp(
        scanner,
        d_in=worker["d_in"],
        d_out=worker["d_out"],
        op=cc.OpKind.PLUS,
        init_value=worker["h_init"],
        num_items=worker["h_in"].size,
        stream=worker["stream"],
    )


def _check_exclusive_scan(cc, worker):
    worker["stream"].sync()
    expected = np.empty_like(worker["h_in"])
    expected[0] = worker["h_init"][0]
    expected[1:] = worker["h_init"][0] + np.cumsum(worker["h_in"][:-1])
    np.testing.assert_array_equal(worker["d_out"].copy_to_host(), expected)


def _make_segmented_reduce_worker(cc, worker_id, iteration):
    stream = _make_stream()
    h_in = np.arange(1, 17, dtype=np.int32) + worker_id * 3 + iteration
    h_start_offsets = np.array([0, 3, 8, 12], dtype=np.int32)
    h_end_offsets = np.array([3, 8, 12, 16], dtype=np.int32)
    h_init = np.array([worker_id], dtype=np.int32)
    d_in = DeviceArray.from_numpy(h_in)
    d_out = DeviceArray.empty(len(h_start_offsets), dtype=np.int32)
    d_start_offsets = DeviceArray.from_numpy(h_start_offsets)
    d_end_offsets = DeviceArray.from_numpy(h_end_offsets)
    return {
        "stream": stream,
        "h_in": h_in,
        "h_start_offsets": h_start_offsets,
        "h_end_offsets": h_end_offsets,
        "h_init": h_init,
        "d_in": d_in,
        "d_out": d_out,
        "d_start_offsets": d_start_offsets,
        "d_end_offsets": d_end_offsets,
    }


def _make_segmented_reduce_shared(cc):
    worker = _make_segmented_reduce_worker(cc, 0, -1)
    return cc.make_segmented_reduce(
        d_in=worker["d_in"],
        d_out=worker["d_out"],
        start_offsets_in=worker["d_start_offsets"],
        end_offsets_in=worker["d_end_offsets"],
        op=cc.OpKind.PLUS,
        h_init=worker["h_init"],
    )


def _run_segmented_reduce(cc, reducer, worker):
    _call_with_temp(
        reducer,
        d_in=worker["d_in"],
        d_out=worker["d_out"],
        num_segments=len(worker["h_start_offsets"]),
        start_offsets_in=worker["d_start_offsets"],
        end_offsets_in=worker["d_end_offsets"],
        op=cc.OpKind.PLUS,
        h_init=worker["h_init"],
        stream=worker["stream"],
    )


def _check_segmented_reduce(cc, worker):
    worker["stream"].sync()
    expected = np.array(
        [
            worker["h_in"][start:end].sum() + worker["h_init"][0]
            for start, end in zip(worker["h_start_offsets"], worker["h_end_offsets"])
        ],
        dtype=np.int32,
    )
    np.testing.assert_array_equal(worker["d_out"].copy_to_host(), expected)


def _make_binary_search_worker(cc, worker_id, iteration):
    stream = _make_stream()
    h_data = np.array([90, 70, 50, 30, 10], dtype=np.int32) - worker_id
    h_values = np.array([95, 70, 45, 10, 5], dtype=np.int32) - worker_id
    d_data = DeviceArray.from_numpy(h_data)
    d_values = DeviceArray.from_numpy(h_values)
    d_out = DeviceArray.empty(h_values.size, dtype=np.uintp)
    return {
        "stream": stream,
        "h_data": h_data,
        "h_values": h_values,
        "d_data": d_data,
        "d_values": d_values,
        "d_out": d_out,
    }


def _make_lower_bound_for_worker(cc, worker):
    return cc.make_lower_bound(
        d_data=worker["d_data"],
        d_values=worker["d_values"],
        d_out=worker["d_out"],
        comp=cc.OpKind.GREATER,
    )


def _make_upper_bound_for_worker(cc, worker):
    return cc.make_upper_bound(
        d_data=worker["d_data"],
        d_values=worker["d_values"],
        d_out=worker["d_out"],
        comp=cc.OpKind.GREATER,
    )


def _run_binary_search(cc, searcher, worker):
    searcher(
        d_data=worker["d_data"],
        num_items=worker["h_data"].size,
        d_values=worker["d_values"],
        num_values=worker["h_values"].size,
        d_out=worker["d_out"],
        comp=cc.OpKind.GREATER,
        stream=worker["stream"],
    )


def _run_binary_search_empty(cc, searcher, worker):
    searcher(
        d_data=worker["d_data"],
        num_items=worker["h_data"].size,
        d_values=worker["d_values"],
        num_values=0,
        d_out=worker["d_out"],
        comp=cc.OpKind.GREATER,
        stream=worker["stream"],
    )


def _check_lower_bound(cc, worker):
    worker["stream"].sync()
    expected = np.searchsorted(-worker["h_data"], -worker["h_values"], side="left")
    np.testing.assert_array_equal(
        worker["d_out"].copy_to_host(), expected.astype(np.uintp)
    )


def _check_upper_bound(cc, worker):
    worker["stream"].sync()
    expected = np.searchsorted(-worker["h_data"], -worker["h_values"], side="right")
    np.testing.assert_array_equal(
        worker["d_out"].copy_to_host(), expected.astype(np.uintp)
    )


def _make_select_worker(cc, worker_id, iteration):
    stream = _make_stream()
    h_in = np.array(
        [True, False, worker_id % 2 == 0, True, False, iteration % 2 == 0],
        dtype=np.bool_,
    )
    d_in = DeviceArray.from_numpy(h_in)
    d_out = DeviceArray.empty(h_in.shape, h_in.dtype)
    d_count = DeviceArray.empty(2, dtype=np.uint64)
    return {
        "stream": stream,
        "h_in": h_in,
        "d_in": d_in,
        "d_out": d_out,
        "d_count": d_count,
    }


def _make_three_way_shared(cc):
    worker = _make_select_worker(cc, 0, -1)
    d_unselected = DeviceArray.empty(worker["h_in"].shape, worker["h_in"].dtype)
    return cc.make_three_way_partition(
        d_in=worker["d_in"],
        d_first_part_out=worker["d_out"],
        d_second_part_out=d_unselected,
        d_unselected_out=DeviceArray.empty(worker["h_in"].shape, worker["h_in"].dtype),
        d_num_selected_out=worker["d_count"],
        select_first_part_op=cc.OpKind.IDENTITY,
        select_second_part_op=cc.OpKind.LOGICAL_NOT,
    )


def _make_three_way_worker(cc, worker_id, iteration):
    worker = _make_select_worker(cc, worker_id, iteration)
    worker["d_second_out"] = DeviceArray.empty(
        worker["h_in"].shape, worker["h_in"].dtype
    )
    worker["d_unselected"] = DeviceArray.empty(
        worker["h_in"].shape, worker["h_in"].dtype
    )
    return worker


def _run_three_way(cc, partitioner, worker):
    _call_with_temp(
        partitioner,
        d_in=worker["d_in"],
        d_first_part_out=worker["d_out"],
        d_second_part_out=worker["d_second_out"],
        d_unselected_out=worker["d_unselected"],
        d_num_selected_out=worker["d_count"],
        select_first_part_op=cc.OpKind.IDENTITY,
        select_second_part_op=cc.OpKind.LOGICAL_NOT,
        num_items=worker["h_in"].size,
        stream=worker["stream"],
    )


def _check_three_way(cc, worker):
    worker["stream"].sync()
    counts = worker["d_count"].copy_to_host()
    true_count = int(np.count_nonzero(worker["h_in"]))
    false_count = int(worker["h_in"].size - true_count)
    assert int(counts[0]) == true_count
    assert int(counts[1]) == false_count
    np.testing.assert_array_equal(
        worker["d_out"].copy_to_host()[:true_count], np.ones(true_count, dtype=np.bool_)
    )
    np.testing.assert_array_equal(
        worker["d_second_out"].copy_to_host()[:false_count],
        np.zeros(false_count, dtype=np.bool_),
    )


def _make_radix_sort_worker(cc, worker_id, iteration):
    stream = _make_stream()
    h_keys = np.array([7, 3, 5, 3, 1, 9], dtype=np.uint32) + np.uint32(worker_id * 11)
    h_values = np.arange(h_keys.size, dtype=np.int32) + iteration * 10
    d_in_keys = DeviceArray.from_numpy(h_keys)
    d_tmp_keys = DeviceArray.empty(h_keys.shape, h_keys.dtype)
    d_in_values = DeviceArray.from_numpy(h_values)
    d_tmp_values = DeviceArray.empty(h_values.shape, h_values.dtype)
    return {
        "stream": stream,
        "h_keys": h_keys,
        "h_values": h_values,
        "keys": cc.DoubleBuffer(d_in_keys, d_tmp_keys),
        "values": cc.DoubleBuffer(d_in_values, d_tmp_values),
    }


def _make_radix_sort_shared(cc):
    worker = _make_radix_sort_worker(cc, 0, -1)
    return cc.make_radix_sort(
        d_in_keys=worker["keys"],
        d_out_keys=None,
        d_in_values=worker["values"],
        d_out_values=None,
        order=cc.SortOrder.ASCENDING,
    )


def _run_radix_sort(cc, sorter, worker):
    _call_with_temp(
        sorter,
        d_in_keys=worker["keys"],
        d_out_keys=None,
        d_in_values=worker["values"],
        d_out_values=None,
        num_items=worker["h_keys"].size,
        stream=worker["stream"],
    )


def _check_radix_sort(cc, worker):
    worker["stream"].sync()
    order = np.argsort(worker["h_keys"], kind="stable")
    np.testing.assert_array_equal(
        worker["keys"].current().copy_to_host(), worker["h_keys"][order]
    )
    np.testing.assert_array_equal(
        worker["values"].current().copy_to_host(), worker["h_values"][order]
    )
    assert worker["keys"].selector == worker["values"].selector


# A representative subset rather than every algorithm: these tests exercise the
# algorithm-agnostic caching machinery (per-thread wrappers, one shared build,
# single-flight coalescing, concurrent deserialize), so the cases are chosen
# for build-result/descriptor variety rather than coverage of every algorithm —
# basic reduce, a transform, a scan (init_value), a segmented op (offset
# iterators), a sort (DoubleBuffer + SortOrder), and a multi-output partition.
# Algorithm-specific native concurrency (transform's launch-config cache, the
# v2 first-call gate) is covered by its own dedicated tests below, and
# per-algorithm correctness lives in the per-algorithm test files.
SHARED_ALGORITHM_CASES = [
    _AlgorithmCase(
        "reduce", _make_reduce_shared, _make_reduce_worker, _run_reduce, _check_reduce
    ),
    _AlgorithmCase(
        "unary_transform",
        _make_unary_shared,
        _make_unary_worker,
        _run_unary,
        _check_unary,
    ),
    _AlgorithmCase(
        "exclusive_scan",
        _make_exclusive_scan_shared,
        _make_scan_worker,
        _run_scan,
        _check_exclusive_scan,
    ),
    _AlgorithmCase(
        "segmented_reduce",
        _make_segmented_reduce_shared,
        _make_segmented_reduce_worker,
        _run_segmented_reduce,
        _check_segmented_reduce,
    ),
    _AlgorithmCase(
        "radix_sort",
        _make_radix_sort_shared,
        _make_radix_sort_worker,
        _run_radix_sort,
        _check_radix_sort,
    ),
    _AlgorithmCase(
        "three_way_partition",
        _make_three_way_shared,
        _make_three_way_worker,
        _run_three_way,
        _check_three_way,
    ),
]


def test_free_threaded_import_keeps_gil_disabled(compute_module):
    """True first-import smoke for the PR's headline claim.

    Asserting around imports in this process cannot guarantee ordering:
    sibling test modules import cuda.compute and cuda.core at collection time.
    A fresh interpreter gives the assert -> import -> assert bracket genuine
    first-import semantics.
    """
    import os
    import subprocess

    del compute_module  # only used to gate on a free-threaded build

    tests_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    code = f"""
import sys

assert not sys._is_gil_enabled(), "the GIL is enabled at interpreter start"
sys.path.insert(0, {tests_dir!r})
from _utils.device_array import DeviceArray

assert not sys._is_gil_enabled(), "the GIL is enabled after importing the device-array utility"
import numpy as np

import cuda.compute as cc

assert not sys._is_gil_enabled(), "the GIL is enabled after importing cuda.compute"
h_in = np.arange(8, dtype=np.int32)
d_in = DeviceArray.from_numpy(h_in)
d_out = DeviceArray.empty(1, dtype=np.int32)
h_init = np.array([0], dtype=np.int32)
cc.reduce_into(d_in=d_in, d_out=d_out, num_items=h_in.size, op=cc.OpKind.PLUS, h_init=h_init)
assert int(d_out.copy_to_host()[0]) == int(h_in.sum())
assert not sys._is_gil_enabled(), "the GIL is enabled after running a cuda.compute operation"
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, (
        f"first-import GIL smoke subprocess failed (exit {result.returncode})\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    _assert_gil_disabled("after the first-import smoke subprocess")


@pytest.mark.parametrize("case", SHARED_ALGORITHM_CASES, ids=str)
def test_thread_local_algorithm_objects_share_build_result(compute_module, case):
    cc = compute_module

    _run_thread_local_algorithm_case(cc, case)


@pytest.mark.parametrize("case", SHARED_ALGORITHM_CASES, ids=str)
def test_concurrent_deserialize_and_execute(compute_module, case):
    cc = compute_module
    _require_serialization_backend()

    source_algorithm = case.make_shared(cc)
    blob = cc.serialize(source_algorithm)
    device_id = Device().device_id
    _assert_gil_disabled("after serializing a cuda.compute algorithm")

    for iteration in range(STRESS_ITERATIONS):
        workers = [
            case.make_worker(cc, worker_id=worker_id, iteration=iteration)
            for worker_id in range(STRESS_THREADS)
        ]
        algorithms = [None] * STRESS_THREADS
        execute_barrier = threading.Barrier(STRESS_THREADS)

        def make_thread(worker_id, worker):
            def thread(barrier):
                Device(device_id).set_current()
                barrier.wait()
                try:
                    algorithm = cc.deserialize(blob)
                    algorithms[worker_id] = algorithm
                    # Force the independently deserialized build results to
                    # perform their first native loads concurrently.
                    execute_barrier.wait(timeout=60)
                    case.run(cc, algorithm, worker)
                    case.check(cc, worker)
                except BaseException:
                    execute_barrier.abort()
                    raise

            return thread

        _run_threaded(
            [make_thread(worker_id, worker) for worker_id, worker in enumerate(workers)]
        )

        # deserialize() intentionally returns independent mutable wrappers and
        # independently owned native build results. Sharing one wrapper between
        # concurrent callers remains unsupported.
        assert len({id(algorithm) for algorithm in algorithms}) == STRESS_THREADS
        assert (
            len({id(_get_build_result(algorithm)) for algorithm in algorithms})
            == STRESS_THREADS
        )


def test_free_threaded_aot_factory_shares_compile_and_first_load(compute_module):
    cc = compute_module
    _require_serialization_backend()

    device = Device()
    target_cc = tuple(device.compute_capability)
    proxy_in = cc.ProxyArray(np.int32)
    proxy_out = cc.ProxyArray(np.int32)
    workers = [
        _make_unary_worker(cc, worker_id=worker_id, iteration=0)
        for worker_id in range(STRESS_THREADS)
    ]
    algorithms = [None] * STRESS_THREADS
    execute_barrier = threading.Barrier(STRESS_THREADS)

    def make_thread(worker_id, worker):
        def thread(barrier):
            Device(device.device_id).set_current()
            barrier.wait()
            try:
                algorithm = cc.make_unary_transform(
                    d_in=proxy_in,
                    d_out=proxy_out,
                    op=cc.OpKind.NEGATE,
                    compute_capability=target_cc,
                )
                algorithms[worker_id] = algorithm
                execute_barrier.wait(timeout=60)
                _run_unary(cc, algorithm, worker)
                _check_unary(cc, worker)
            except BaseException:
                execute_barrier.abort()
                raise

        return thread

    _run_threaded(
        [make_thread(worker_id, worker) for worker_id, worker in enumerate(workers)]
    )

    assert len({id(algorithm) for algorithm in algorithms}) == STRESS_THREADS
    assert len({id(algorithm.build_results) for algorithm in algorithms}) == 1
    assert len({id(_get_build_result(algorithm)) for algorithm in algorithms}) == 1
    assert _get_build_result(algorithms[0])._loaded


def test_free_threaded_aot_blob_concurrent_deserialize_and_load(compute_module):
    cc = compute_module
    _require_serialization_backend()

    device = Device()
    target_cc = tuple(device.compute_capability)
    source_algorithm = cc.make_unary_transform(
        d_in=cc.ProxyArray(np.int32),
        d_out=cc.ProxyArray(np.int32),
        op=cc.OpKind.NEGATE,
        compute_capability=target_cc,
    )
    source_build_result = _get_build_result(source_algorithm)
    assert not source_build_result._loaded
    blob = cc.serialize(source_algorithm)

    workers = [
        _make_unary_worker(cc, worker_id=worker_id, iteration=0)
        for worker_id in range(STRESS_THREADS)
    ]
    algorithms = [None] * STRESS_THREADS
    execute_barrier = threading.Barrier(STRESS_THREADS)

    def make_thread(worker_id, worker):
        def thread(barrier):
            Device(device.device_id).set_current()
            barrier.wait()
            try:
                algorithm = cc.deserialize(blob)
                algorithms[worker_id] = algorithm
                assert not _get_build_result(algorithm)._loaded
                execute_barrier.wait(timeout=60)
                _run_unary(cc, algorithm, worker)
                _check_unary(cc, worker)
            except BaseException:
                execute_barrier.abort()
                raise

        return thread

    _run_threaded(
        [make_thread(worker_id, worker) for worker_id, worker in enumerate(workers)]
    )

    build_results = [_get_build_result(algorithm) for algorithm in algorithms]
    assert len({id(algorithm) for algorithm in algorithms}) == STRESS_THREADS
    assert len({id(build_result) for build_result in build_results}) == STRESS_THREADS
    assert all(build_result._loaded for build_result in build_results)
    assert not source_build_result._loaded


def test_free_threaded_multi_cc_blob_concurrent_deserialize_and_execute(
    compute_module,
):
    """Concurrent per-thread use of a genuine multi-arch blob.

    Multi-cc blobs take a different path from the single-cc blobs above: the
    cc check is deferred from deserialize time to call time, and each call
    resolves the current device's entry out of a multi-entry collection. The
    entry for the other arch must stay lazy in every thread.
    """
    cc = compute_module
    _require_serialization_backend()

    device = Device()
    cc_major, cc_minor = device.compute_capability
    current_key = cc_major * 10 + cc_minor

    # Find one more arch this toolchain can compile for.
    other_key = None
    for candidate in (90, 100, 89, 120, 80, 86, 75):
        if candidate == current_key:
            continue
        try:
            cc.make_unary_transform(
                d_in=cc.ProxyArray(np.int32),
                d_out=cc.ProxyArray(np.int32),
                op=cc.OpKind.NEGATE,
                compute_capability=candidate,
            )
        except Exception:
            continue
        other_key = candidate
        break
    if other_key is None:
        pytest.skip("toolchain compiles for <2 target arches")

    source_algorithm = cc.make_unary_transform(
        d_in=cc.ProxyArray(np.int32),
        d_out=cc.ProxyArray(np.int32),
        op=cc.OpKind.NEGATE,
        compute_capability=[current_key, other_key],
    )
    blob = cc.serialize(source_algorithm)

    workers = [
        _make_unary_worker(cc, worker_id=worker_id, iteration=0)
        for worker_id in range(STRESS_THREADS)
    ]
    algorithms = [None] * STRESS_THREADS
    execute_barrier = threading.Barrier(STRESS_THREADS)

    def make_thread(worker_id, worker):
        def thread(barrier):
            Device(device.device_id).set_current()
            barrier.wait()
            try:
                algorithm = cc.deserialize(blob)
                algorithms[worker_id] = algorithm
                execute_barrier.wait(timeout=60)
                _run_unary(cc, algorithm, worker)
                _check_unary(cc, worker)
            except BaseException:
                execute_barrier.abort()
                raise

        return thread

    _run_threaded(
        [make_thread(worker_id, worker) for worker_id, worker in enumerate(workers)]
    )

    for algorithm in algorithms:
        assert set(algorithm.build_results.keys()) == {current_key, other_key}
        # Only the current device's arch loads; the other entry stays lazy.
        assert algorithm.build_results[current_key]._loaded
        assert not algorithm.build_results[other_key]._loaded


def _cache_miss_reduce(cc, worker_id, iteration):
    worker = _make_reduce_worker(cc, worker_id, iteration)
    reducer = cc.make_reduce_into(
        d_in=worker["d_in"],
        d_out=worker["d_out"],
        op=cc.OpKind.PLUS,
        h_init=worker["h_init"],
    )
    _run_reduce(cc, reducer, worker)
    _check_reduce(cc, worker)
    return reducer


def _cache_miss_unary_transform(cc, worker_id, iteration):
    worker = _make_unary_worker(cc, worker_id, iteration)
    transformer = cc.make_unary_transform(
        d_in=worker["d_in"], d_out=worker["d_out"], op=cc.OpKind.NEGATE
    )
    _run_unary(cc, transformer, worker)
    _check_unary(cc, worker)
    return transformer


def _cache_miss_binary_transform(cc, worker_id, iteration):
    worker = _make_binary_worker(cc, worker_id, iteration)
    transformer = cc.make_binary_transform(
        d_in1=worker["d_in1"],
        d_in2=worker["d_in2"],
        d_out=worker["d_out"],
        op=cc.OpKind.PLUS,
    )
    _run_binary(cc, transformer, worker)
    _check_binary(cc, worker)
    return transformer


@pytest.mark.parametrize(
    "factory",
    [_cache_miss_reduce, _cache_miss_unary_transform, _cache_miss_binary_transform],
    ids=["reduce", "unary_transform", "binary_transform"],
)
def test_same_key_factory_cache_miss_storm(compute_module, factory):
    cc = compute_module

    for iteration in range(STRESS_ITERATIONS):
        cc.clear_all_caches()
        returned_objects = [None] * STRESS_THREADS

        def make_thread(worker_id):
            def thread(barrier):
                barrier.wait()
                returned_objects[worker_id] = factory(cc, worker_id, iteration)

            return thread

        _run_threaded([make_thread(worker_id) for worker_id in range(STRESS_THREADS)])

        assert len({id(obj) for obj in returned_objects}) == len(returned_objects)
        assert len({id(_get_build_result(obj)) for obj in returned_objects}) == 1


def _storm_reduce_int32(cc, worker_id, iteration):
    stream = _make_stream()
    h_in = np.arange(64, dtype=np.int32) + worker_id + iteration
    h_init = np.array([worker_id], dtype=np.int32)
    d_in = DeviceArray.from_numpy(h_in)
    d_out = DeviceArray.empty(1, dtype=np.int32)
    reducer = cc.make_reduce_into(
        d_in=d_in, d_out=d_out, op=cc.OpKind.PLUS, h_init=h_init
    )
    _call_with_temp(
        reducer,
        d_in=d_in,
        d_out=d_out,
        op=cc.OpKind.PLUS,
        h_init=h_init,
        num_items=h_in.size,
        stream=stream,
    )
    stream.sync()
    assert int(d_out.copy_to_host()[0]) == int(h_in.sum()) + int(h_init[0])
    return reducer


def _storm_reduce_float64(cc, worker_id, iteration):
    stream = _make_stream()
    h_in = np.arange(64, dtype=np.float64) + worker_id + iteration
    h_init = np.array([worker_id], dtype=np.float64)
    d_in = DeviceArray.from_numpy(h_in)
    d_out = DeviceArray.empty(1, dtype=np.float64)
    reducer = cc.make_reduce_into(
        d_in=d_in, d_out=d_out, op=cc.OpKind.PLUS, h_init=h_init
    )
    _call_with_temp(
        reducer,
        d_in=d_in,
        d_out=d_out,
        op=cc.OpKind.PLUS,
        h_init=h_init,
        num_items=h_in.size,
        stream=stream,
    )
    stream.sync()
    assert float(d_out.copy_to_host()[0]) == float(h_in.sum()) + float(h_init[0])
    return reducer


def _storm_unary_transform_int64(cc, worker_id, iteration):
    stream = _make_stream()
    h_in = np.arange(64, dtype=np.int64) + worker_id + iteration
    d_in = DeviceArray.from_numpy(h_in)
    d_out = DeviceArray.empty(h_in.shape, h_in.dtype)
    transformer = cc.make_unary_transform(d_in=d_in, d_out=d_out, op=cc.OpKind.NEGATE)
    transformer(
        d_in=d_in,
        d_out=d_out,
        op=cc.OpKind.NEGATE,
        num_items=h_in.size,
        stream=stream,
    )
    stream.sync()
    np.testing.assert_array_equal(d_out.copy_to_host(), -h_in)
    return transformer


def _storm_binary_transform_float32(cc, worker_id, iteration):
    stream = _make_stream()
    h_in1 = np.arange(64, dtype=np.float32) + worker_id
    h_in2 = np.arange(64, dtype=np.float32) + iteration
    d_in1 = DeviceArray.from_numpy(h_in1)
    d_in2 = DeviceArray.from_numpy(h_in2)
    d_out = DeviceArray.empty(h_in1.shape, h_in1.dtype)
    transformer = cc.make_binary_transform(
        d_in1=d_in1, d_in2=d_in2, d_out=d_out, op=cc.OpKind.PLUS
    )
    transformer(
        d_in1=d_in1,
        d_in2=d_in2,
        d_out=d_out,
        op=cc.OpKind.PLUS,
        num_items=h_in1.size,
        stream=stream,
    )
    stream.sync()
    np.testing.assert_array_equal(d_out.copy_to_host(), h_in1 + h_in2)
    return transformer


_DISTINCT_KEY_STORM_FACTORIES = [
    _storm_reduce_int32,
    _storm_reduce_float64,
    _storm_unary_transform_int64,
    _storm_binary_transform_float32,
]


def test_distinct_key_cold_build_storm(compute_module):
    """Force truly concurrent native compilations.

    Same-key storms coalesce through _cache_single_flight to a single builder
    thread, so they never overlap the native compile pipeline. Distinct keys
    (different algorithms and dtypes) elect one builder per worker, running
    NVRTC/nvJitLink (v1) or HostJIT (v2) compilations genuinely in parallel.
    """
    cc = compute_module

    for iteration in range(DISTINCT_KEY_STORM_ITERATIONS):
        cc.clear_all_caches()
        returned_objects = [None] * len(_DISTINCT_KEY_STORM_FACTORIES)

        def make_thread(worker_id, factory):
            def thread(barrier):
                barrier.wait()
                returned_objects[worker_id] = factory(cc, worker_id, iteration)

            return thread

        _run_threaded(
            [
                make_thread(worker_id, factory)
                for worker_id, factory in enumerate(_DISTINCT_KEY_STORM_FACTORIES)
            ]
        )

        # Distinct keys must not share builds: one build result per worker.
        build_ids = {id(_get_build_result(obj)) for obj in returned_objects}
        assert len(build_ids) == len(_DISTINCT_KEY_STORM_FACTORIES)


def test_shared_raw_op_object_direct_algorithm_stress(compute_module):
    cc = compute_module

    from cuda.compute._cpp_compile import compile_cpp_op_code
    from cuda.compute.op import RawOp

    source = """
    extern "C" __device__ void raw_add_i32(void* a, void* b, void* result) {
        *static_cast<int*>(result) = *static_cast<int*>(a) + *static_cast<int*>(b);
    }
    """
    shared_op = RawOp(ltoir=compile_cpp_op_code(source), name="raw_add_i32")

    for iteration in range(STRESS_ITERATIONS):
        cc.clear_all_caches()
        returned_reducers = [None] * STRESS_THREADS

        def make_thread(worker_id):
            stream = _make_stream()
            h_in = np.arange(32, dtype=np.int32) + worker_id * 31 + iteration
            h_init = np.array([worker_id + 5], dtype=np.int32)
            d_in = DeviceArray.from_numpy(h_in)
            d_out = DeviceArray.empty(1, dtype=np.int32)

            def thread(barrier):
                barrier.wait()
                reducer = cc.make_reduce_into(
                    d_in=d_in,
                    d_out=d_out,
                    op=shared_op,
                    h_init=h_init,
                )
                returned_reducers[worker_id] = reducer
                _call_with_temp(
                    reducer,
                    d_in=d_in,
                    d_out=d_out,
                    op=shared_op,
                    h_init=h_init,
                    num_items=h_in.size,
                    stream=stream,
                )
                stream.sync()
                expected = int(h_in.sum(dtype=np.int64) + h_init[0])
                assert int(d_out.copy_to_host()[0]) == expected

            return thread

        _run_threaded([make_thread(worker_id) for worker_id in range(STRESS_THREADS)])

        assert len({id(reducer) for reducer in returned_reducers}) == len(
            returned_reducers
        )
        assert (
            len({id(_get_build_result(reducer)) for reducer in returned_reducers}) == 1
        )


@dataclass(frozen=True)
class _IteratorCase:
    name: str
    make_iterator: Callable
    dtype: np.dtype
    num_items: int
    expected_sum: int

    def __str__(self):
        return self.name


@dataclass(frozen=True)
class _ColdTransformCase:
    name: str
    make_worker: Callable
    make_transformer: Callable
    run: Callable
    check: Callable

    def __str__(self):
        return self.name


def _run_cold_transform_native_cache_case(cc, case: _ColdTransformCase) -> None:
    for iteration in range(STRESS_ITERATIONS):
        cc.clear_all_caches()
        workers = [
            case.make_worker(cc, worker_id=worker_id, iteration=iteration)
            for worker_id in range(TRANSFORM_NATIVE_CACHE_THREADS)
        ]
        returned_algorithms = [None] * TRANSFORM_NATIVE_CACHE_THREADS
        # Transform's native launch config cache is filled on first execution,
        # so build wrappers first and synchronize the first call separately.
        execute_barrier = threading.Barrier(TRANSFORM_NATIVE_CACHE_THREADS)

        def make_thread(worker_id, worker):
            def thread(barrier):
                barrier.wait()
                try:
                    algorithm = case.make_transformer(cc, worker)
                    returned_algorithms[worker_id] = algorithm
                except BaseException:
                    execute_barrier.abort()
                    raise

                execute_barrier.wait(timeout=60)
                case.run(cc, algorithm, worker)
                case.check(cc, worker)

            return thread

        _run_threaded(
            [make_thread(worker_id, worker) for worker_id, worker in enumerate(workers)]
        )

        assert len({id(algorithm) for algorithm in returned_algorithms}) == len(
            returned_algorithms
        )
        assert (
            len({id(_get_build_result(algorithm)) for algorithm in returned_algorithms})
            == 1
        )


@pytest.mark.parametrize(
    "case",
    [
        _ColdTransformCase(
            "unary_transform",
            _make_unary_worker,
            _make_unary_for_worker,
            _run_unary,
            _check_unary,
        ),
        _ColdTransformCase(
            "binary_transform",
            _make_binary_worker,
            _make_binary_for_worker,
            _run_binary,
            _check_binary,
        ),
    ],
    ids=str,
)
def test_cold_transform_native_cache_initialization_stress(compute_module, case):
    cc = compute_module

    _run_cold_transform_native_cache_case(cc, case)


@dataclass(frozen=True)
class _V2FirstCallCase:
    name: str
    make_worker: Callable
    make_algorithm: Callable
    run_empty: Callable
    run: Callable
    check: Callable

    def __str__(self):
        return self.name


def _run_v2_first_call_gate_case(cc, case: _V2FirstCallCase) -> None:
    for iteration in range(STRESS_ITERATIONS):
        cc.clear_all_caches()
        workers = [
            case.make_worker(cc, worker_id=worker_id, iteration=iteration)
            for worker_id in range(TRANSFORM_NATIVE_CACHE_THREADS)
        ]
        returned_algorithms = [None] * TRANSFORM_NATIVE_CACHE_THREADS
        algorithms_built_barrier = threading.Barrier(TRANSFORM_NATIVE_CACHE_THREADS)
        nonempty_call_barrier = threading.Barrier(TRANSFORM_NATIVE_CACHE_THREADS)

        def make_thread(worker_id, worker):
            def thread(barrier):
                barrier.wait()
                try:
                    algorithm = case.make_algorithm(cc, worker)
                    returned_algorithms[worker_id] = algorithm
                    algorithms_built_barrier.wait(timeout=60)

                    # Empty calls return before CUB initializes its static launch
                    # configuration and therefore must not complete the gate.
                    if worker_id == 0:
                        case.run_empty(cc, algorithm, worker)

                    nonempty_call_barrier.wait(timeout=60)
                    case.run(cc, algorithm, worker)
                    case.check(cc, worker)
                except BaseException:
                    algorithms_built_barrier.abort()
                    nonempty_call_barrier.abort()
                    raise

            return thread

        _run_threaded(
            [make_thread(worker_id, worker) for worker_id, worker in enumerate(workers)]
        )

        assert len({id(algorithm) for algorithm in returned_algorithms}) == len(
            returned_algorithms
        )
        assert (
            len({id(_get_build_result(algorithm)) for algorithm in returned_algorithms})
            == 1
        )


_V2_FIRST_CALL_CASES = [
    _V2FirstCallCase(
        "unary_transform",
        _make_unary_worker,
        _make_unary_for_worker,
        _run_unary_empty,
        _run_unary,
        _check_unary,
    ),
    _V2FirstCallCase(
        "binary_transform",
        _make_binary_worker,
        _make_binary_for_worker,
        _run_binary_empty,
        _run_binary,
        _check_binary,
    ),
    _V2FirstCallCase(
        "lower_bound",
        _make_binary_search_worker,
        _make_lower_bound_for_worker,
        _run_binary_search_empty,
        _run_binary_search,
        _check_lower_bound,
    ),
    _V2FirstCallCase(
        "upper_bound",
        _make_binary_search_worker,
        _make_upper_bound_for_worker,
        _run_binary_search_empty,
        _run_binary_search,
        _check_upper_bound,
    ),
]


def _run_concurrent_cold_llvm_initialization():
    """Body of test_v2_concurrent_cold_llvm_initialization, run in a fresh process."""
    import cuda.compute as cc

    cc.clear_all_caches()
    workers = [
        case.make_worker(cc, worker_id=worker_id, iteration=0)
        for worker_id, case in enumerate(_V2_FIRST_CALL_CASES)
    ]
    returned_algorithms = [None] * len(_V2_FIRST_CALL_CASES)
    build_barrier = threading.Barrier(len(_V2_FIRST_CALL_CASES))

    def make_thread(worker_id, case, worker):
        def thread():
            build_barrier.wait(timeout=60)
            algorithm = case.make_algorithm(cc, worker)
            returned_algorithms[worker_id] = algorithm
            case.run(cc, algorithm, worker)
            case.check(cc, worker)

        return thread

    try:
        # Distinct algorithm keys prevent the Python build cache from
        # coalescing these concurrent HostJIT compiler initializations.
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(_V2_FIRST_CALL_CASES)
        ) as executor:
            futures = [
                executor.submit(make_thread(worker_id, case, worker))
                for worker_id, (case, worker) in enumerate(
                    zip(_V2_FIRST_CALL_CASES, workers)
                )
            ]
            for future in futures:
                future.result()
    finally:
        cc.clear_all_caches()

    assert len(
        {id(_get_build_result(algorithm)) for algorithm in returned_algorithms}
    ) == len(returned_algorithms)


def test_v2_concurrent_cold_llvm_initialization():
    from cuda.compute._build_info import USING_V2

    if not USING_V2:
        pytest.skip("requires the C Parallel v2 backend")

    import os
    import subprocess

    # LLVM target registration is process-wide and only cold once, so earlier
    # tests in this pytest process would leave it warm and this test would no
    # longer exercise concurrent *cold* initialization. Run the storm in a
    # fresh interpreter regardless of what already ran here. The tests root
    # must be on sys.path for the _utils.device_array import (pytest's
    # pythonpath setting does not apply to a raw subprocess).
    compute_dir = os.path.dirname(os.path.abspath(__file__))
    tests_dir = os.path.dirname(compute_dir)
    code = (
        "import sys; "
        f"sys.path.insert(0, {tests_dir!r}); "
        f"sys.path.insert(0, {compute_dir!r}); "
        "import test_free_threading_stress as m; "
        "m._run_concurrent_cold_llvm_initialization()"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, (
        f"cold LLVM initialization subprocess failed "
        f"(exit {result.returncode})\nstdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


@pytest.mark.parametrize("case", _V2_FIRST_CALL_CASES, ids=str)
def test_v2_first_call_gate_stress(compute_module, case):
    cc = compute_module

    from cuda.compute._build_info import USING_V2

    if not USING_V2:
        pytest.skip("requires the C Parallel v2 backend")

    _run_v2_first_call_gate_case(cc, case)


def _iterator_counting(cc):
    return cc.CountingIterator(np.int32(0)), np.dtype(np.int32), 32, sum(range(32))


def _iterator_permutation(cc):
    h_values = np.arange(32, dtype=np.int32)
    h_indices = np.arange(31, -1, -1, dtype=np.int32)
    d_values = DeviceArray.from_numpy(h_values)
    d_indices = DeviceArray.from_numpy(h_indices)
    return (
        cc.PermutationIterator(d_values, d_indices),
        h_values.dtype,
        h_indices.size,
        int(h_values[h_indices].sum()),
    )


def _iterator_transform(cc):
    from cuda.compute import types
    from cuda.compute._cpp_compile import compile_cpp_op_code
    from cuda.compute.op import RawOp

    num_items = 32
    source = """
    extern "C" __device__ void negate_i32(void* input, void* result) {
        *static_cast<int*>(result) = -*static_cast<int*>(input);
    }
    """
    op = RawOp(ltoir=compile_cpp_op_code(source), name="negate_i32")
    return (
        cc.TransformIterator(
            cc.CountingIterator(np.int32(0)), op, value_type=types.int32
        ),
        np.dtype(np.int32),
        num_items,
        -sum(range(num_items)),
    )


# A representative subset: the shared-iterator race is IteratorBase's per-instance
# lazy Op construction (guarded by _op_lock), which is identical for every
# iterator type, so these cover the structural variety rather than every iterator
# — a leaf iterator, a compound iterator that wraps a child and memoizes a
# compiled op (transform), and a compound iterator with two children
# (permutation). Per-iterator correctness lives in the dedicated iterator tests.
ITERATOR_FACTORIES = [
    _iterator_counting,
    _iterator_permutation,
    _iterator_transform,
]


@pytest.mark.parametrize(
    "make_iterator",
    ITERATOR_FACTORIES,
    ids=lambda fn: fn.__name__.removeprefix("_iterator_"),
)
def test_shared_iterator_object_stress(compute_module, make_iterator):
    cc = compute_module

    device = Device()
    device.set_current()
    device.sync()

    for iteration in range(STRESS_ITERATIONS):
        cc.clear_all_caches()
        # Recreate the shared iterator every iteration: its lazy advance/deref
        # Op construction (guarded by IteratorBase._op_lock) is only racy while
        # the per-instance caches are cold, so a fresh instance re-arms that
        # first-construction race on each pass instead of only once per test.
        shared_iterator, dtype, num_items, expected_sum = make_iterator(cc)

        def make_thread(worker_id):
            stream = _make_stream()
            h_init = np.array([worker_id], dtype=dtype)
            d_out = DeviceArray.empty(1, dtype=dtype)

            def thread(barrier):
                barrier.wait()
                reducer = cc.make_reduce_into(
                    d_in=shared_iterator,
                    d_out=d_out,
                    op=cc.OpKind.PLUS,
                    h_init=h_init,
                )
                _call_with_temp(
                    reducer,
                    d_in=shared_iterator,
                    d_out=d_out,
                    op=cc.OpKind.PLUS,
                    h_init=h_init,
                    num_items=num_items,
                    stream=stream,
                )
                stream.sync()
                assert int(d_out.copy_to_host()[0]) == int(expected_sum + h_init[0])

            return thread

        _run_threaded([make_thread(worker_id) for worker_id in range(STRESS_THREADS)])


def test_runtime_ownership_isolation(compute_module):
    cc = compute_module

    def make_thread(worker_id):
        def thread(barrier):
            barrier.wait()
            stream = _make_stream()
            h_in = np.arange(16, dtype=np.int32) + worker_id * 10
            h_init = np.array([worker_id], dtype=np.int32)

            d_in = DeviceArray.from_numpy(h_in)
            d_reduce_out = DeviceArray.empty(1, dtype=np.int32)
            d_scan_out = DeviceArray.empty(h_in.shape, h_in.dtype)
            d_transform_out = DeviceArray.empty(h_in.shape, h_in.dtype)
            d_hist = DeviceArray.from_numpy(np.zeros(4, dtype=np.int32))
            h_keys = np.array([3, 1, 2, 1], dtype=np.uint32) + worker_id
            d_keys_in = DeviceArray.from_numpy(h_keys)
            d_keys_tmp = DeviceArray.empty(h_keys.shape, h_keys.dtype)

            cc.reduce_into(
                d_in=d_in,
                d_out=d_reduce_out,
                num_items=h_in.size,
                op=cc.OpKind.PLUS,
                h_init=h_init,
                stream=stream,
            )
            cc.exclusive_scan(
                d_in=d_in,
                d_out=d_scan_out,
                op=cc.OpKind.PLUS,
                init_value=h_init,
                num_items=h_in.size,
                stream=stream,
            )
            cc.unary_transform(
                d_in=d_in,
                d_out=d_transform_out,
                op=cc.OpKind.NEGATE,
                num_items=h_in.size,
                stream=stream,
            )
            cc.histogram_even(
                d_samples=d_in,
                d_histogram=d_hist,
                num_output_levels=5,
                lower_level=np.int32(worker_id * 10),
                upper_level=np.int32(worker_id * 10 + 16),
                num_samples=h_in.size,
                stream=stream,
            )
            keys = cc.DoubleBuffer(d_keys_in, d_keys_tmp)
            cc.radix_sort(
                d_in_keys=keys,
                d_out_keys=None,
                d_in_values=None,
                d_out_values=None,
                num_items=h_keys.size,
                order=cc.SortOrder.ASCENDING,
                stream=stream,
            )

            stream.sync()
            assert int(d_reduce_out.copy_to_host()[0]) == int(h_in.sum() + worker_id)
            expected_scan = np.empty_like(h_in)
            expected_scan[0] = worker_id
            expected_scan[1:] = worker_id + np.cumsum(h_in[:-1])
            np.testing.assert_array_equal(d_scan_out.copy_to_host(), expected_scan)
            np.testing.assert_array_equal(d_transform_out.copy_to_host(), -h_in)
            assert int(d_hist.copy_to_host().sum()) == h_in.size
            np.testing.assert_array_equal(
                keys.current().copy_to_host(), np.sort(h_keys)
            )

        return thread

    for _ in range(STRESS_ITERATIONS):
        _run_threaded([make_thread(worker_id) for worker_id in range(STRESS_THREADS)])
