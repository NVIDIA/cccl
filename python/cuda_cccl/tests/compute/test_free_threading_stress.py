# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import concurrent.futures
import os
import sys
import sysconfig
import threading
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pytest


pytestmark = [
    pytest.mark.free_threading,
    pytest.mark.no_numba,
    pytest.mark.no_verify_sass(
        reason="Free-threading stress tests intentionally run concurrent workers."
    ),
]

STRESS_ITERATIONS = int(os.environ.get("CCCL_FREE_THREADING_STRESS_ITERATIONS", "10"))
STRESS_THREADS = int(os.environ.get("CCCL_FREE_THREADING_STRESS_THREADS", "2"))
TRANSFORM_NATIVE_CACHE_THREADS = int(
    os.environ.get(
        "CCCL_FREE_THREADING_TRANSFORM_NATIVE_CACHE_THREADS",
        str(max(STRESS_THREADS, 4)),
    )
)


def _is_free_threaded_build() -> bool:
    return sysconfig.get_config_var("Py_GIL_DISABLED") in (1, "1")


def _assert_gil_disabled(where: str) -> None:
    is_gil_enabled = getattr(sys, "_is_gil_enabled", None)
    if is_gil_enabled is not None and is_gil_enabled():
        pytest.fail(f"the GIL is enabled {where}")


def _require_free_threaded_python() -> None:
    if not _is_free_threaded_build():
        pytest.skip("requires a free-threaded CPython build")
    _assert_gil_disabled("before importing cuda.compute")


@pytest.fixture
def compute_modules():
    _require_free_threaded_python()

    import cupy as cp

    _assert_gil_disabled("after importing cupy")

    import cuda.compute as cc

    _assert_gil_disabled("after importing cuda.compute")
    cc.clear_all_caches()
    try:
        yield cp, cc
    finally:
        cc.clear_all_caches()


class _CudaStream:
    def __init__(self, stream):
        self.stream = stream

    def __cuda_stream__(self):
        return (0, self.stream.ptr)

    @property
    def ptr(self):
        return self.stream.ptr


def _make_stream(cp):
    stream = cp.cuda.Stream()
    return stream, _CudaStream(stream)


def _run_threaded(workers: list[Callable[[threading.Barrier], None]]) -> None:
    barrier = threading.Barrier(len(workers))
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(workers)) as executor:
        futures = [executor.submit(worker, barrier) for worker in workers]
        for future in futures:
            future.result()
    _assert_gil_disabled("after concurrent cuda.compute operations")


def _call_with_temp(cp, algorithm, **kwargs):
    temp_storage_bytes = algorithm(temp_storage=None, **kwargs)
    temp_storage = cp.empty(temp_storage_bytes, dtype=np.uint8)
    return algorithm(temp_storage=temp_storage, **kwargs)


def _get_build_result(algorithm):
    if hasattr(algorithm, "build_result"):
        return algorithm.build_result
    if hasattr(algorithm, "partitioner"):
        return _get_build_result(algorithm.partitioner)
    raise AssertionError(f"{type(algorithm).__name__} does not expose a build result")


def _selected_segments(keys, values, starts, ends, descending=False):
    out_keys = keys.copy()
    out_values = values.copy()
    for start, end in zip(starts, ends):
        segment_keys = keys[start:end]
        order = np.argsort(segment_keys, kind="stable")
        if descending:
            order = order[::-1]
        out_keys[start:end] = segment_keys[order]
        out_values[start:end] = values[start:end][order]
    return out_keys, out_values


@dataclass(frozen=True)
class _AlgorithmCase:
    name: str
    make_shared: Callable
    make_worker: Callable
    run: Callable
    check: Callable

    def __str__(self):
        return self.name


def _run_thread_local_algorithm_case(cp, cc, case: _AlgorithmCase) -> None:
    warm_algorithm = case.make_shared(cp, cc)

    warm_worker = case.make_worker(cp, cc, worker_id=0, iteration=-1)
    case.run(cp, cc, warm_algorithm, warm_worker)
    case.check(cp, cc, warm_worker)

    for iteration in range(STRESS_ITERATIONS):
        worker_state = [
            case.make_worker(cp, cc, worker_id=worker_id, iteration=iteration)
            for worker_id in range(STRESS_THREADS)
        ]
        returned_algorithms = [None] * STRESS_THREADS

        def make_thread(worker_id, worker):
            def thread(barrier):
                barrier.wait()
                algorithm = case.make_shared(cp, cc)
                returned_algorithms[worker_id] = algorithm
                case.run(cp, cc, algorithm, worker)
                case.check(cp, cc, worker)

            return thread

        _run_threaded(
            [make_thread(worker_id, worker) for worker_id, worker in enumerate(worker_state)]
        )

        assert len({id(algorithm) for algorithm in returned_algorithms}) == len(
            returned_algorithms
        )
        assert len(
            {id(_get_build_result(algorithm)) for algorithm in returned_algorithms}
        ) == 1


def _make_reduce_worker(cp, cc, worker_id, iteration):
    stream, cuda_stream = _make_stream(cp)
    h_in = np.arange(64, dtype=np.int32) + worker_id * 101 + iteration
    h_init = np.array([7 + worker_id], dtype=np.int32)
    with stream:
        d_in = cp.asarray(h_in)
        d_out = cp.empty(1, dtype=np.int32)
    return {
        "stream": stream,
        "cuda_stream": cuda_stream,
        "h_in": h_in,
        "d_in": d_in,
        "d_out": d_out,
        "h_init": h_init,
    }


def _make_reduce_shared(cp, cc):
    worker = _make_reduce_worker(cp, cc, 0, -1)
    return cc.make_reduce_into(
        d_in=worker["d_in"],
        d_out=worker["d_out"],
        op=cc.OpKind.PLUS,
        h_init=worker["h_init"],
    )


def _run_reduce(cp, cc, reducer, worker):
    _call_with_temp(
        cp,
        reducer,
        d_in=worker["d_in"],
        d_out=worker["d_out"],
        op=cc.OpKind.PLUS,
        h_init=worker["h_init"],
        num_items=worker["h_in"].size,
        stream=worker["cuda_stream"],
    )


def _check_reduce(cp, cc, worker):
    worker["stream"].synchronize()
    expected = worker["h_in"].sum(dtype=np.int64) + int(worker["h_init"][0])
    assert int(worker["d_out"].get()[0]) == int(expected)


def _make_unary_worker(cp, cc, worker_id, iteration):
    stream, cuda_stream = _make_stream(cp)
    h_in = np.arange(32, dtype=np.int32) + worker_id * 17 + iteration
    with stream:
        d_in = cp.asarray(h_in)
        d_out = cp.empty_like(d_in)
    return {
        "stream": stream,
        "cuda_stream": cuda_stream,
        "h_in": h_in,
        "d_in": d_in,
        "d_out": d_out,
    }


def _make_unary_shared(cp, cc):
    worker = _make_unary_worker(cp, cc, 0, -1)
    return cc.make_unary_transform(
        d_in=worker["d_in"], d_out=worker["d_out"], op=cc.OpKind.NEGATE
    )


def _make_unary_for_worker(cp, cc, worker):
    return cc.make_unary_transform(
        d_in=worker["d_in"], d_out=worker["d_out"], op=cc.OpKind.NEGATE
    )


def _run_unary(cp, cc, transformer, worker):
    transformer(
        d_in=worker["d_in"],
        d_out=worker["d_out"],
        op=cc.OpKind.NEGATE,
        num_items=worker["h_in"].size,
        stream=worker["cuda_stream"],
    )


def _check_unary(cp, cc, worker):
    worker["stream"].synchronize()
    np.testing.assert_array_equal(worker["d_out"].get(), -worker["h_in"])


def _make_binary_worker(cp, cc, worker_id, iteration):
    stream, cuda_stream = _make_stream(cp)
    h_in1 = np.arange(32, dtype=np.int32) + worker_id * 13
    h_in2 = np.arange(32, dtype=np.int32) + iteration * 7
    with stream:
        d_in1 = cp.asarray(h_in1)
        d_in2 = cp.asarray(h_in2)
        d_out = cp.empty_like(d_in1)
    return {
        "stream": stream,
        "cuda_stream": cuda_stream,
        "h_in1": h_in1,
        "h_in2": h_in2,
        "d_in1": d_in1,
        "d_in2": d_in2,
        "d_out": d_out,
    }


def _make_binary_shared(cp, cc):
    worker = _make_binary_worker(cp, cc, 0, -1)
    return cc.make_binary_transform(
        d_in1=worker["d_in1"],
        d_in2=worker["d_in2"],
        d_out=worker["d_out"],
        op=cc.OpKind.PLUS,
    )


def _make_binary_for_worker(cp, cc, worker):
    return cc.make_binary_transform(
        d_in1=worker["d_in1"],
        d_in2=worker["d_in2"],
        d_out=worker["d_out"],
        op=cc.OpKind.PLUS,
    )


def _run_binary(cp, cc, transformer, worker):
    transformer(
        d_in1=worker["d_in1"],
        d_in2=worker["d_in2"],
        d_out=worker["d_out"],
        op=cc.OpKind.PLUS,
        num_items=worker["h_in1"].size,
        stream=worker["cuda_stream"],
    )


def _check_binary(cp, cc, worker):
    worker["stream"].synchronize()
    np.testing.assert_array_equal(worker["d_out"].get(), worker["h_in1"] + worker["h_in2"])


def _make_scan_worker(cp, cc, worker_id, iteration):
    stream, cuda_stream = _make_stream(cp)
    h_in = np.arange(1, 33, dtype=np.int32) + worker_id + iteration
    h_init = np.array([3 + worker_id], dtype=np.int32)
    with stream:
        d_in = cp.asarray(h_in)
        d_out = cp.empty_like(d_in)
    return {
        "stream": stream,
        "cuda_stream": cuda_stream,
        "h_in": h_in,
        "h_init": h_init,
        "d_in": d_in,
        "d_out": d_out,
    }


def _make_exclusive_scan_shared(cp, cc):
    worker = _make_scan_worker(cp, cc, 0, -1)
    return cc.make_exclusive_scan(
        d_in=worker["d_in"],
        d_out=worker["d_out"],
        op=cc.OpKind.PLUS,
        init_value=worker["h_init"],
    )


def _make_inclusive_scan_shared(cp, cc):
    worker = _make_scan_worker(cp, cc, 0, -1)
    return cc.make_inclusive_scan(
        d_in=worker["d_in"],
        d_out=worker["d_out"],
        op=cc.OpKind.PLUS,
        init_value=worker["h_init"],
    )


def _run_scan(cp, cc, scanner, worker):
    _call_with_temp(
        cp,
        scanner,
        d_in=worker["d_in"],
        d_out=worker["d_out"],
        op=cc.OpKind.PLUS,
        init_value=worker["h_init"],
        num_items=worker["h_in"].size,
        stream=worker["cuda_stream"],
    )


def _check_exclusive_scan(cp, cc, worker):
    worker["stream"].synchronize()
    expected = np.empty_like(worker["h_in"])
    expected[0] = worker["h_init"][0]
    expected[1:] = worker["h_init"][0] + np.cumsum(worker["h_in"][:-1])
    np.testing.assert_array_equal(worker["d_out"].get(), expected)


def _check_inclusive_scan(cp, cc, worker):
    worker["stream"].synchronize()
    expected = worker["h_init"][0] + np.cumsum(worker["h_in"])
    np.testing.assert_array_equal(worker["d_out"].get(), expected)


def _make_segmented_reduce_worker(cp, cc, worker_id, iteration):
    stream, cuda_stream = _make_stream(cp)
    h_in = np.arange(1, 17, dtype=np.int32) + worker_id * 3 + iteration
    h_start_offsets = np.array([0, 3, 8, 12], dtype=np.int32)
    h_end_offsets = np.array([3, 8, 12, 16], dtype=np.int32)
    h_init = np.array([worker_id], dtype=np.int32)
    with stream:
        d_in = cp.asarray(h_in)
        d_out = cp.empty(len(h_start_offsets), dtype=np.int32)
        d_start_offsets = cp.asarray(h_start_offsets)
        d_end_offsets = cp.asarray(h_end_offsets)
    return {
        "stream": stream,
        "cuda_stream": cuda_stream,
        "h_in": h_in,
        "h_start_offsets": h_start_offsets,
        "h_end_offsets": h_end_offsets,
        "h_init": h_init,
        "d_in": d_in,
        "d_out": d_out,
        "d_start_offsets": d_start_offsets,
        "d_end_offsets": d_end_offsets,
    }


def _make_segmented_reduce_shared(cp, cc):
    worker = _make_segmented_reduce_worker(cp, cc, 0, -1)
    return cc.make_segmented_reduce(
        d_in=worker["d_in"],
        d_out=worker["d_out"],
        start_offsets_in=worker["d_start_offsets"],
        end_offsets_in=worker["d_end_offsets"],
        op=cc.OpKind.PLUS,
        h_init=worker["h_init"],
    )


def _run_segmented_reduce(cp, cc, reducer, worker):
    _call_with_temp(
        cp,
        reducer,
        d_in=worker["d_in"],
        d_out=worker["d_out"],
        num_segments=len(worker["h_start_offsets"]),
        start_offsets_in=worker["d_start_offsets"],
        end_offsets_in=worker["d_end_offsets"],
        op=cc.OpKind.PLUS,
        h_init=worker["h_init"],
        stream=worker["cuda_stream"],
    )


def _check_segmented_reduce(cp, cc, worker):
    worker["stream"].synchronize()
    expected = np.array(
        [
            worker["h_in"][start:end].sum() + worker["h_init"][0]
            for start, end in zip(worker["h_start_offsets"], worker["h_end_offsets"])
        ],
        dtype=np.int32,
    )
    np.testing.assert_array_equal(worker["d_out"].get(), expected)


def _make_histogram_worker(cp, cc, worker_id, iteration):
    stream, cuda_stream = _make_stream(cp)
    lower = np.float32(worker_id * 10)
    upper = np.float32(lower + 8)
    h_samples = np.array(
        [
            lower + 0.5,
            lower + 1.5,
            lower + 2.0,
            lower + 3.5,
            lower + 6.0,
            upper + 1.0,
        ],
        dtype=np.float32,
    )
    h_num_levels = np.array([5], dtype=np.int32)
    h_lower = np.array([lower], dtype=np.float32)
    h_upper = np.array([upper], dtype=np.float32)
    with stream:
        d_samples = cp.asarray(h_samples)
        d_histogram = cp.zeros(h_num_levels[0] - 1, dtype=np.int32)
    return {
        "stream": stream,
        "cuda_stream": cuda_stream,
        "h_samples": h_samples,
        "h_num_levels": h_num_levels,
        "h_lower": h_lower,
        "h_upper": h_upper,
        "d_samples": d_samples,
        "d_histogram": d_histogram,
    }


def _make_histogram_shared(cp, cc):
    worker = _make_histogram_worker(cp, cc, 0, -1)
    return cc.make_histogram_even(
        d_samples=worker["d_samples"],
        d_histogram=worker["d_histogram"],
        h_num_output_levels=worker["h_num_levels"],
        h_lower_level=worker["h_lower"],
        h_upper_level=worker["h_upper"],
        num_samples=worker["h_samples"].size,
    )


def _run_histogram(cp, cc, histogrammer, worker):
    with worker["stream"]:
        worker["d_histogram"].fill(0)
    _call_with_temp(
        cp,
        histogrammer,
        d_samples=worker["d_samples"],
        d_histogram=worker["d_histogram"],
        h_num_output_levels=worker["h_num_levels"],
        h_lower_level=worker["h_lower"],
        h_upper_level=worker["h_upper"],
        num_samples=worker["h_samples"].size,
        stream=worker["cuda_stream"],
    )


def _check_histogram(cp, cc, worker):
    worker["stream"].synchronize()
    expected, _ = np.histogram(
        worker["h_samples"],
        bins=int(worker["h_num_levels"][0] - 1),
        range=(float(worker["h_lower"][0]), float(worker["h_upper"][0])),
    )
    np.testing.assert_array_equal(worker["d_histogram"].get(), expected.astype(np.int32))


def _make_binary_search_worker(cp, cc, worker_id, iteration):
    stream, cuda_stream = _make_stream(cp)
    h_data = np.array([90, 70, 50, 30, 10], dtype=np.int32) - worker_id
    h_values = np.array([95, 70, 45, 10, 5], dtype=np.int32) - worker_id
    with stream:
        d_data = cp.asarray(h_data)
        d_values = cp.asarray(h_values)
        d_out = cp.empty(h_values.size, dtype=np.uintp)
    return {
        "stream": stream,
        "cuda_stream": cuda_stream,
        "h_data": h_data,
        "h_values": h_values,
        "d_data": d_data,
        "d_values": d_values,
        "d_out": d_out,
    }


def _make_lower_bound_shared(cp, cc):
    worker = _make_binary_search_worker(cp, cc, 0, -1)
    return cc.make_lower_bound(
        d_data=worker["d_data"],
        d_values=worker["d_values"],
        d_out=worker["d_out"],
        comp=cc.OpKind.GREATER,
    )


def _make_upper_bound_shared(cp, cc):
    worker = _make_binary_search_worker(cp, cc, 0, -1)
    return cc.make_upper_bound(
        d_data=worker["d_data"],
        d_values=worker["d_values"],
        d_out=worker["d_out"],
        comp=cc.OpKind.GREATER,
    )


def _run_binary_search(cp, cc, searcher, worker):
    searcher(
        d_data=worker["d_data"],
        num_items=worker["h_data"].size,
        d_values=worker["d_values"],
        num_values=worker["h_values"].size,
        d_out=worker["d_out"],
        comp=cc.OpKind.GREATER,
        stream=worker["cuda_stream"],
    )


def _check_lower_bound(cp, cc, worker):
    worker["stream"].synchronize()
    expected = np.searchsorted(-worker["h_data"], -worker["h_values"], side="left")
    np.testing.assert_array_equal(worker["d_out"].get(), expected.astype(np.uintp))


def _check_upper_bound(cp, cc, worker):
    worker["stream"].synchronize()
    expected = np.searchsorted(-worker["h_data"], -worker["h_values"], side="right")
    np.testing.assert_array_equal(worker["d_out"].get(), expected.astype(np.uintp))


def _make_select_worker(cp, cc, worker_id, iteration):
    stream, cuda_stream = _make_stream(cp)
    h_in = np.array(
        [True, False, worker_id % 2 == 0, True, False, iteration % 2 == 0],
        dtype=np.bool_,
    )
    with stream:
        d_in = cp.asarray(h_in)
        d_out = cp.empty_like(d_in)
        d_count = cp.empty(2, dtype=np.uint64)
    return {
        "stream": stream,
        "cuda_stream": cuda_stream,
        "h_in": h_in,
        "d_in": d_in,
        "d_out": d_out,
        "d_count": d_count,
    }


def _make_select_shared(cp, cc):
    worker = _make_select_worker(cp, cc, 0, -1)
    return cc.make_select(
        d_in=worker["d_in"],
        d_out=worker["d_out"],
        d_num_selected_out=worker["d_count"],
        cond=cc.OpKind.IDENTITY,
    )


def _run_select(cp, cc, selector, worker):
    _call_with_temp(
        cp,
        selector,
        d_in=worker["d_in"],
        d_out=worker["d_out"],
        d_num_selected_out=worker["d_count"],
        cond=cc.OpKind.IDENTITY,
        num_items=worker["h_in"].size,
        stream=worker["cuda_stream"],
    )


def _check_select(cp, cc, worker):
    worker["stream"].synchronize()
    count = int(worker["d_count"].get()[0])
    expected = worker["h_in"][worker["h_in"]]
    assert count == expected.size
    np.testing.assert_array_equal(worker["d_out"].get()[:count], expected)


def _make_three_way_shared(cp, cc):
    worker = _make_select_worker(cp, cc, 0, -1)
    d_unselected = cp.empty_like(worker["d_in"])
    return cc.make_three_way_partition(
        d_in=worker["d_in"],
        d_first_part_out=worker["d_out"],
        d_second_part_out=d_unselected,
        d_unselected_out=cp.empty_like(worker["d_in"]),
        d_num_selected_out=worker["d_count"],
        select_first_part_op=cc.OpKind.IDENTITY,
        select_second_part_op=cc.OpKind.LOGICAL_NOT,
    )


def _make_three_way_worker(cp, cc, worker_id, iteration):
    worker = _make_select_worker(cp, cc, worker_id, iteration)
    stream = worker["stream"]
    with stream:
        worker["d_second_out"] = cp.empty_like(worker["d_in"])
        worker["d_unselected"] = cp.empty_like(worker["d_in"])
    return worker


def _run_three_way(cp, cc, partitioner, worker):
    _call_with_temp(
        cp,
        partitioner,
        d_in=worker["d_in"],
        d_first_part_out=worker["d_out"],
        d_second_part_out=worker["d_second_out"],
        d_unselected_out=worker["d_unselected"],
        d_num_selected_out=worker["d_count"],
        select_first_part_op=cc.OpKind.IDENTITY,
        select_second_part_op=cc.OpKind.LOGICAL_NOT,
        num_items=worker["h_in"].size,
        stream=worker["cuda_stream"],
    )


def _check_three_way(cp, cc, worker):
    worker["stream"].synchronize()
    counts = worker["d_count"].get()
    true_count = int(np.count_nonzero(worker["h_in"]))
    false_count = int(worker["h_in"].size - true_count)
    assert int(counts[0]) == true_count
    assert int(counts[1]) == false_count
    np.testing.assert_array_equal(
        worker["d_out"].get()[:true_count], np.ones(true_count, dtype=np.bool_)
    )
    np.testing.assert_array_equal(
        worker["d_second_out"].get()[:false_count], np.zeros(false_count, dtype=np.bool_)
    )


def _make_unique_worker(cp, cc, worker_id, iteration):
    stream, cuda_stream = _make_stream(cp)
    base = worker_id * 10 + iteration
    h_keys = np.array([base, base, base + 1, base + 2, base + 2, base + 3], dtype=np.int32)
    h_items = np.arange(h_keys.size, dtype=np.int32) + worker_id * 100
    with stream:
        d_in_keys = cp.asarray(h_keys)
        d_in_items = cp.asarray(h_items)
        d_out_keys = cp.empty_like(d_in_keys)
        d_out_items = cp.empty_like(d_in_items)
        d_count = cp.empty(1, dtype=np.int32)
    return {
        "stream": stream,
        "cuda_stream": cuda_stream,
        "h_keys": h_keys,
        "h_items": h_items,
        "d_in_keys": d_in_keys,
        "d_in_items": d_in_items,
        "d_out_keys": d_out_keys,
        "d_out_items": d_out_items,
        "d_count": d_count,
    }


def _make_unique_shared(cp, cc):
    worker = _make_unique_worker(cp, cc, 0, -1)
    return cc.make_unique_by_key(
        d_in_keys=worker["d_in_keys"],
        d_in_items=worker["d_in_items"],
        d_out_keys=worker["d_out_keys"],
        d_out_items=worker["d_out_items"],
        d_out_num_selected=worker["d_count"],
        op=cc.OpKind.EQUAL_TO,
    )


def _run_unique(cp, cc, uniquer, worker):
    _call_with_temp(
        cp,
        uniquer,
        d_in_keys=worker["d_in_keys"],
        d_in_items=worker["d_in_items"],
        d_out_keys=worker["d_out_keys"],
        d_out_items=worker["d_out_items"],
        d_out_num_selected=worker["d_count"],
        op=cc.OpKind.EQUAL_TO,
        num_items=worker["h_keys"].size,
        stream=worker["cuda_stream"],
    )


def _check_unique(cp, cc, worker):
    worker["stream"].synchronize()
    selected = np.concatenate(([True], worker["h_keys"][1:] != worker["h_keys"][:-1]))
    expected_keys = worker["h_keys"][selected]
    expected_items = worker["h_items"][selected]
    count = int(worker["d_count"].get()[0])
    assert count == expected_keys.size
    np.testing.assert_array_equal(worker["d_out_keys"].get()[:count], expected_keys)
    np.testing.assert_array_equal(worker["d_out_items"].get()[:count], expected_items)


def _make_merge_sort_worker(cp, cc, worker_id, iteration):
    stream, cuda_stream = _make_stream(cp)
    h_keys = np.array([5, 1, 3, 1, 4, 2], dtype=np.int32) + worker_id * 10
    h_values = np.arange(h_keys.size, dtype=np.int32) + iteration * 100
    with stream:
        d_in_keys = cp.asarray(h_keys)
        d_in_values = cp.asarray(h_values)
        d_out_keys = cp.empty_like(d_in_keys)
        d_out_values = cp.empty_like(d_in_values)
    return {
        "stream": stream,
        "cuda_stream": cuda_stream,
        "h_keys": h_keys,
        "h_values": h_values,
        "d_in_keys": d_in_keys,
        "d_in_values": d_in_values,
        "d_out_keys": d_out_keys,
        "d_out_values": d_out_values,
    }


def _make_merge_sort_shared(cp, cc):
    worker = _make_merge_sort_worker(cp, cc, 0, -1)
    return cc.make_merge_sort(
        d_in_keys=worker["d_in_keys"],
        d_in_values=worker["d_in_values"],
        d_out_keys=worker["d_out_keys"],
        d_out_values=worker["d_out_values"],
        op=cc.OpKind.LESS,
    )


def _run_merge_sort(cp, cc, sorter, worker):
    _call_with_temp(
        cp,
        sorter,
        d_in_keys=worker["d_in_keys"],
        d_in_values=worker["d_in_values"],
        d_out_keys=worker["d_out_keys"],
        d_out_values=worker["d_out_values"],
        op=cc.OpKind.LESS,
        num_items=worker["h_keys"].size,
        stream=worker["cuda_stream"],
    )


def _check_merge_sort(cp, cc, worker):
    worker["stream"].synchronize()
    order = np.argsort(worker["h_keys"], kind="stable")
    np.testing.assert_array_equal(worker["d_out_keys"].get(), worker["h_keys"][order])
    np.testing.assert_array_equal(worker["d_out_values"].get(), worker["h_values"][order])


def _make_radix_sort_worker(cp, cc, worker_id, iteration):
    stream, cuda_stream = _make_stream(cp)
    h_keys = np.array([7, 3, 5, 3, 1, 9], dtype=np.uint32) + np.uint32(worker_id * 11)
    h_values = np.arange(h_keys.size, dtype=np.int32) + iteration * 10
    with stream:
        d_in_keys = cp.asarray(h_keys)
        d_tmp_keys = cp.empty_like(d_in_keys)
        d_in_values = cp.asarray(h_values)
        d_tmp_values = cp.empty_like(d_in_values)
    return {
        "stream": stream,
        "cuda_stream": cuda_stream,
        "h_keys": h_keys,
        "h_values": h_values,
        "keys": cc.DoubleBuffer(d_in_keys, d_tmp_keys),
        "values": cc.DoubleBuffer(d_in_values, d_tmp_values),
    }


def _make_radix_sort_shared(cp, cc):
    worker = _make_radix_sort_worker(cp, cc, 0, -1)
    return cc.make_radix_sort(
        d_in_keys=worker["keys"],
        d_out_keys=None,
        d_in_values=worker["values"],
        d_out_values=None,
        order=cc.SortOrder.ASCENDING,
    )


def _run_radix_sort(cp, cc, sorter, worker):
    _call_with_temp(
        cp,
        sorter,
        d_in_keys=worker["keys"],
        d_out_keys=None,
        d_in_values=worker["values"],
        d_out_values=None,
        num_items=worker["h_keys"].size,
        stream=worker["cuda_stream"],
    )


def _check_radix_sort(cp, cc, worker):
    worker["stream"].synchronize()
    order = np.argsort(worker["h_keys"], kind="stable")
    np.testing.assert_array_equal(worker["keys"].current().get(), worker["h_keys"][order])
    np.testing.assert_array_equal(worker["values"].current().get(), worker["h_values"][order])
    assert worker["keys"].selector == worker["values"].selector


def _make_segmented_sort_worker(cp, cc, worker_id, iteration):
    stream, cuda_stream = _make_stream(cp)
    h_keys = np.array([4, 2, 3, 8, 6, 7, 1, 5], dtype=np.int32) + worker_id * 13
    h_values = np.arange(h_keys.size, dtype=np.int32) + iteration * 100
    h_start_offsets = np.array([0, 3, 6], dtype=np.int32)
    h_end_offsets = np.array([3, 6, 8], dtype=np.int32)
    with stream:
        d_in_keys = cp.asarray(h_keys)
        d_tmp_keys = cp.empty_like(d_in_keys)
        d_in_values = cp.asarray(h_values)
        d_tmp_values = cp.empty_like(d_in_values)
        d_start_offsets = cp.asarray(h_start_offsets)
        d_end_offsets = cp.asarray(h_end_offsets)
    return {
        "stream": stream,
        "cuda_stream": cuda_stream,
        "h_keys": h_keys,
        "h_values": h_values,
        "h_start_offsets": h_start_offsets,
        "h_end_offsets": h_end_offsets,
        "keys": cc.DoubleBuffer(d_in_keys, d_tmp_keys),
        "values": cc.DoubleBuffer(d_in_values, d_tmp_values),
        "d_start_offsets": d_start_offsets,
        "d_end_offsets": d_end_offsets,
    }


def _make_segmented_sort_shared(cp, cc):
    worker = _make_segmented_sort_worker(cp, cc, 0, -1)
    return cc.make_segmented_sort(
        d_in_keys=worker["keys"],
        d_out_keys=None,
        d_in_values=worker["values"],
        d_out_values=None,
        start_offsets_in=worker["d_start_offsets"],
        end_offsets_in=worker["d_end_offsets"],
        order=cc.SortOrder.ASCENDING,
    )


def _run_segmented_sort(cp, cc, sorter, worker):
    _call_with_temp(
        cp,
        sorter,
        d_in_keys=worker["keys"],
        d_out_keys=None,
        d_in_values=worker["values"],
        d_out_values=None,
        num_items=worker["h_keys"].size,
        num_segments=worker["h_start_offsets"].size,
        start_offsets_in=worker["d_start_offsets"],
        end_offsets_in=worker["d_end_offsets"],
        stream=worker["cuda_stream"],
    )


def _check_segmented_sort(cp, cc, worker):
    worker["stream"].synchronize()
    expected_keys, expected_values = _selected_segments(
        worker["h_keys"],
        worker["h_values"],
        worker["h_start_offsets"],
        worker["h_end_offsets"],
    )
    np.testing.assert_array_equal(worker["keys"].current().get(), expected_keys)
    np.testing.assert_array_equal(worker["values"].current().get(), expected_values)
    assert worker["keys"].selector == worker["values"].selector


SHARED_ALGORITHM_CASES = [
    _AlgorithmCase("reduce", _make_reduce_shared, _make_reduce_worker, _run_reduce, _check_reduce),
    _AlgorithmCase(
        "unary_transform", _make_unary_shared, _make_unary_worker, _run_unary, _check_unary
    ),
    _AlgorithmCase(
        "binary_transform",
        _make_binary_shared,
        _make_binary_worker,
        _run_binary,
        _check_binary,
    ),
    _AlgorithmCase(
        "exclusive_scan",
        _make_exclusive_scan_shared,
        _make_scan_worker,
        _run_scan,
        _check_exclusive_scan,
    ),
    _AlgorithmCase(
        "inclusive_scan",
        _make_inclusive_scan_shared,
        _make_scan_worker,
        _run_scan,
        _check_inclusive_scan,
    ),
    _AlgorithmCase(
        "segmented_reduce",
        _make_segmented_reduce_shared,
        _make_segmented_reduce_worker,
        _run_segmented_reduce,
        _check_segmented_reduce,
    ),
    _AlgorithmCase(
        "histogram",
        _make_histogram_shared,
        _make_histogram_worker,
        _run_histogram,
        _check_histogram,
    ),
    _AlgorithmCase(
        "lower_bound",
        _make_lower_bound_shared,
        _make_binary_search_worker,
        _run_binary_search,
        _check_lower_bound,
    ),
    _AlgorithmCase(
        "upper_bound",
        _make_upper_bound_shared,
        _make_binary_search_worker,
        _run_binary_search,
        _check_upper_bound,
    ),
    _AlgorithmCase("select", _make_select_shared, _make_select_worker, _run_select, _check_select),
    _AlgorithmCase(
        "three_way_partition",
        _make_three_way_shared,
        _make_three_way_worker,
        _run_three_way,
        _check_three_way,
    ),
    _AlgorithmCase(
        "unique_by_key", _make_unique_shared, _make_unique_worker, _run_unique, _check_unique
    ),
    _AlgorithmCase(
        "merge_sort",
        _make_merge_sort_shared,
        _make_merge_sort_worker,
        _run_merge_sort,
        _check_merge_sort,
    ),
    _AlgorithmCase(
        "radix_sort",
        _make_radix_sort_shared,
        _make_radix_sort_worker,
        _run_radix_sort,
        _check_radix_sort,
    ),
    _AlgorithmCase(
        "segmented_sort",
        _make_segmented_sort_shared,
        _make_segmented_sort_worker,
        _run_segmented_sort,
        _check_segmented_sort,
    ),
]


def test_free_threaded_import_keeps_gil_disabled(compute_modules):
    cp, cc = compute_modules

    h_in = np.arange(8, dtype=np.int32)
    d_in = cp.asarray(h_in)
    d_out = cp.empty(1, dtype=np.int32)
    h_init = np.array([0], dtype=np.int32)

    cc.reduce_into(
        d_in=d_in,
        d_out=d_out,
        num_items=h_in.size,
        op=cc.OpKind.PLUS,
        h_init=h_init,
    )

    assert int(d_out.get()[0]) == int(h_in.sum())
    _assert_gil_disabled("after running cuda.compute smoke operation")


@pytest.mark.parametrize("case", SHARED_ALGORITHM_CASES, ids=str)
def test_thread_local_algorithm_objects_share_build_result(compute_modules, case):
    cp, cc = compute_modules

    _run_thread_local_algorithm_case(cp, cc, case)


def _cache_miss_reduce(cp, cc, worker_id, iteration):
    worker = _make_reduce_worker(cp, cc, worker_id, iteration)
    reducer = cc.make_reduce_into(
        d_in=worker["d_in"],
        d_out=worker["d_out"],
        op=cc.OpKind.PLUS,
        h_init=worker["h_init"],
    )
    _run_reduce(cp, cc, reducer, worker)
    _check_reduce(cp, cc, worker)
    return reducer


def _cache_miss_unary_transform(cp, cc, worker_id, iteration):
    worker = _make_unary_worker(cp, cc, worker_id, iteration)
    transformer = cc.make_unary_transform(
        d_in=worker["d_in"], d_out=worker["d_out"], op=cc.OpKind.NEGATE
    )
    _run_unary(cp, cc, transformer, worker)
    _check_unary(cp, cc, worker)
    return transformer


def _cache_miss_binary_transform(cp, cc, worker_id, iteration):
    worker = _make_binary_worker(cp, cc, worker_id, iteration)
    transformer = cc.make_binary_transform(
        d_in1=worker["d_in1"],
        d_in2=worker["d_in2"],
        d_out=worker["d_out"],
        op=cc.OpKind.PLUS,
    )
    _run_binary(cp, cc, transformer, worker)
    _check_binary(cp, cc, worker)
    return transformer


@pytest.mark.parametrize(
    "factory",
    [_cache_miss_reduce, _cache_miss_unary_transform, _cache_miss_binary_transform],
    ids=["reduce", "unary_transform", "binary_transform"],
)
def test_same_key_factory_cache_miss_storm(compute_modules, factory):
    cp, cc = compute_modules

    for iteration in range(STRESS_ITERATIONS):
        cc.clear_all_caches()
        returned_objects = [None] * STRESS_THREADS

        def make_thread(worker_id):
            def thread(barrier):
                barrier.wait()
                returned_objects[worker_id] = factory(cp, cc, worker_id, iteration)

            return thread

        _run_threaded([make_thread(worker_id) for worker_id in range(STRESS_THREADS)])

        assert len({id(obj) for obj in returned_objects}) == len(returned_objects)
        assert len({id(_get_build_result(obj)) for obj in returned_objects}) == 1


def test_shared_raw_op_object_direct_algorithm_stress(compute_modules):
    cp, cc = compute_modules

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
            stream, cuda_stream = _make_stream(cp)
            h_in = np.arange(32, dtype=np.int32) + worker_id * 31 + iteration
            h_init = np.array([worker_id + 5], dtype=np.int32)
            with stream:
                d_in = cp.asarray(h_in)
                d_out = cp.empty(1, dtype=np.int32)

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
                    cp,
                    reducer,
                    d_in=d_in,
                    d_out=d_out,
                    op=shared_op,
                    h_init=h_init,
                    num_items=h_in.size,
                    stream=cuda_stream,
                )
                stream.synchronize()
                expected = int(h_in.sum(dtype=np.int64) + h_init[0])
                assert int(d_out.get()[0]) == expected

            return thread

        _run_threaded([make_thread(worker_id) for worker_id in range(STRESS_THREADS)])

        assert len({id(reducer) for reducer in returned_reducers}) == len(
            returned_reducers
        )
        assert len({id(_get_build_result(reducer)) for reducer in returned_reducers}) == 1


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


def _run_cold_transform_native_cache_case(cp, cc, case: _ColdTransformCase) -> None:
    for iteration in range(STRESS_ITERATIONS):
        cc.clear_all_caches()
        workers = [
            case.make_worker(cp, cc, worker_id=worker_id, iteration=iteration)
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
                    algorithm = case.make_transformer(cp, cc, worker)
                    returned_algorithms[worker_id] = algorithm
                except BaseException:
                    execute_barrier.abort()
                    raise

                execute_barrier.wait(timeout=60)
                case.run(cp, cc, algorithm, worker)
                case.check(cp, cc, worker)

            return thread

        _run_threaded(
            [make_thread(worker_id, worker) for worker_id, worker in enumerate(workers)]
        )

        assert len({id(algorithm) for algorithm in returned_algorithms}) == len(
            returned_algorithms
        )
        assert len(
            {id(_get_build_result(algorithm)) for algorithm in returned_algorithms}
        ) == 1


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
def test_cold_transform_native_cache_initialization_stress(compute_modules, case):
    cp, cc = compute_modules

    _run_cold_transform_native_cache_case(cp, cc, case)


def _iterator_counting(cp, cc):
    return cc.CountingIterator(np.int32(0)), np.dtype(np.int32), 32, sum(range(32))


def _iterator_constant(cp, cc):
    return cc.ConstantIterator(np.int32(5)), np.dtype(np.int32), 32, 32 * 5


def _iterator_cache_modified(cp, cc):
    h_in = np.arange(32, dtype=np.int32)
    d_in = cp.asarray(h_in)
    return cc.CacheModifiedInputIterator(d_in, "stream"), h_in.dtype, h_in.size, int(h_in.sum())


def _iterator_reverse(cp, cc):
    h_in = np.arange(32, dtype=np.int32)
    d_in = cp.asarray(h_in)
    return cc.ReverseIterator(d_in), h_in.dtype, h_in.size, int(h_in.sum())


def _iterator_permutation(cp, cc):
    h_values = np.arange(32, dtype=np.int32)
    h_indices = np.arange(31, -1, -1, dtype=np.int32)
    d_values = cp.asarray(h_values)
    d_indices = cp.asarray(h_indices)
    return (
        cc.PermutationIterator(d_values, d_indices),
        h_values.dtype,
        h_indices.size,
        int(h_values[h_indices].sum()),
    )


def _iterator_shuffle(cp, cc):
    num_items = 32
    return (
        cc.ShuffleIterator(num_items, seed=1234),
        np.dtype(np.int64),
        num_items,
        sum(range(num_items)),
    )


def _iterator_transform(cp, cc):
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
        cc.TransformIterator(cc.CountingIterator(np.int32(0)), op, value_type=types.int32),
        np.dtype(np.int32),
        num_items,
        -sum(range(num_items)),
    )


ITERATOR_FACTORIES = [
    _iterator_counting,
    _iterator_constant,
    _iterator_cache_modified,
    _iterator_reverse,
    _iterator_permutation,
    _iterator_shuffle,
    _iterator_transform,
]


@pytest.mark.parametrize(
    "make_iterator",
    ITERATOR_FACTORIES,
    ids=lambda fn: fn.__name__.removeprefix("_iterator_"),
)
def test_shared_iterator_object_stress(compute_modules, make_iterator):
    cp, cc = compute_modules

    shared_iterator, dtype, num_items, expected_sum = make_iterator(cp, cc)
    cp.cuda.Device().synchronize()

    for iteration in range(STRESS_ITERATIONS):
        cc.clear_all_caches()

        def make_thread(worker_id):
            stream, cuda_stream = _make_stream(cp)
            h_init = np.array([worker_id], dtype=dtype)
            with stream:
                d_out = cp.empty(1, dtype=dtype)

            def thread(barrier):
                barrier.wait()
                reducer = cc.make_reduce_into(
                    d_in=shared_iterator,
                    d_out=d_out,
                    op=cc.OpKind.PLUS,
                    h_init=h_init,
                )
                _call_with_temp(
                    cp,
                    reducer,
                    d_in=shared_iterator,
                    d_out=d_out,
                    op=cc.OpKind.PLUS,
                    h_init=h_init,
                    num_items=num_items,
                    stream=cuda_stream,
                )
                stream.synchronize()
                assert int(d_out.get()[0]) == int(expected_sum + h_init[0])

            return thread

        _run_threaded([make_thread(worker_id) for worker_id in range(STRESS_THREADS)])


def test_runtime_ownership_isolation(compute_modules):
    cp, cc = compute_modules

    def make_thread(worker_id):
        def thread(barrier):
            barrier.wait()
            stream, cuda_stream = _make_stream(cp)
            h_in = np.arange(16, dtype=np.int32) + worker_id * 10
            h_init = np.array([worker_id], dtype=np.int32)

            with stream:
                d_in = cp.asarray(h_in)
                d_reduce_out = cp.empty(1, dtype=np.int32)
                d_scan_out = cp.empty_like(d_in)
                d_transform_out = cp.empty_like(d_in)
                d_hist = cp.zeros(4, dtype=np.int32)
                h_keys = np.array([3, 1, 2, 1], dtype=np.uint32) + worker_id
                d_keys_in = cp.asarray(h_keys)
                d_keys_tmp = cp.empty_like(d_keys_in)

            cc.reduce_into(
                d_in=d_in,
                d_out=d_reduce_out,
                num_items=h_in.size,
                op=cc.OpKind.PLUS,
                h_init=h_init,
                stream=cuda_stream,
            )
            cc.exclusive_scan(
                d_in=d_in,
                d_out=d_scan_out,
                op=cc.OpKind.PLUS,
                init_value=h_init,
                num_items=h_in.size,
                stream=cuda_stream,
            )
            cc.unary_transform(
                d_in=d_in,
                d_out=d_transform_out,
                op=cc.OpKind.NEGATE,
                num_items=h_in.size,
                stream=cuda_stream,
            )
            cc.histogram_even(
                d_samples=d_in,
                d_histogram=d_hist,
                num_output_levels=5,
                lower_level=np.int32(worker_id * 10),
                upper_level=np.int32(worker_id * 10 + 16),
                num_samples=h_in.size,
                stream=cuda_stream,
            )
            keys = cc.DoubleBuffer(d_keys_in, d_keys_tmp)
            cc.radix_sort(
                d_in_keys=keys,
                d_out_keys=None,
                d_in_values=None,
                d_out_values=None,
                num_items=d_keys_in.size,
                order=cc.SortOrder.ASCENDING,
                stream=cuda_stream,
            )

            stream.synchronize()
            assert int(d_reduce_out.get()[0]) == int(h_in.sum() + worker_id)
            expected_scan = np.empty_like(h_in)
            expected_scan[0] = worker_id
            expected_scan[1:] = worker_id + np.cumsum(h_in[:-1])
            np.testing.assert_array_equal(d_scan_out.get(), expected_scan)
            np.testing.assert_array_equal(d_transform_out.get(), -h_in)
            assert int(d_hist.sum().get()) == h_in.size
            np.testing.assert_array_equal(keys.current().get(), np.sort(h_keys))

        return thread

    for _ in range(STRESS_ITERATIONS):
        _run_threaded([make_thread(worker_id) for worker_id in range(STRESS_THREADS)])


def test_cache_clear_while_active_operations_is_not_a_supported_contract():
    pytest.skip(
        "clear_all_caches() while cached operations are active is an unsupported "
        "contract decision; see ST-19 in stress_tests.md."
    )
