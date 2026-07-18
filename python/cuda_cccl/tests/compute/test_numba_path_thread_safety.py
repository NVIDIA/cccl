# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Thread-safety of the shared build/cache machinery for the numba-dependent
op paths, on regular (GIL) interpreters.

cuda.compute releases the GIL around native builds and
launches, so concurrent calls overlap in the C layer even under the GIL.
So the shared caching/native machinery must be thread-safe on every interpreter
build, and these tests validate that for the numba op paths.

They cover the paths the free-threading stress suite
(test_free_threading_stress.py) cannot: that suite runs on the minimal extra,
which omits numba, so Python-callable ops, stateful closure ops, gpu_struct
types, and return-type inference are untested there. This file exercises them
from multiple threads on a regular GIL interpreter. The C-layer build/launch
machinery itself is op-source-agnostic and already covered by the stress suite;
the unique surface here is the numba frontend (CachableFunction hashing,
inference, closure/struct handling).

Only the native build/launch phase genuinely overlaps (the GIL is released
around it); numba's own compilation serializes on its compiler lock and the
Python-side key hashing serializes under the GIL. So the goal is not to prove
numba runs in parallel, but that cuda.compute's caching, key hashing, and
descriptor machinery stay correct when these paths are entered concurrently.
"""

import concurrent.futures
import threading

import numpy as np
import pytest
from _utils.device_array import DeviceArray

import cuda.compute
from cuda.compute import (
    CountingIterator,
    OpKind,
    TransformIterator,
    gpu_struct,
    make_reduce_into,
    make_unary_transform,
)
from cuda.core import Device

pytestmark = pytest.mark.no_verify_sass(
    reason="Concurrency tests intentionally run concurrent workers."
)

THREADS = 4
ITERATIONS = 3


def _run_threaded(workers):
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


def _reduce_with_temp(reducer, **kwargs):
    temp_storage_bytes = reducer(temp_storage=None, **kwargs)
    temp_storage = DeviceArray.empty(temp_storage_bytes, np.uint8)
    return reducer(temp_storage=temp_storage, **kwargs)


def _single_build_result(algorithm):
    assert len(algorithm.build_results) == 1
    return next(iter(algorithm.build_results.values()))


def _make_clamped_max_op(k):
    def clamped_max(a, b):
        m = a if a > b else b
        return m if m > k else k

    return clamped_max


def test_concurrent_distinct_python_ops_build_storm():
    """Distinct Python callables force a separate numba+native build per worker.

    Distinct closure constants give each worker its own cache key, so
    _cache_single_flight elects a separate builder per thread. The native
    builds overlap (the GIL is released around them) while the CachableFunction
    hashing and numba compilation serialize (under the GIL and numba's own
    compiler lock); the target is that this concurrent entry keeps each
    worker's op and build uncontaminated. The op computes max(a, b, k) with a
    per-worker k that dominates every input, so the expected result is exactly
    k however CUB shapes the reduction tree — and a wrong k directly exposes
    any cross-thread op/build contamination.
    """
    num_items = 64
    for iteration in range(ITERATIONS):
        cuda.compute.clear_all_caches()
        returned_reducers = [None] * THREADS

        def make_thread(worker_id):
            k = 10_000 + worker_id * 7 + iteration
            op = _make_clamped_max_op(k)
            h_in = np.arange(num_items, dtype=np.int64) + worker_id
            h_init = np.array([0], dtype=np.int64)
            d_in = DeviceArray.from_numpy(h_in)
            d_out = DeviceArray.empty(1, np.int64)

            def thread(barrier):
                # cuda.core device state is per-thread; initialize explicitly
                # rather than relying on DeviceArray-construction side effects.
                Device().set_current()
                barrier.wait()
                reducer = make_reduce_into(d_in=d_in, d_out=d_out, op=op, h_init=h_init)
                returned_reducers[worker_id] = reducer
                _reduce_with_temp(
                    reducer,
                    d_in=d_in,
                    d_out=d_out,
                    op=op,
                    h_init=h_init,
                    num_items=num_items,
                )
                Device().sync()
                assert int(d_out.copy_to_host()[0]) == k

            return thread

        _run_threaded([make_thread(worker_id) for worker_id in range(THREADS)])

        build_ids = {id(_single_build_result(r)) for r in returned_reducers}
        assert len(build_ids) == THREADS


def test_concurrent_shared_python_op_coalesces():
    """One shared Python callable from all threads coalesces to a single build.

    All workers hash the same function concurrently (CachableFunction walks
    bytecode, constants, and closures on every factory call) and race the
    same cache key; exactly one build must result, with one wrapper per thread.
    """

    def add(a, b):
        return a + b

    num_items = 64
    for iteration in range(ITERATIONS):
        cuda.compute.clear_all_caches()
        returned_reducers = [None] * THREADS

        def make_thread(worker_id):
            h_in = np.arange(num_items, dtype=np.int32) + worker_id + iteration
            h_init = np.array([worker_id], dtype=np.int32)
            d_in = DeviceArray.from_numpy(h_in)
            d_out = DeviceArray.empty(1, np.int32)

            def thread(barrier):
                # cuda.core device state is per-thread; initialize explicitly
                # rather than relying on DeviceArray-construction side effects.
                Device().set_current()
                barrier.wait()
                reducer = make_reduce_into(
                    d_in=d_in, d_out=d_out, op=add, h_init=h_init
                )
                returned_reducers[worker_id] = reducer
                _reduce_with_temp(
                    reducer,
                    d_in=d_in,
                    d_out=d_out,
                    op=add,
                    h_init=h_init,
                    num_items=num_items,
                )
                Device().sync()
                assert int(d_out.copy_to_host()[0]) == int(h_in.sum()) + worker_id

            return thread

        _run_threaded([make_thread(worker_id) for worker_id in range(THREADS)])

        # Per-thread wrappers, one shared native build.
        assert len({id(r) for r in returned_reducers}) == THREADS
        assert len({id(_single_build_result(r)) for r in returned_reducers}) == 1


def _make_scale_op(k):
    def scale(x):
        return x * k

    return scale


def test_concurrent_transform_iterator_return_type_inference():
    """Unannotated TransformIterator ops infer return types concurrently.

    TransformIterator without value_type routes through _infer_return_type
    (numba typing). Ops are distinct per worker and per iteration so the
    inference cache stays cold and each thread runs the typing path itself.
    """
    num_items = 32
    for iteration in range(ITERATIONS):
        cuda.compute.clear_all_caches()

        def make_thread(worker_id):
            k = np.int32(worker_id + 1 + iteration * THREADS)
            op = _make_scale_op(k)
            h_init = np.array([0], dtype=np.int32)
            d_out = DeviceArray.empty(1, np.int32)

            def thread(barrier):
                # cuda.core device state is per-thread; initialize explicitly
                # rather than relying on DeviceArray-construction side effects.
                Device().set_current()
                barrier.wait()
                # Inference widens int32 * int32-closure to an int64 value type;
                # the int64-iterator/int32-accumulator mix is a supported,
                # CI-exercised configuration and all expected values fit int32.
                d_in = TransformIterator(CountingIterator(np.int32(0)), op)
                reducer = make_reduce_into(
                    d_in=d_in, d_out=d_out, op=OpKind.PLUS, h_init=h_init
                )
                _reduce_with_temp(
                    reducer,
                    d_in=d_in,
                    d_out=d_out,
                    op=OpKind.PLUS,
                    h_init=h_init,
                    num_items=num_items,
                )
                Device().sync()
                expected = sum(i * int(k) for i in range(num_items))
                assert int(d_out.copy_to_host()[0]) == expected

            return thread

        _run_threaded([make_thread(worker_id) for worker_id in range(THREADS)])


def test_concurrent_stateful_closure_ops_isolate_state():
    """Same op code with different captured device arrays across threads.

    The captured arrays do not change the cache key (stateful op machinery
    updates pointers per call), so all threads share one native build while
    each thread's wrapper must carry its own captured-state pointer. If state
    isolation broke, one thread's output would use another thread's offset.
    """

    def make_adder(arr):
        def add_offset(x):
            return x + arr[0]

        return add_offset

    num_items = 64
    for iteration in range(ITERATIONS):
        cuda.compute.clear_all_caches()
        returned_transformers = [None] * THREADS

        def make_thread(worker_id):
            offset = worker_id * 10 + iteration
            d_offset = DeviceArray.from_numpy(np.array([offset], dtype=np.int32))
            op = make_adder(d_offset)
            h_in = np.arange(num_items, dtype=np.int32)
            d_in = DeviceArray.from_numpy(h_in)
            d_out = DeviceArray.empty(h_in.shape, h_in.dtype)

            def thread(barrier):
                # cuda.core device state is per-thread; initialize explicitly
                # rather than relying on DeviceArray-construction side effects.
                Device().set_current()
                barrier.wait()
                transformer = make_unary_transform(d_in=d_in, d_out=d_out, op=op)
                returned_transformers[worker_id] = transformer
                transformer(d_in=d_in, d_out=d_out, op=op, num_items=num_items)
                Device().sync()
                np.testing.assert_array_equal(d_out.copy_to_host(), h_in + offset)

            return thread

        _run_threaded([make_thread(worker_id) for worker_id in range(THREADS)])

        assert len({id(t) for t in returned_transformers}) == THREADS
        assert len({id(_single_build_result(t)) for t in returned_transformers}) == 1


def test_concurrent_gpu_struct_reduce():
    """gpu_struct types and struct-typed Python ops used from multiple threads."""

    @gpu_struct
    class MinMax:
        min_val: np.int32
        max_val: np.int32

    def minmax_op(a, b):
        c_min = min(a.min_val, b.min_val)
        c_max = max(a.max_val, b.max_val)
        return MinMax(c_min, c_max)

    num_items = 128
    info = np.iinfo(np.int32)

    # Warm the numba struct registration once on the main thread (mirroring the
    # stress suite's warm pass). First-time registration goes through
    # non-coalescing lru_caches in _jit.py that mutate numba's global
    # registries, so racing it cold from four threads is a separate library
    # hardening question, not this test's target.
    warm_in = DeviceArray.from_numpy(
        np.zeros((1, 2), dtype=np.int32).view(MinMax.dtype)
    )
    warm_out = DeviceArray.empty(1, MinMax.dtype)
    cuda.compute.reduce_into(
        d_in=warm_in,
        d_out=warm_out,
        op=minmax_op,
        h_init=MinMax(info.max, info.min),
        num_items=1,
    )
    Device().sync()

    for iteration in range(ITERATIONS):
        cuda.compute.clear_all_caches()

        def make_thread(worker_id):
            base = worker_id * 1000 + iteration
            h_pairs = np.stack(
                [
                    np.arange(num_items, dtype=np.int32) + base,
                    np.arange(num_items, dtype=np.int32) * 2 + base,
                ],
                axis=1,
            )
            d_in = DeviceArray.from_numpy(h_pairs.view(MinMax.dtype))
            d_out = DeviceArray.empty(1, MinMax.dtype)
            h_init = MinMax(info.max, info.min)

            def thread(barrier):
                # cuda.core device state is per-thread; initialize explicitly
                # rather than relying on DeviceArray-construction side effects.
                Device().set_current()
                barrier.wait()
                cuda.compute.reduce_into(
                    d_in=d_in,
                    d_out=d_out,
                    op=minmax_op,
                    h_init=h_init,
                    num_items=num_items,
                )
                Device().sync()
                result = d_out.copy_to_host()
                assert int(result["min_val"][0]) == int(h_pairs[:, 0].min())
                assert int(result["max_val"][0]) == int(h_pairs[:, 1].max())

            return thread

        _run_threaded([make_thread(worker_id) for worker_id in range(THREADS)])
