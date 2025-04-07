import cupy as cp
import numpy as np
import pytest

import cuda.parallel.experimental.algorithms as algorithms
import cuda.parallel.experimental.iterators as iterators


def reduce_pointer(size, build_only):
    d = cp.ones(size, dtype="i4")
    res = cp.empty(tuple(), dtype=d.dtype)
    h_init = np.zeros(tuple(), dtype=d.dtype)

    def my_add(a, b):
        return a + b

    alg = algorithms.reduce_into(d, res, my_add, h_init)
    temp_bytes = alg(None, d, res, size, h_init)
    scratch = cp.empty(temp_bytes, dtype=cp.uint8)

    if not build_only:
        alg(scratch, d, res, size, h_init)

    cp.cuda.runtime.deviceSynchronize()


def reduce_iterator(size, build_only):
    dt = cp.int32
    d = iterators.CountingIterator(np.int32(0))
    res = cp.empty(tuple(), dtype=dt)
    h_init = np.zeros(tuple(), dtype=dt)

    def my_add(a, b):
        return a + b

    alg = algorithms.reduce_into(d, res, my_add, h_init)
    temp_bytes = alg(None, d, res, size, h_init)
    scratch = cp.empty(temp_bytes, dtype=cp.uint8)

    if not build_only:
        alg(scratch, d, res, size, h_init)

    cp.cuda.runtime.deviceSynchronize()


def merge_sort_pointer(size, build_only):
    keys = cp.arange(size, dtype="i4")
    vals = cp.arange(size, dtype="i8")
    res_keys = cp.empty_like(keys)
    res_vals = cp.empty_like(vals)

    def my_cmp(a: np.int32, b: np.int32) -> np.int32:
        return np.int32(a < b)

    alg = algorithms.merge_sort(keys, vals, res_keys, res_vals, my_cmp)
    temp_bytes = alg(None, keys, vals, res_keys, res_vals, size)
    scratch = cp.empty(temp_bytes, dtype=cp.uint8)

    if not build_only:
        alg(scratch, keys, vals, res_keys, res_vals, size)

    cp.cuda.runtime.deviceSynchronize()


def merge_sort_iterator(size, build_only):
    keys_dt = cp.int32
    vals_dt = cp.int64
    keys = iterators.CountingIterator(np.int32(0))
    vals = iterators.CountingIterator(np.int64(0))
    res_keys = cp.empty(size, dtype=keys_dt)
    res_vals = cp.empty(size, dtype=vals_dt)

    def my_cmp(a: np.int32, b: np.int32) -> np.int32:
        return np.int32(a < b)

    alg = algorithms.merge_sort(keys, vals, res_keys, res_vals, my_cmp)
    temp_bytes = alg(None, keys, vals, res_keys, res_vals, size)
    scratch = cp.empty(temp_bytes, dtype=cp.uint8)

    if not build_only:
        alg(scratch, keys, vals, res_keys, res_vals, size)

    cp.cuda.runtime.deviceSynchronize()


@pytest.fixture(params=[True, False])
def build_only(request):
    return request.param


@pytest.fixture(params=[10_000, 100_000, 1_000_000])
def size(request):
    return request.param


def bench_compile_reduce_pointer(benchmark):
    def setup():
        # This function is called once before the benchmark runs
        # to set up the environment.
        algorithms.reduce_into.cache_clear()

    benchmark.pedantic(
        reduce_pointer,
        kwargs={"size": 10, "build_only": True},
        rounds=3,
        iterations=1,
        setup=setup,
    )


def bench_compile_reduce_iterator(benchmark):
    def setup():
        # This function is called once before the benchmark runs
        # to set up the environment.
        algorithms.reduce_into.cache_clear()

    benchmark.pedantic(
        reduce_iterator,
        kwargs={"size": 10, "build_only": True},
        rounds=3,
        iterations=1,
        setup=setup,
    )


def bench_compile_merge_sort_pointer(benchmark):
    def setup():
        # This function is called once before the benchmark runs
        # to set up the environment.
        algorithms.merge_sort.cache_clear()

    benchmark.pedantic(
        merge_sort_pointer,
        kwargs={"size": 10, "build_only": True},
        rounds=3,
        iterations=1,
        setup=setup,
    )


def bench_compile_merge_sort_iterator(benchmark):
    def setup():
        # This function is called once before the benchmark runs
        # to set up the environment.
        algorithms.merge_sort.cache_clear()

    benchmark.pedantic(
        merge_sort_iterator,
        kwargs={"size": 10, "build_only": True},
        rounds=3,
        iterations=1,
        setup=setup,
    )


def bench_reduce_pointer(benchmark, size):
    benchmark(reduce_pointer, size, build_only=False)


def bench_reduce_iterator(benchmark, size):
    benchmark(reduce_iterator, size, build_only=False)


def bench_merge_sort_pointer(benchmark, size):
    benchmark(merge_sort_pointer, size, build_only=False)


def bench_merge_sort_iterator(benchmark, size):
    benchmark(merge_sort_iterator, size, build_only=False)
