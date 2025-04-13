import cupy as cp
import numpy as np

import cuda.parallel.experimental.algorithms as algorithms
import cuda.parallel.experimental.iterators as iterators


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


def bench_compile_merge_sort_pointer(compile_benchmark):
    compile_benchmark(algorithms.merge_sort, merge_sort_pointer)


def bench_compile_merge_sort_iterator(compile_benchmark):
    compile_benchmark(algorithms.merge_sort, merge_sort_iterator)


def bench_merge_sort_pointer(benchmark, size):
    benchmark(merge_sort_pointer, size, build_only=False)


def bench_merge_sort_iterator(benchmark, size):
    benchmark(merge_sort_iterator, size, build_only=False)
