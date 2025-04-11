import cupy as cp
import numpy as np

import cuda.parallel.experimental.algorithms as algorithms
import cuda.parallel.experimental.iterators as iterators


def merge_sort_pointer(keys, vals, output_keys, output_vals, build_only):
    size = len(keys)

    def my_cmp(a: np.int32, b: np.int32) -> np.int32:
        return np.int32(a < b)

    alg = algorithms.merge_sort(keys, vals, output_keys, output_vals, my_cmp)
    temp_bytes = alg(None, keys, vals, output_keys, output_vals, size)
    scratch = cp.empty(temp_bytes, dtype=cp.uint8)

    if not build_only:
        alg(scratch, keys, vals, output_keys, output_vals, size)

    cp.cuda.runtime.deviceSynchronize()


def merge_sort_iterator(size, output_keys, output_vals, build_only):
    keys_dt = cp.int32
    vals_dt = cp.int64
    keys = iterators.CountingIterator(np.int32(0))
    vals = iterators.CountingIterator(np.int64(0))
    output_keys = cp.empty(size, dtype=keys_dt)
    output_vals = cp.empty(size, dtype=vals_dt)

    def my_cmp(a: np.int32, b: np.int32) -> np.int32:
        return np.int32(a < b)

    alg = algorithms.merge_sort(keys, vals, output_keys, output_vals, my_cmp)
    temp_bytes = alg(None, keys, vals, output_keys, output_vals, size)
    scratch = cp.empty(temp_bytes, dtype=cp.uint8)

    if not build_only:
        alg(scratch, keys, vals, output_keys, output_vals, size)

    cp.cuda.runtime.deviceSynchronize()


def bench_compile_merge_sort_pointer(compile_benchmark):
    size = 100
    keys = cp.random.randint(0, 10, size)
    vals = cp.random.randint(0, 10, size)
    output_keys = cp.empty_like(keys)
    output_vals = cp.empty_like(vals)

    def run():
        merge_sort_pointer(keys, vals, output_keys, output_vals, build_only=True)

    compile_benchmark(algorithms.merge_sort, run)


def bench_compile_merge_sort_iterator(compile_benchmark):
    size = 100
    output_keys = cp.zeros(size, dtype="int32")
    output_vals = cp.zeros(size, dtype="int64")

    def run():
        merge_sort_iterator(size, output_keys, output_vals, build_only=True)

    compile_benchmark(algorithms.merge_sort, run)


def bench_merge_sort_pointer(benchmark, size):
    keys = cp.random.randint(0, 10, size)
    vals = cp.random.randint(0, 10, size)
    output_keys = cp.empty_like(keys)
    output_vals = cp.empty_like(vals)

    def run():
        merge_sort_pointer(keys, vals, output_keys, output_vals, build_only=False)

    benchmark(run)


def bench_merge_sort_iterator(benchmark, size):
    output_keys = cp.zeros(size, dtype="int32")
    output_vals = cp.zeros(size, dtype="int64")

    def run():
        merge_sort_iterator(size, output_keys, output_vals, build_only=False)

    benchmark(run)
