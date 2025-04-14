import cupy as cp
import numpy as np

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


def bench_compile_reduce_pointer(compile_benchmark):
    compile_benchmark(algorithms.reduce_into, reduce_pointer)


def bench_compile_merge_sort_iterator(compile_benchmark):
    compile_benchmark(algorithms.reduce_into, reduce_iterator)


def bench_reduce_pointer(benchmark, size):
    benchmark(reduce_pointer, size, build_only=False)


def bench_reduce_iterator(benchmark, size):
    benchmark(reduce_iterator, size, build_only=False)
