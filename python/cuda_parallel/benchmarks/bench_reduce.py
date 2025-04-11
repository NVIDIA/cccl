import cupy as cp
import numpy as np

import cuda.parallel.experimental.algorithms as algorithms
import cuda.parallel.experimental.iterators as iterators


def reduce_pointer(input_array, build_only):
    size = len(input_array)
    res = cp.empty(tuple(), dtype=input_array.dtype)
    h_init = np.zeros(tuple(), dtype=input_array.dtype)

    def my_add(a, b):
        return a + b

    alg = algorithms.reduce_into(input_array, res, my_add, h_init)
    temp_bytes = alg(None, input_array, res, size, h_init)
    scratch = cp.empty(temp_bytes, dtype=cp.uint8)

    if not build_only:
        alg(scratch, input_array, res, size, h_init)

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
    input_array = cp.random.randint(0, 10, 10)

    def run():
        reduce_pointer(input_array, build_only=True)

    compile_benchmark(algorithms.reduce_into, run)


def bench_compile_reduce_iterator(compile_benchmark):
    def run():
        reduce_iterator(10, build_only=True)

    compile_benchmark(algorithms.reduce_into, run)


def bench_reduce_pointer(benchmark, size):
    input_array = cp.random.randint(0, 10, size)

    def run():
        reduce_pointer(input_array, build_only=False)

    benchmark(run)


def bench_reduce_iterator(benchmark, size):
    def run():
        reduce_iterator(size, build_only=False)

    benchmark(run)
