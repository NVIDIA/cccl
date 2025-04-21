import cupy as cp
import numpy as np

import cuda.parallel.experimental.algorithms as algorithms
import cuda.parallel.experimental.iterators as iterators
from cuda.parallel.experimental.struct import gpu_struct


def reduce_pointer(input_array, build_only):
    size = len(input_array)
    res = cp.empty(tuple(), dtype=input_array.dtype)
    h_init = np.zeros(tuple(), dtype=input_array.dtype)

    def my_add(a, b):
        return a + b

    alg = algorithms.reduce_into(input_array, res, my_add, h_init)

    if not build_only:
        temp_bytes = alg(None, input_array, res, size, h_init)
        scratch = cp.empty(temp_bytes, dtype=cp.uint8)
        alg(scratch, input_array, res, size, h_init)

    cp.cuda.runtime.deviceSynchronize()


def reduce_struct(input_array, build_only):
    size = len(input_array)
    res = cp.empty(tuple(), dtype=input_array.dtype)
    h_init = MyStruct(0, 0)

    def my_add(a, b):
        return MyStruct(a.x + b.x, a.y + b.y)

    alg = algorithms.reduce_into(input_array, res, my_add, h_init)

    if not build_only:
        temp_bytes = alg(None, input_array, res, size, h_init)
        scratch = cp.empty(temp_bytes, dtype=cp.uint8)
        alg(scratch, input_array, res, size, h_init)

    cp.cuda.runtime.deviceSynchronize()


def reduce_iterator(inp, size, build_only):
    dt = cp.int32
    res = cp.empty(tuple(), dtype=dt)
    h_init = np.zeros(tuple(), dtype=dt)

    def my_add(a, b):
        return a + b

    alg = algorithms.reduce_into(inp, res, my_add, h_init)

    if not build_only:
        temp_bytes = alg(None, inp, res, size, h_init)
        scratch = cp.empty(temp_bytes, dtype=cp.uint8)
        alg(scratch, inp, res, size, h_init)

    cp.cuda.runtime.deviceSynchronize()


@gpu_struct
class MyStruct:
    x: np.int32
    y: np.int32


def bench_compile_reduce_pointer(compile_benchmark):
    input_array = cp.random.randint(0, 10, 10)

    def run():
        reduce_pointer(input_array, build_only=True)

    compile_benchmark(algorithms.reduce_into, run)


def bench_compile_reduce_iterator(compile_benchmark):
    inp = iterators.CountingIterator(np.int32(0))

    def run():
        reduce_iterator(inp, 10, build_only=True)

    compile_benchmark(algorithms.reduce_into, run)


def bench_reduce_pointer(benchmark, size):
    input_array = cp.random.randint(0, 10, size)

    def run():
        reduce_pointer(input_array, build_only=False)

    benchmark(run)


def bench_reduce_iterator(benchmark, size):
    inp = iterators.CountingIterator(np.int32(0))

    def run():
        reduce_iterator(inp, size, build_only=False)

    benchmark(run)


def bench_reduce_struct(benchmark, size):
    input_array = cp.random.randint(0, 10, (size, 2), dtype="int32").view(MyStruct)

    def run():
        reduce_struct(input_array, build_only=False)

    benchmark(run)
