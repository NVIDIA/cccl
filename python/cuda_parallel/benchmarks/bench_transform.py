import cupy as cp
import numpy as np

import cuda.parallel.experimental.algorithms as algorithms
import cuda.parallel.experimental.iterators as iterators


def unary_transform_pointer(size, build_only):
    d_in = cp.arange(size, dtype="i4")
    d_out = cp.empty_like(d_in)

    def op(a):
        return a + 1

    transform = algorithms.unary_transform(d_in, d_out, op)

    if not build_only:
        transform(d_in, d_out, size)

    cp.cuda.runtime.deviceSynchronize()


def unary_transform_iterator(size, build_only):
    d_in = iterators.CountingIterator(np.int32(0))
    d_out = cp.empty(size, dtype=np.int32)

    def op(a):
        return a + 1

    transform = algorithms.unary_transform(d_in, d_out, op)

    if not build_only:
        transform(d_in, d_out, size)

    cp.cuda.runtime.deviceSynchronize()


def binary_transform_pointer(size, build_only):
    d_in1 = cp.arange(size, dtype="i4")
    d_in2 = cp.arange(size, dtype="i4")
    d_out = cp.empty_like(d_in1)

    def op(a, b):
        return a + b

    transform = algorithms.binary_transform(d_in1, d_in2, d_out, op)

    if not build_only:
        transform(d_in1, d_in2, d_out, size)

    cp.cuda.runtime.deviceSynchronize()


def binary_transform_iterator(size, build_only):
    d_in1 = iterators.CountingIterator(np.int32(0))
    d_in2 = iterators.CountingIterator(np.int32(1))
    d_out = cp.empty(size, dtype=np.int32)

    def op(a, b):
        return a + b

    transform = algorithms.binary_transform(d_in1, d_in2, d_out, op)

    if not build_only:
        transform(d_in1, d_in2, d_out, size)

    cp.cuda.runtime.deviceSynchronize()


def bench_compile_unary_transform_pointer(compile_benchmark):
    compile_benchmark(algorithms.unary_transform, unary_transform_pointer)


def bench_compile_unary_transform_iterator(compile_benchmark):
    compile_benchmark(algorithms.unary_transform, unary_transform_iterator)


def bench_compile_binary_transform_pointer(compile_benchmark):
    compile_benchmark(algorithms.binary_transform, binary_transform_pointer)


def bench_compile_binary_transform_iterator(compile_benchmark):
    compile_benchmark(algorithms.binary_transform, binary_transform_iterator)


def bench_unary_transform_pointer(benchmark, size):
    benchmark(unary_transform_pointer, size, build_only=False)


def bench_unary_transform_iterator(benchmark, size):
    benchmark(unary_transform_iterator, size, build_only=False)


def bench_binary_transform_pointer(benchmark, size):
    benchmark(binary_transform_pointer, size, build_only=False)


def bench_binary_transform_iterator(benchmark, size):
    benchmark(binary_transform_iterator, size, build_only=False)
