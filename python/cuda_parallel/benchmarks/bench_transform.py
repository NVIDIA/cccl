import cupy as cp
import numpy as np

import cuda.parallel.experimental.algorithms as algorithms
import cuda.parallel.experimental.iterators as iterators
from cuda.parallel.experimental.struct import gpu_struct


def unary_transform_pointer(inp, out, build_only):
    size = len(inp)

    def op(a):
        return a + 1

    transform = algorithms.unary_transform(inp, out, op)

    if not build_only:
        transform(inp, out, size)

    cp.cuda.runtime.deviceSynchronize()


def unary_transform_iterator(size, out, build_only):
    d_in = iterators.CountingIterator(np.int32(0))

    def op(a):
        return a + 1

    transform = algorithms.unary_transform(d_in, out, op)

    if not build_only:
        transform(d_in, out, size)

    cp.cuda.runtime.deviceSynchronize()


@gpu_struct
class MyStruct:
    x: np.int32
    y: np.int32


def unary_transform_struct(inp, out, build_only):
    size = len(inp)

    def op(a):
        return MyStruct(a.x + 1, a.y + 1)

    transform = algorithms.unary_transform(inp, out, op)

    if not build_only:
        transform(inp, out, size)

    cp.cuda.runtime.deviceSynchronize()


def binary_transform_pointer(inp1, inp2, out, build_only):
    size = len(inp1)

    def op(a, b):
        return a + b

    transform = algorithms.binary_transform(inp1, inp2, out, op)

    if not build_only:
        transform(inp1, inp2, out, size)

    cp.cuda.runtime.deviceSynchronize()


def binary_transform_iterator(size, out, build_only):
    d_in1 = iterators.CountingIterator(np.int32(0))
    d_in2 = iterators.CountingIterator(np.int32(1))

    def op(a, b):
        return a + b

    transform = algorithms.binary_transform(d_in1, d_in2, out, op)

    if not build_only:
        transform(d_in1, d_in2, out, size)

    cp.cuda.runtime.deviceSynchronize()


def binary_transform_struct(inp1, inp2, out, build_only):
    size = len(inp1)

    def op(a, b):
        return MyStruct(a.x + b.x, a.y + b.y)

    transform = algorithms.binary_transform(inp1, inp2, out, op)

    if not build_only:
        transform(inp1, inp2, out, size)

    cp.cuda.runtime.deviceSynchronize()


def bench_compile_unary_transform_pointer(compile_benchmark):
    size = 100
    inp = cp.random.randint(0, 10, size)
    out = cp.empty_like(inp)

    def run():
        unary_transform_pointer(inp, out, build_only=True)

    compile_benchmark(algorithms.unary_transform, run)


def bench_compile_unary_transform_iterator(compile_benchmark):
    size = 100
    out = cp.empty(size, dtype="int32")

    def run():
        unary_transform_iterator(size, out, build_only=True)

    compile_benchmark(algorithms.unary_transform, run)


def bench_compile_binary_transform_pointer(compile_benchmark):
    size = 100
    inp1 = cp.random.randint(0, 10, size)
    inp2 = cp.random.randint(0, 10, size)
    out = cp.empty_like(inp1)

    def run():
        binary_transform_pointer(inp1, inp2, out, build_only=True)

    compile_benchmark(algorithms.binary_transform, run)


def bench_compile_binary_transform_iterator(compile_benchmark):
    size = 100
    out = cp.empty(size, dtype="int32")

    def run():
        binary_transform_iterator(size, out, build_only=True)

    compile_benchmark(algorithms.binary_transform, run)


def bench_unary_transform_pointer(benchmark, size):
    inp = cp.random.randint(0, 10, size)
    out = cp.empty_like(inp)

    def run():
        unary_transform_pointer(inp, out, build_only=False)

    benchmark(run)


def bench_unary_transform_iterator(benchmark, size):
    out = cp.empty(size, dtype="int32")

    def run():
        unary_transform_iterator(size, out, build_only=False)

    benchmark(run)


def bench_unary_transform_struct(benchmark, size):
    inp = cp.random.randint(0, 10, (size, 2)).view(MyStruct.dtype)
    out = cp.empty_like(inp)

    def run():
        unary_transform_struct(inp, out, build_only=False)

    benchmark(run)


def bench_binary_transform_pointer(benchmark, size):
    inp1 = cp.random.randint(0, 10, size)
    inp2 = cp.random.randint(0, 10, size)
    out = cp.empty_like(inp1)

    def run():
        binary_transform_pointer(inp1, inp2, out, build_only=False)

    benchmark(run)


def bench_binary_transform_iterator(benchmark, size):
    out = cp.empty(size, dtype="int32")

    def run():
        binary_transform_iterator(size, out, build_only=False)

    benchmark(run)


def bench_binary_transform_struct(benchmark, size):
    inp1 = cp.random.randint(0, 10, (size, 2)).view(MyStruct.dtype)
    inp2 = cp.random.randint(0, 10, (size, 2)).view(MyStruct.dtype)
    out = cp.empty_like(inp1)

    def run():
        binary_transform_struct(inp1, inp2, out, build_only=False)

    benchmark(run)
