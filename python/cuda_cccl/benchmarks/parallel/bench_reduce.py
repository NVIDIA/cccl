import cupy as cp
import numpy as np

import cuda.cccl.parallel.experimental as parallel


def reduce_pointer(input_array, build_only):
    size = len(input_array)
    res = cp.empty(tuple(), dtype=input_array.dtype)
    h_init = np.zeros(tuple(), dtype=input_array.dtype)

    def my_add(a, b):
        return a + b

    alg = parallel.make_reduce_into(input_array, res, my_add, h_init)
    if not build_only:
        temp_storage_bytes = alg(None, input_array, res, size, h_init)
        temp_storage = cp.empty(temp_storage_bytes, dtype=np.uint8)
        alg(temp_storage, input_array, res, size, h_init)

    cp.cuda.runtime.deviceSynchronize()


def reduce_struct(input_array, build_only):
    size = len(input_array)
    res = cp.empty(tuple(), dtype=input_array.dtype)
    h_init = MyStruct(0, 0)

    def my_add(a, b):
        return MyStruct(a.x + b.x, a.y + b.y)

    alg = parallel.make_reduce_into(input_array, res, my_add, h_init)
    if not build_only:
        temp_storage_bytes = alg(None, input_array, res, size, h_init)
        temp_storage = cp.empty(temp_storage_bytes, dtype=np.uint8)
        alg(temp_storage, input_array, res, size, h_init)

    cp.cuda.runtime.deviceSynchronize()


def reduce_iterator(inp, size, build_only):
    dt = cp.int32
    res = cp.empty(tuple(), dtype=dt)
    h_init = np.zeros(tuple(), dtype=dt)

    def my_add(a, b):
        return a + b

    alg = parallel.make_reduce_into(inp, res, my_add, h_init)
    if not build_only:
        temp_storage_bytes = alg(None, inp, res, size, h_init)
        temp_storage = cp.empty(temp_storage_bytes, dtype=np.uint8)
        alg(temp_storage, inp, res, size, h_init)

    cp.cuda.runtime.deviceSynchronize()


@parallel.gpu_struct
class MyStruct:
    x: np.int32
    y: np.int32


def bench_compile_reduce_pointer(compile_benchmark):
    input_array = cp.random.randint(0, 10, 10)

    def run():
        reduce_pointer(input_array, build_only=True)

    compile_benchmark(parallel.make_reduce_into, run)


def bench_compile_reduce_iterator(compile_benchmark):
    inp = parallel.CountingIterator(np.int32(0))

    def run():
        reduce_iterator(inp, 10, build_only=True)

    compile_benchmark(parallel.make_reduce_into, run)


def bench_reduce_pointer(benchmark, size):
    input_array = cp.random.randint(0, 10, size)

    def run():
        reduce_pointer(input_array, build_only=False)

    benchmark(run)


def bench_reduce_iterator(benchmark, size):
    inp = parallel.CountingIterator(np.int32(0))

    def run():
        reduce_iterator(inp, size, build_only=False)

    benchmark(run)


def bench_reduce_struct(benchmark, size):
    input_array = cp.random.randint(0, 10, (size, 2), dtype="int32").view(MyStruct)

    def run():
        reduce_struct(input_array, build_only=False)

    benchmark(run)


def bench_reduce_pointer_single_phase(benchmark, size):
    input_array = cp.random.randint(0, 10, size)

    # warm up run
    reduce_pointer_single_phase(input_array, build_only=False)

    # benchmark run
    def run():
        reduce_pointer_single_phase(input_array, build_only=False)

    benchmark(run)


def bench_reduce_iterator_single_phase(benchmark, size):
    inp = parallel.CountingIterator(np.int32(0))

    # warm up run
    reduce_iterator_single_phase(inp, size, build_only=False)

    # benchmark run
    def run():
        reduce_iterator_single_phase(inp, size, build_only=False)

    benchmark(run)


def bench_reduce_struct_single_phase(benchmark, size):
    input_array = cp.random.randint(0, 10, (size, 2), dtype="int32").view(MyStruct)

    # warm up run
    reduce_struct_single_phase(input_array, build_only=False)

    # benchmark run
    def run():
        reduce_struct_single_phase(input_array, build_only=False)

    benchmark(run)


def reduce_pointer_single_phase(input_array, build_only):
    """Single-phase API that automatically manages temporary storage."""
    size = len(input_array)
    res = cp.empty(tuple(), dtype=input_array.dtype)
    h_init = np.zeros(tuple(), dtype=input_array.dtype)

    def my_add(a, b):
        return a + b

    parallel.reduce_into(input_array, res, my_add, size, h_init)

    cp.cuda.runtime.deviceSynchronize()


def reduce_struct_single_phase(input_array, build_only):
    """Single-phase API that automatically manages temporary storage for structs."""
    size = len(input_array)
    res = cp.empty(tuple(), dtype=input_array.dtype)
    h_init = MyStruct(0, 0)

    def my_add(a, b):
        return MyStruct(a.x + b.x, a.y + b.y)

    parallel.reduce_into(input_array, res, my_add, size, h_init)

    cp.cuda.runtime.deviceSynchronize()


def reduce_iterator_single_phase(inp, size, build_only):
    """Single-phase API that automatically manages temporary storage for iterators."""
    dt = cp.int32
    res = cp.empty(tuple(), dtype=dt)
    h_init = np.zeros(tuple(), dtype=dt)

    def my_add(a, b):
        return a + b

    parallel.reduce_into(inp, res, my_add, size, h_init)

    cp.cuda.runtime.deviceSynchronize()
