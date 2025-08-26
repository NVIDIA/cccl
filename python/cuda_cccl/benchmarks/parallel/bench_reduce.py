import cupy as cp
import numpy as np
import pytest

import cuda.cccl.parallel.experimental as parallel


def reduce_pointer(input_array, build_only):
    size = len(input_array)
    res = cp.empty(1, dtype=input_array.dtype)
    h_init = np.zeros(1, dtype=input_array.dtype)

    alg = parallel.make_reduce_into(input_array, res, parallel.OpKind.PLUS, h_init)
    if not build_only:
        temp_storage_bytes = alg(None, input_array, res, size, h_init)
        temp_storage = cp.empty(temp_storage_bytes, dtype=np.uint8)
        alg(temp_storage, input_array, res, size, h_init)

    cp.cuda.runtime.deviceSynchronize()


def reduce_pointer_custom_op(input_array, build_only):
    size = len(input_array)
    res = cp.empty(1, dtype=input_array.dtype)
    h_init = np.zeros(1, dtype=input_array.dtype)

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
    res = cp.empty(1, dtype=input_array.dtype)
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
    res = cp.empty(1, dtype=dt)
    h_init = np.zeros(1, dtype=dt)

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


@pytest.mark.parametrize("bench_fixture", ["compile_benchmark", "benchmark"])
def bench_reduce_pointer(bench_fixture, request, size):
    # Use small size for compile benchmarks, parameterized size for runtime benchmarks
    actual_size = 10 if bench_fixture == "compile_benchmark" else size
    input_array = cp.random.randint(0, 10, actual_size)

    def run():
        reduce_pointer(input_array, build_only=(bench_fixture == "compile_benchmark"))

    fixture = request.getfixturevalue(bench_fixture)
    if bench_fixture == "compile_benchmark":
        fixture(parallel.make_reduce_into, run)
    else:
        fixture(run)


@pytest.mark.parametrize("bench_fixture", ["compile_benchmark", "benchmark"])
def bench_reduce_iterator(bench_fixture, request, size):
    inp = parallel.CountingIterator(np.int32(0))
    # Use small size for compile benchmarks, parameterized size for runtime benchmarks
    actual_size = 10 if bench_fixture == "compile_benchmark" else size

    def run():
        reduce_iterator(
            inp, actual_size, build_only=(bench_fixture == "compile_benchmark")
        )

    fixture = request.getfixturevalue(bench_fixture)
    if bench_fixture == "compile_benchmark":
        fixture(parallel.make_reduce_into, run)
    else:
        fixture(run)


@pytest.mark.parametrize("bench_fixture", ["compile_benchmark", "benchmark"])
def bench_reduce_struct(bench_fixture, request, size):
    # Use small size for compile benchmarks, parameterized size for runtime benchmarks
    actual_size = 10 if bench_fixture == "compile_benchmark" else size
    input_array = cp.random.randint(0, 10, (actual_size, 2), dtype="int32").view(
        MyStruct
    )

    def run():
        reduce_struct(input_array, build_only=(bench_fixture == "compile_benchmark"))

    fixture = request.getfixturevalue(bench_fixture)
    if bench_fixture == "compile_benchmark":
        fixture(parallel.make_reduce_into, run)
    else:
        fixture(run)


@pytest.mark.parametrize("bench_fixture", ["compile_benchmark", "benchmark"])
def bench_reduce_pointer_custom_op(bench_fixture, request, size):
    input_array = cp.random.randint(0, 10, size)

    def run():
        reduce_pointer_custom_op(input_array, build_only=False)

    fixture = request.getfixturevalue(bench_fixture)
    if bench_fixture == "compile_benchmark":
        fixture(parallel.make_reduce_into, run)
    else:
        fixture(run)


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
    res = cp.empty(1, dtype=input_array.dtype)
    h_init = np.zeros(1, dtype=input_array.dtype)

    parallel.reduce_into(input_array, res, parallel.OpKind.PLUS, size, h_init)

    cp.cuda.runtime.deviceSynchronize()


def reduce_struct_single_phase(input_array, build_only):
    """Single-phase API that automatically manages temporary storage for structs."""
    size = len(input_array)
    res = cp.empty(1, dtype=input_array.dtype)
    h_init = MyStruct(0, 0)

    def my_add(a, b):
        return MyStruct(a.x + b.x, a.y + b.y)

    parallel.reduce_into(input_array, res, my_add, size, h_init)

    cp.cuda.runtime.deviceSynchronize()


def reduce_iterator_single_phase(inp, size, build_only):
    """Single-phase API that automatically manages temporary storage for iterators."""
    dt = cp.int32
    res = cp.empty(1, dtype=dt)
    h_init = np.zeros(1, dtype=dt)

    parallel.reduce_into(inp, res, parallel.OpKind.PLUS, size, h_init)

    cp.cuda.runtime.deviceSynchronize()
