import cupy as cp
import numpy as np
import pytest

import cuda.compute
from cuda.compute import (
    CountingIterator,
    ZipIterator,
)

Single_dtype = np.dtype([("value", np.int32)], align=True)
Pair_dtype = np.dtype([("first", np.int32), ("second", np.int32)], align=True)


def reduce_zip_array(input_array, build_only):
    size = len(input_array)
    res = cp.empty(1, dtype=Single_dtype)
    h_init = np.void((0,), dtype=Single_dtype)

    # Create zip iterator with single array - wraps each element in Single struct
    zip_iter = ZipIterator(input_array)

    def my_add(a, b):
        return (a.value + b.value,)

    alg = cuda.compute.make_reduce_into(zip_iter, res, my_add, h_init)
    if not build_only:
        temp_storage_bytes = alg(None, zip_iter, res, size, h_init)
        temp_storage = cp.empty(temp_storage_bytes, dtype=np.uint8)
        alg(temp_storage, zip_iter, res, size, h_init)

    cp.cuda.runtime.deviceSynchronize()


def reduce_zip_iterator(inp, size, build_only):
    res = cp.empty(1, dtype=Single_dtype)
    h_init = np.void((0,), dtype=Single_dtype)

    # Create zip iterator with single iterator - wraps each element in Single struct
    zip_iter = ZipIterator(inp)

    def my_add(a, b):
        return (a.value + b.value,)

    alg = cuda.compute.make_reduce_into(zip_iter, res, my_add, h_init)
    if not build_only:
        temp_storage_bytes = alg(None, zip_iter, res, size, h_init)
        temp_storage = cp.empty(temp_storage_bytes, dtype=np.uint8)
        alg(temp_storage, zip_iter, res, size, h_init)

    cp.cuda.runtime.deviceSynchronize()


def reduce_zip_array_iterator(input_array, size, build_only):
    res = cp.empty(1, dtype=Pair_dtype)
    h_init = np.void((0, 0), dtype=Pair_dtype)

    # Create a counting iterator to zip with the array
    inp = CountingIterator(np.int32(0))
    zip_iter = ZipIterator(input_array, inp)

    def my_add(a, b):
        return (a.first + b.first, a.second + b.second)

    alg = cuda.compute.make_reduce_into(zip_iter, res, my_add, h_init)
    if not build_only:
        temp_storage_bytes = alg(None, zip_iter, res, size, h_init)
        temp_storage = cp.empty(temp_storage_bytes, dtype=np.uint8)
        alg(temp_storage, zip_iter, res, size, h_init)

    cp.cuda.runtime.deviceSynchronize()


def transform_zip_array_iterator(zip_iter1, zip_iter2, size, build_only):
    # Create output array with Pair_dtype to store the transformed results
    res = cp.empty(size, dtype=Pair_dtype)

    def my_transform(a, b) -> Pair_dtype:
        # The transform function should handle the zip iterator values
        # Since we're working with zip iterators, a and b will be structs
        # We need to extract the values and create a new Pair
        return (a[0] + b[0], a[1] + b[1])

    alg = cuda.compute.make_binary_transform(zip_iter1, zip_iter2, res, my_transform)

    if not build_only:
        alg(zip_iter1, zip_iter2, res, size)

    cp.cuda.runtime.deviceSynchronize()


@pytest.mark.parametrize("bench_fixture", ["compile_benchmark", "benchmark"])
def bench_zip_array(bench_fixture, request):
    input_array = cp.arange(1000, dtype=cp.int32)

    def run():
        reduce_zip_array(input_array, build_only=(bench_fixture == "compile_benchmark"))

    fixture = request.getfixturevalue(bench_fixture)
    if bench_fixture == "compile_benchmark":
        fixture(cuda.compute.make_reduce_into, run)
    else:
        fixture(run)


@pytest.mark.parametrize("bench_fixture", ["compile_benchmark", "benchmark"])
def bench_zip_iterator(bench_fixture, request):
    inp = CountingIterator(np.int32(0))

    def run():
        reduce_zip_iterator(
            inp, 1000, build_only=(bench_fixture == "compile_benchmark")
        )

    fixture = request.getfixturevalue(bench_fixture)
    if bench_fixture == "compile_benchmark":
        fixture(cuda.compute.make_reduce_into, run)
    else:
        fixture(run)


@pytest.mark.parametrize("bench_fixture", ["compile_benchmark", "benchmark"])
def bench_zip_array_iterator(bench_fixture, request):
    input_array = cp.arange(1000, dtype=cp.int32)

    def run():
        reduce_zip_array_iterator(
            input_array, 1000, build_only=(bench_fixture == "compile_benchmark")
        )

    fixture = request.getfixturevalue(bench_fixture)
    if bench_fixture == "compile_benchmark":
        fixture(cuda.compute.make_reduce_into, run)
    else:
        fixture(run)


@pytest.mark.parametrize("bench_fixture", ["compile_benchmark", "benchmark"])
def bench_transform_zip_array_iterator(bench_fixture, request):
    input_array = cp.arange(1000, dtype=cp.int32)

    # Create two zip iterators with consistent Pair structures
    counting_iter1 = CountingIterator(np.int32(0))
    zip_iter1 = ZipIterator(input_array, counting_iter1)
    counting_iter2 = CountingIterator(np.int32(1))
    zip_iter2 = ZipIterator(input_array, counting_iter2)

    def run():
        transform_zip_array_iterator(
            zip_iter1,
            zip_iter2,
            1000,
            build_only=(bench_fixture == "compile_benchmark"),
        )

    fixture = request.getfixturevalue(bench_fixture)
    if bench_fixture == "compile_benchmark":
        fixture(cuda.compute.make_binary_transform, run)
    else:
        fixture(run)
