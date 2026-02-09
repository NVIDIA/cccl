import cupy as cp
import numpy as np
import pytest

import cuda.compute
from cuda.compute import (
    CountingIterator,
    OpKind,
    gpu_struct,
)


def merge_sort_pointer(keys, vals, output_keys, output_vals, build_only):
    size = len(keys)

    alg = cuda.compute.make_merge_sort(
        keys, vals, output_keys, output_vals, OpKind.LESS
    )
    if not build_only:
        temp_storage_bytes = alg(
            None, keys, vals, output_keys, output_vals, OpKind.LESS, size
        )
        temp_storage = cp.empty(temp_storage_bytes, dtype=np.uint8)
        alg(temp_storage, keys, vals, output_keys, output_vals, OpKind.LESS, size)

    cp.cuda.runtime.deviceSynchronize()


def merge_sort_pointer_custom_op(keys, vals, output_keys, output_vals, build_only):
    size = len(keys)

    def my_cmp(a: np.int32, b: np.int32) -> np.int32:
        return np.int32(a < b)

    alg = cuda.compute.make_merge_sort(keys, vals, output_keys, output_vals, my_cmp)
    if not build_only:
        temp_storage_bytes = alg(
            None, keys, vals, output_keys, output_vals, my_cmp, size
        )
        temp_storage = cp.empty(temp_storage_bytes, dtype=np.uint8)
        alg(temp_storage, keys, vals, output_keys, output_vals, my_cmp, size)

    cp.cuda.runtime.deviceSynchronize()


def merge_sort_iterator(size, keys, vals, output_keys, output_vals, build_only):
    alg = cuda.compute.make_merge_sort(
        keys, vals, output_keys, output_vals, OpKind.LESS
    )
    if not build_only:
        temp_storage_bytes = alg(
            None, keys, vals, output_keys, output_vals, OpKind.LESS, size
        )
        temp_storage = cp.empty(temp_storage_bytes, dtype=np.uint8)
        alg(temp_storage, keys, vals, output_keys, output_vals, OpKind.LESS, size)

    cp.cuda.runtime.deviceSynchronize()


@gpu_struct
class MyStruct:
    x: np.int32
    y: np.int32


def merge_sort_struct(size, keys, vals, output_keys, output_vals, build_only):
    size = len(keys)

    def my_cmp(a: MyStruct, b: MyStruct) -> np.int8:
        return np.int8(a.x < b.x)

    alg = cuda.compute.make_merge_sort(keys, vals, output_keys, output_vals, my_cmp)
    if not build_only:
        temp_storage_bytes = alg(
            None, keys, vals, output_keys, output_vals, my_cmp, size
        )
        temp_storage = cp.empty(temp_storage_bytes, dtype=np.uint8)
        alg(temp_storage, keys, vals, output_keys, output_vals, my_cmp, size)

    cp.cuda.runtime.deviceSynchronize()


@pytest.mark.parametrize("bench_fixture", ["compile_benchmark", "benchmark"])
def bench_merge_sort_pointer(bench_fixture, request, size):
    # Use small size for compile benchmarks, parameterized size for runtime benchmarks
    actual_size = 100 if bench_fixture == "compile_benchmark" else size
    keys = cp.random.randint(0, 10, actual_size)
    vals = cp.random.randint(0, 10, actual_size)
    output_keys = cp.empty_like(keys)
    output_vals = cp.empty_like(vals)

    def run():
        merge_sort_pointer(
            keys,
            vals,
            output_keys,
            output_vals,
            build_only=(bench_fixture == "compile_benchmark"),
        )

    fixture = request.getfixturevalue(bench_fixture)
    fixture(run)


@pytest.mark.parametrize("bench_fixture", ["compile_benchmark", "benchmark"])
def bench_merge_sort_pointer_custom_op(bench_fixture, request, size):
    actual_size = 100 if bench_fixture == "compile_benchmark" else size
    keys = cp.random.randint(0, 10, actual_size)
    vals = cp.random.randint(0, 10, actual_size)
    output_keys = cp.empty_like(keys)
    output_vals = cp.empty_like(vals)

    def run():
        merge_sort_pointer_custom_op(
            keys,
            vals,
            output_keys,
            output_vals,
            build_only=(bench_fixture == "compile_benchmark"),
        )

    fixture = request.getfixturevalue(bench_fixture)
    fixture(run)


@pytest.mark.parametrize("bench_fixture", ["compile_benchmark", "benchmark"])
def bench_merge_sort_iterator(bench_fixture, request, size):
    # Use small size for compile benchmarks, parameterized size for runtime benchmarks
    actual_size = 100 if bench_fixture == "compile_benchmark" else size
    keys = CountingIterator(np.int32(0))
    vals = CountingIterator(np.int64(0))
    output_keys = cp.empty(actual_size, dtype="int32")
    output_vals = cp.empty(actual_size, dtype="int64")

    def run():
        merge_sort_iterator(
            actual_size,
            keys,
            vals,
            output_keys,
            output_vals,
            build_only=(bench_fixture == "compile_benchmark"),
        )

    fixture = request.getfixturevalue(bench_fixture)
    fixture(run)


@pytest.mark.parametrize("bench_fixture", ["compile_benchmark", "benchmark"])
def bench_merge_sort_struct(bench_fixture, request, size):
    # Use small size for compile benchmarks, parameterized size for runtime benchmarks
    actual_size = 100 if bench_fixture == "compile_benchmark" else size
    keys = cp.random.randint(0, 10, (actual_size, 2)).view(MyStruct.dtype)
    vals = cp.random.randint(0, 10, (actual_size, 2)).view(MyStruct.dtype)
    output_keys = cp.empty_like(keys)
    output_vals = cp.empty_like(vals)

    def run():
        merge_sort_struct(
            actual_size,
            keys,
            vals,
            output_keys,
            output_vals,
            build_only=(bench_fixture == "compile_benchmark"),
        )

    fixture = request.getfixturevalue(bench_fixture)
    fixture(run)
