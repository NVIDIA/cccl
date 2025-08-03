import cupy as cp
import numpy as np

import cuda.cccl.parallel.experimental as parallel


def merge_sort_pointer(keys, vals, output_keys, output_vals, build_only):
    size = len(keys)

    def my_cmp(a: np.int32, b: np.int32) -> np.int32:
        return np.int32(a < b)

    alg = parallel.make_merge_sort(keys, vals, output_keys, output_vals, my_cmp)
    if not build_only:
        temp_storage_bytes = alg(None, keys, vals, output_keys, output_vals, size)
        temp_storage = cp.empty(temp_storage_bytes, dtype=np.uint8)
        alg(temp_storage, keys, vals, output_keys, output_vals, size)

    cp.cuda.runtime.deviceSynchronize()


def merge_sort_iterator(size, keys, vals, output_keys, output_vals, build_only):
    def my_cmp(a: np.int32, b: np.int32) -> np.int32:
        return np.int32(a < b)

    alg = parallel.make_merge_sort(keys, vals, output_keys, output_vals, my_cmp)
    if not build_only:
        temp_storage_bytes = alg(None, keys, vals, output_keys, output_vals, size)
        temp_storage = cp.empty(temp_storage_bytes, dtype=np.uint8)
        alg(temp_storage, keys, vals, output_keys, output_vals, size)

    cp.cuda.runtime.deviceSynchronize()


@parallel.gpu_struct
class MyStruct:
    x: np.int32
    y: np.int32


def merge_sort_struct(size, keys, vals, output_keys, output_vals, build_only):
    size = len(keys)

    def my_cmp(a: MyStruct, b: MyStruct) -> np.int8:
        return np.int8(a.x < b.x)

    alg = parallel.make_merge_sort(keys, vals, output_keys, output_vals, my_cmp)
    if not build_only:
        temp_storage_bytes = alg(None, keys, vals, output_keys, output_vals, size)
        temp_storage = cp.empty(temp_storage_bytes, dtype=np.uint8)
        alg(temp_storage, keys, vals, output_keys, output_vals, size)

    cp.cuda.runtime.deviceSynchronize()


def bench_compile_merge_sort_pointer(compile_benchmark):
    size = 100
    keys = cp.random.randint(0, 10, size)
    vals = cp.random.randint(0, 10, size)
    output_keys = cp.empty_like(keys)
    output_vals = cp.empty_like(vals)

    def run():
        merge_sort_pointer(keys, vals, output_keys, output_vals, build_only=True)

    compile_benchmark(parallel.make_merge_sort, run)


def bench_compile_merge_sort_iterator(compile_benchmark):
    size = 100
    keys = parallel.CountingIterator(np.int32(0))
    vals = parallel.CountingIterator(np.int64(0))
    output_keys = cp.empty(size, dtype="int32")
    output_vals = cp.empty(size, dtype="int64")

    def run():
        merge_sort_iterator(size, keys, vals, output_keys, output_vals, build_only=True)

    compile_benchmark(parallel.make_merge_sort, run)


def bench_merge_sort_pointer(benchmark, size):
    keys = cp.random.randint(0, 10, size)
    vals = cp.random.randint(0, 10, size)
    output_keys = cp.empty_like(keys)
    output_vals = cp.empty_like(vals)

    def run():
        merge_sort_pointer(keys, vals, output_keys, output_vals, build_only=False)

    benchmark(run)


def bench_merge_sort_iterator(benchmark, size):
    keys = parallel.CountingIterator(np.int32(0))
    vals = parallel.CountingIterator(np.int64(0))
    output_keys = cp.empty(size, dtype="int32")
    output_vals = cp.empty(size, dtype="int64")

    def run():
        merge_sort_iterator(
            size, keys, vals, output_keys, output_vals, build_only=False
        )

    benchmark(run)


def bench_merge_sort_struct(benchmark, size):
    keys = cp.random.randint(0, 10, (size, 2)).view(MyStruct.dtype)
    vals = cp.random.randint(0, 10, (size, 2)).view(MyStruct.dtype)
    output_keys = cp.empty_like(keys)
    output_vals = cp.empty_like(vals)

    def run():
        merge_sort_struct(size, keys, vals, output_keys, output_vals, build_only=False)

    benchmark(run)
