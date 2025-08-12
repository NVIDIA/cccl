import cupy as cp
import numpy as np

import cuda.cccl.parallel.experimental as parallel


@parallel.gpu_struct
class Single:
    value: np.int32


@parallel.gpu_struct
class Pair:
    first: np.int32
    second: np.int32


def reduce_zip_array(input_array, build_only):
    size = len(input_array)
    res = cp.empty(1, dtype=Single.dtype)
    h_init = Single(0)

    # Create zip iterator with single array - wraps each element in Single struct
    zip_iter = parallel.ZipIterator(input_array)

    def my_add(a, b):
        return Single(a.value + b.value)

    alg = parallel.make_reduce_into(zip_iter, res, my_add, h_init)
    if not build_only:
        temp_storage_bytes = alg(None, zip_iter, res, size, h_init)
        temp_storage = cp.empty(temp_storage_bytes, dtype=np.uint8)
        alg(temp_storage, zip_iter, res, size, h_init)

    cp.cuda.runtime.deviceSynchronize()


def reduce_zip_iterator(inp, size, build_only):
    res = cp.empty(1, dtype=Single.dtype)
    h_init = Single(0)

    # Create zip iterator with single iterator - wraps each element in Single struct
    zip_iter = parallel.ZipIterator(inp)

    def my_add(a, b):
        return Single(a.value + b.value)

    alg = parallel.make_reduce_into(zip_iter, res, my_add, h_init)
    if not build_only:
        temp_storage_bytes = alg(None, zip_iter, res, size, h_init)
        temp_storage = cp.empty(temp_storage_bytes, dtype=np.uint8)
        alg(temp_storage, zip_iter, res, size, h_init)

    cp.cuda.runtime.deviceSynchronize()


def reduce_zip_array_iterator(input_array, size, build_only):
    res = cp.empty(1, dtype=Pair.dtype)
    h_init = Pair(0, 0)

    # Create a counting iterator to zip with the array
    inp = parallel.CountingIterator(np.int32(0))
    zip_iter = parallel.ZipIterator(input_array, inp)

    def my_add(a, b):
        return Pair(a.first + b.first, a.second + b.second)

    alg = parallel.make_reduce_into(zip_iter, res, my_add, h_init)
    if not build_only:
        temp_storage_bytes = alg(None, zip_iter, res, size, h_init)
        temp_storage = cp.empty(temp_storage_bytes, dtype=np.uint8)
        alg(temp_storage, zip_iter, res, size, h_init)

    cp.cuda.runtime.deviceSynchronize()


def transform_zip_array_iterator(zip_iter1, zip_iter2, size, build_only):
    # Create output array with Pair.dtype to store the transformed results
    res = cp.empty(size, dtype=Pair.dtype)

    def my_transform(a, b):
        # The transform function should handle the zip iterator values
        # Since we're working with zip iterators, a and b will be structs
        # We need to extract the values and create a new Pair
        return Pair(a[0] + b[0], a[1] + b[1])

    parallel.binary_transform(zip_iter1, zip_iter2, res, my_transform, size)

    # Return a zip iterator that combines the result with a counting iterator
    # This creates a zip iterator that pairs the transformed result with indices
    counting_iter = parallel.CountingIterator(np.int32(0))
    result_zip_iter = parallel.ZipIterator(res, counting_iter)

    cp.cuda.runtime.deviceSynchronize()

    return result_zip_iter


def bench_compile_reduce_zip_array(compile_benchmark):
    input_array = cp.arange(1000, dtype=cp.int32)

    def run():
        reduce_zip_array(input_array, build_only=True)

    compile_benchmark(parallel.make_reduce_into, run)


def bench_compile_reduce_zip_iterator(compile_benchmark):
    inp = parallel.CountingIterator(np.int32(0))

    def run():
        reduce_zip_iterator(inp, 1000, build_only=True)

    compile_benchmark(parallel.make_reduce_into, run)


def bench_compile_reduce_zip_array_iterator(compile_benchmark):
    input_array = cp.arange(1000, dtype=cp.int32)

    def run():
        reduce_zip_array_iterator(input_array, 1000, build_only=True)

    compile_benchmark(parallel.make_reduce_into, run)


def bench_compile_transform_zip_array_iterator(compile_benchmark):
    input_array = cp.arange(1000, dtype=cp.int32)

    # Create two zip iterators with consistent Pair structures
    # First zip iterator: pairs input array values with counting iterator (0,1,2,...)
    counting_iter1 = parallel.CountingIterator(np.int32(0))
    zip_iter1 = parallel.ZipIterator(input_array, counting_iter1)

    # Second zip iterator: pairs input array values (shifted by 1) with counting iterator (1,2,3,...)
    counting_iter2 = parallel.CountingIterator(np.int32(1))
    zip_iter2 = parallel.ZipIterator(input_array, counting_iter2)

    def run():
        transform_zip_array_iterator(zip_iter1, zip_iter2, 1000, build_only=True)

    compile_benchmark(parallel.make_binary_transform, run)


def bench_reduce_zip_array(benchmark):
    input_array = cp.arange(1000, dtype=cp.int32)

    def run():
        reduce_zip_array(input_array, build_only=False)

    benchmark(run)


def bench_reduce_zip_iterator(benchmark):
    inp = parallel.CountingIterator(np.int32(0))

    def run():
        reduce_zip_iterator(inp, 1000, build_only=False)

    benchmark(run)


def bench_reduce_zip_array_iterator(benchmark):
    input_array = cp.arange(1000, dtype=cp.int32)

    def run():
        reduce_zip_array_iterator(input_array, 1000, build_only=False)

    benchmark(run)


def bench_transform_zip_array_iterator(benchmark):
    input_array = cp.arange(1000, dtype=cp.int32)

    # Create two zip iterators with consistent Pair structures
    # First zip iterator: pairs input array values with counting iterator (0,1,2,...)
    counting_iter1 = parallel.CountingIterator(np.int32(0))
    zip_iter1 = parallel.ZipIterator(input_array, counting_iter1)

    # Second zip iterator: pairs input array values (shifted by 1) with counting iterator (1,2,3,...)
    counting_iter2 = parallel.CountingIterator(np.int32(1))
    zip_iter2 = parallel.ZipIterator(input_array, counting_iter2)

    def run():
        transform_zip_array_iterator(zip_iter1, zip_iter2, 1000, build_only=False)

    benchmark(run)
