import math

import cupy as cp
import numpy as np

import cuda.compute.algorithms as algorithms
import cuda.compute.iterators as iterators

from .strided_iterator import make_ndarray_iterator


def test_segmented_reduce_for_multiaxis_sum():
    # example-begin segmented-reduce-multiaxis-sum

    def binary_op(a, b):
        return a + b

    n0, n1, n2, n3 = 123, 18, 231, 17
    rng = cp.random.default_rng()
    arr = rng.integers(low=-31, high=32, dtype=np.int32, size=(n0, n1, n2, n3))

    def make_scaler(step):
        def scale(id):
            return id * step

        return scale

    reduce_axis = (0, 2)
    iterate_axis = (1, 3)

    reduce_nelems = math.prod([arr.shape[i] for i in reduce_axis])
    iter_nelems = math.prod([arr.shape[i] for i in iterate_axis])

    zero = np.int32(0)
    scaler_fn = make_scaler(np.int32(reduce_nelems))
    start_offsets = iterators.TransformIterator(
        iterators.CountingIterator(zero), scaler_fn
    )

    end_offsets = start_offsets + 1

    d_input = arr
    # identity of plus operator is 0
    h_init = np.zeros(tuple(), dtype=np.int32)
    d_output = cp.empty(iter_nelems, dtype=d_input.dtype)

    # iterator input array permutted so that columns are traversed first
    input_it = make_ndarray_iterator(d_input, iterate_axis + reduce_axis)

    alg = algorithms.segmented_reduce(
        input_it, d_output, start_offsets, end_offsets, binary_op, h_init
    )

    # query size of temporary storage and allocate
    temp_nbytes = alg(
        None, input_it, d_output, iter_nelems, start_offsets, end_offsets, h_init
    )
    temp_storage = cp.empty(temp_nbytes, dtype=cp.uint8)
    # launch computation
    alg(
        temp_storage,
        input_it,
        d_output,
        iter_nelems,
        start_offsets,
        end_offsets,
        h_init,
    )

    # Verify correctness
    actual = cp.reshape(d_output, tuple(arr.shape[i] for i in iterate_axis))
    expected = cp.sum(arr, axis=reduce_axis)

    assert cp.all(actual == expected)
    # example-end segmented-reduce-multiaxis-sum
