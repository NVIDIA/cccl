import cupy as cp
import numpy as np

import cuda.compute.algorithms as algorithms
import cuda.compute.iterators as iterators

from .strided_iterator import make_ndarray_iterator


def test_segmented_reduce_for_columnwise_max():
    # example-begin segmented-reduce-columnwise-maximum

    def binary_op(a, b):
        return max(a, b)

    n_rows, n_cols = 123456, 78
    rng = cp.random.default_rng()
    mat = rng.integers(low=-31, high=32, dtype=np.int16, size=(n_rows, n_cols))

    def make_scaler(step):
        def scale(col_id):
            return col_id * step

        return scale

    zero = np.int32(0)
    row_offset = make_scaler(np.int32(n_rows))
    start_offsets = iterators.TransformIterator(
        iterators.CountingIterator(zero), row_offset
    )

    end_offsets = start_offsets + 1

    d_input = cp.asarray(mat)
    # identity of max operator is the smallest value held by a type
    h_init = np.asarray(np.iinfo(np.int16).min, dtype=np.int16)
    d_output = cp.empty(n_cols, dtype=d_input.dtype)

    # iterator input array permutted so that columns are traversed first
    input_it = make_ndarray_iterator(d_input, (1, 0))

    alg = algorithms.segmented_reduce(
        input_it, d_output, start_offsets, end_offsets, binary_op, h_init
    )

    # query size of temporary storage and allocate
    temp_nbytes = alg(
        None, input_it, d_output, n_cols, start_offsets, end_offsets, h_init
    )
    temp_storage = cp.empty(temp_nbytes, dtype=cp.uint8)
    # launch computation
    alg(temp_storage, input_it, d_output, n_cols, start_offsets, end_offsets, h_init)

    # Verify correctness
    expected = cp.max(mat, axis=0)
    assert cp.all(d_output == expected)
    # example-end segmented-reduce-columnwise-maximum
