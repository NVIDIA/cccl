# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


def test_device_reduce():
    # example-begin reduce-min
    import cupy as cp
    import numpy as np

    import cuda.parallel.experimental.algorithms as algorithms

    def min_op(a, b):
        return a if a < b else b

    dtype = np.int32
    h_init = np.array([42], dtype=dtype)
    d_input = cp.array([8, 6, 7, 5, 3, 0, 9], dtype=dtype)
    d_output = cp.empty(1, dtype=dtype)

    # Instantiate reduction for the given operator and initial value
    reduce_into = algorithms.reduce_into(d_output, d_output, min_op, h_init)

    # Determine temporary device storage requirements
    temp_storage_size = reduce_into(None, d_input, d_output, None, h_init)

    # Allocate temporary storage
    d_temp_storage = cp.empty(temp_storage_size, dtype=np.uint8)

    # Run reduction
    reduce_into(d_temp_storage, d_input, d_output, None, h_init)

    # Check the result is correct
    expected_output = 0
    assert (d_output == expected_output).all()
    # example-end reduce-min


def test_cache_modified_input_iterator():
    # example-begin cache-iterator
    import functools

    import cupy as cp
    import numpy as np

    import cuda.parallel.experimental.algorithms as algorithms
    import cuda.parallel.experimental.iterators as iterators

    def add_op(a, b):
        return a + b

    values = [8, 6, 7, 5, 3, 0, 9]
    d_input = cp.array(values, dtype=np.int32)
    d_output = cp.empty(1, dtype=np.int32)

    iterator = iterators.CacheModifiedInputIterator(
        d_input, modifier="stream"
    )  # Input sequence
    h_init = np.array([0], dtype=np.int32)  # Initial value for the reduction
    d_output = cp.empty(1, dtype=np.int32)  # Storage for output

    # Instantiate reduction, determine storage requirements, and allocate storage
    reduce_into = algorithms.reduce_into(iterator, d_output, add_op, h_init)
    temp_storage_size = reduce_into(None, iterator, d_output, len(values), h_init)
    d_temp_storage = cp.empty(temp_storage_size, dtype=np.uint8)

    # Run reduction
    reduce_into(d_temp_storage, iterator, d_output, len(values), h_init)

    expected_output = functools.reduce(lambda a, b: a + b, values)
    assert (d_output == expected_output).all()
    # example-end cache-iterator


def test_constant_iterator():
    # example-begin constant-iterator
    import functools

    import cupy as cp
    import numpy as np

    import cuda.parallel.experimental.algorithms as algorithms
    import cuda.parallel.experimental.iterators as iterators

    def add_op(a, b):
        return a + b

    value = 10
    num_items = 3

    constant_it = iterators.ConstantIterator(np.int32(value))  # Input sequence
    h_init = np.array([0], dtype=np.int32)  # Initial value for the reduction
    d_output = cp.empty(1, dtype=np.int32)  # Storage for output

    # Instantiate reduction, determine storage requirements, and allocate storage
    reduce_into = algorithms.reduce_into(constant_it, d_output, add_op, h_init)
    temp_storage_size = reduce_into(None, constant_it, d_output, num_items, h_init)
    d_temp_storage = cp.empty(temp_storage_size, dtype=np.uint8)

    # Run reduction
    reduce_into(d_temp_storage, constant_it, d_output, num_items, h_init)

    expected_output = functools.reduce(lambda a, b: a + b, [value] * num_items)
    assert (d_output == expected_output).all()
    # example-end constant-iterator


def test_counting_iterator():
    # example-begin counting-iterator
    import functools

    import cupy as cp
    import numpy as np

    import cuda.parallel.experimental.algorithms as algorithms
    import cuda.parallel.experimental.iterators as iterators

    def add_op(a, b):
        return a + b

    first_item = 10
    num_items = 3

    first_it = iterators.CountingIterator(np.int32(first_item))  # Input sequence
    h_init = np.array([0], dtype=np.int32)  # Initial value for the reduction
    d_output = cp.empty(1, dtype=np.int32)  # Storage for output

    # Instantiate reduction, determine storage requirements, and allocate storage
    reduce_into = algorithms.reduce_into(first_it, d_output, add_op, h_init)
    temp_storage_size = reduce_into(None, first_it, d_output, num_items, h_init)
    d_temp_storage = cp.empty(temp_storage_size, dtype=np.uint8)

    # Run reduction
    reduce_into(d_temp_storage, first_it, d_output, num_items, h_init)

    expected_output = functools.reduce(
        lambda a, b: a + b, range(first_item, first_item + num_items)
    )
    assert (d_output == expected_output).all()
    # example-end counting-iterator


def test_transform_iterator():
    # example-begin transform-iterator
    import functools

    import cupy as cp
    import numpy as np

    import cuda.parallel.experimental.algorithms as algorithms
    import cuda.parallel.experimental.iterators as iterators

    def add_op(a, b):
        return a + b

    def square_op(a):
        return a**2

    first_item = 10
    num_items = 3

    transform_it = iterators.TransformIterator(
        iterators.CountingIterator(np.int32(first_item)), square_op
    )  # Input sequence
    h_init = np.array([0], dtype=np.int32)  # Initial value for the reduction
    d_output = cp.empty(1, dtype=np.int32)  # Storage for output

    # Instantiate reduction, determine storage requirements, and allocate storage
    reduce_into = algorithms.reduce_into(transform_it, d_output, add_op, h_init)
    temp_storage_size = reduce_into(None, transform_it, d_output, num_items, h_init)
    d_temp_storage = cp.empty(temp_storage_size, dtype=np.uint8)

    # Run reduction
    reduce_into(d_temp_storage, transform_it, d_output, num_items, h_init)

    expected_output = functools.reduce(
        lambda a, b: a + b, [a**2 for a in range(first_item, first_item + num_items)]
    )
    assert (d_output == expected_output).all()
    # example-end transform-iterator


def test_reduce_struct_type():
    # example-begin reduce-struct
    import cupy as cp
    import numpy as np

    from cuda.parallel.experimental import algorithms
    from cuda.parallel.experimental.struct import gpu_struct

    @gpu_struct
    class Pixel:
        r: np.int32
        g: np.int32
        b: np.int32

    def max_g_value(x, y):
        return x if x.g > y.g else y

    d_rgb = cp.random.randint(0, 256, (10, 3), dtype=np.int32).view(Pixel.dtype)
    d_out = cp.empty(1, Pixel.dtype)

    h_init = Pixel(0, 0, 0)

    reduce_into = algorithms.reduce_into(d_rgb, d_out, max_g_value, h_init)
    temp_storage_bytes = reduce_into(None, d_rgb, d_out, len(d_rgb), h_init)

    d_temp_storage = cp.empty(temp_storage_bytes, dtype=np.uint8)
    _ = reduce_into(d_temp_storage, d_rgb, d_out, len(d_rgb), h_init)

    h_rgb = d_rgb.get()
    expected = h_rgb[h_rgb.view("int32")[:, 1].argmax()]

    np.testing.assert_equal(expected["g"], d_out.get()["g"])
    # example-end reduce-struct
