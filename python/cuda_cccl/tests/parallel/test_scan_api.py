# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


def test_exclusive_scan_max():
    # example-begin exclusive-scan-max
    import cupy as cp
    import numpy as np

    import cuda.cccl.parallel.experimental.algorithms as algorithms

    def max_op(a, b):
        return max(a, b)

    h_init = np.array([1], dtype="int32")
    d_input = cp.array([-5, 0, 2, -3, 2, 4, 0, -1, 2, 8], dtype="int32")
    d_output = cp.empty_like(d_input, dtype="int32")

    # Instantiate scan for the given operator and initial value
    scanner = algorithms.exclusive_scan(d_output, d_output, max_op, h_init)

    # Determine temporary device storage requirements
    temp_storage_size = scanner(None, d_input, d_output, d_input.size, h_init)

    # Allocate temporary storage
    d_temp_storage = cp.empty(temp_storage_size, dtype=np.uint8)

    # Run reduction
    scanner(d_temp_storage, d_input, d_output, d_input.size, h_init)

    # Check the result is correct
    expected = np.asarray([1, 1, 1, 2, 2, 2, 4, 4, 4, 4])
    np.testing.assert_equal(d_output.get(), expected)
    # example-end exclusive-scan-max


def test_inclusive_scan_add():
    # example-begin inclusive-scan-add
    import cupy as cp
    import numpy as np

    import cuda.cccl.parallel.experimental.algorithms as algorithms

    def add_op(a, b):
        return a + b

    h_init = np.array([0], dtype="int32")
    d_input = cp.array([-5, 0, 2, -3, 2, 4, 0, -1, 2, 8], dtype="int32")
    d_output = cp.empty_like(d_input, dtype="int32")

    # Instantiate scan for the given operator and initial value
    scanner = algorithms.inclusive_scan(d_output, d_output, add_op, h_init)

    # Determine temporary device storage requirements
    temp_storage_size = scanner(None, d_input, d_output, d_input.size, h_init)

    # Allocate temporary storage
    d_temp_storage = cp.empty(temp_storage_size, dtype=np.uint8)

    # Run reduction
    scanner(d_temp_storage, d_input, d_output, d_input.size, h_init)

    # Check the result is correct
    expected = np.asarray([-5, -5, -3, -6, -4, 0, 0, -1, 1, 9])
    np.testing.assert_equal(d_output.get(), expected)
    # example-end inclusive-scan-add


def test_reverse_input_iterator():
    # example-begin reverse-input-iterator
    import cupy as cp
    import numpy as np

    import cuda.cccl.parallel.experimental.algorithms as algorithms
    import cuda.cccl.parallel.experimental.iterators as iterators

    def add_op(a, b):
        return a + b

    h_init = np.array([0], dtype="int32")
    d_input = cp.array([-5, 0, 2, -3, 2, 4, 0, -1, 2, 8], dtype="int32")
    d_output = cp.empty_like(d_input, dtype="int32")
    reverse_it = iterators.ReverseInputIterator(d_input)

    # Instantiate scan, determine storage requirements, and allocate storage
    inclusive_scan = algorithms.inclusive_scan(reverse_it, d_output, add_op, h_init)
    temp_storage_size = inclusive_scan(None, reverse_it, d_output, len(d_input), h_init)
    d_temp_storage = cp.empty(temp_storage_size, dtype=np.uint8)

    # Run reduction
    inclusive_scan(d_temp_storage, reverse_it, d_output, len(d_input), h_init)

    # Check the result is correct
    expected = np.asarray([8, 10, 9, 9, 13, 15, 12, 14, 14, 9])
    np.testing.assert_equal(d_output.get(), expected)
    # example-end reverse-input-iterator


def test_reverse_output_iterator():
    # example-begin reverse-output-iterator
    import cupy as cp
    import numpy as np

    import cuda.cccl.parallel.experimental.algorithms as algorithms
    import cuda.cccl.parallel.experimental.iterators as iterators

    def add_op(a, b):
        return a + b

    h_init = np.array([0], dtype="int32")
    d_input = cp.array([-5, 0, 2, -3, 2, 4, 0, -1, 2, 8], dtype="int32")
    d_output = cp.empty_like(d_input, dtype="int32")
    reverse_it = iterators.ReverseOutputIterator(d_output)

    # Instantiate scan, determine storage requirements, and allocate storage
    inclusive_scan = algorithms.inclusive_scan(d_input, reverse_it, add_op, h_init)
    temp_storage_size = inclusive_scan(None, d_input, reverse_it, len(d_input), h_init)
    d_temp_storage = cp.empty(temp_storage_size, dtype=np.uint8)

    # Run reduction
    inclusive_scan(d_temp_storage, d_input, reverse_it, len(d_input), h_init)

    # Check the result is correct
    expected = np.asarray([9, 1, -1, 0, 0, -4, -6, -3, -5, -5])
    np.testing.assert_equal(d_output.get(), expected)
    # example-end reverse-output-iterator
