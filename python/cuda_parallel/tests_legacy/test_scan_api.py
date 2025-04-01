# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


def test_exclusive_scan_max():
    # example-begin exclusive-scan-max
    import cupy as cp
    import numpy as np

    import cuda.parallel.experimental.algorithms.legacy as algorithms

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

    import cuda.parallel.experimental.algorithms.legacy as algorithms

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
