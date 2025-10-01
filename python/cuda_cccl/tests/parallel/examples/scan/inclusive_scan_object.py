# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# example-begin
"""
Inclusive scan example demonstrating the object API.
"""

import cupy as cp
import numpy as np

import cuda.cccl.parallel.experimental as parallel

# Prepare the input and output arrays.
dtype = np.int32
h_init = np.array([0], dtype=dtype)
h_input = np.array([1, 2, 3, 4], dtype=dtype)
d_input = cp.asarray(h_input)
d_output = cp.empty(len(h_input), dtype=dtype)

# Create the scanner object and allocate temporary storage.
scanner = parallel.make_inclusive_scan(d_input, d_output, parallel.OpKind.PLUS, h_init)
temp_storage_size = scanner(None, d_input, d_output, len(h_input), h_init)
d_temp_storage = cp.empty(temp_storage_size, dtype=np.uint8)

# Perform the inclusive scan.
scanner(d_temp_storage, d_input, d_output, len(h_input), h_init)

# Verify the result.
expected_result = np.array([1, 3, 6, 10], dtype=dtype)
actual_result = d_output.get()
np.testing.assert_array_equal(actual_result, expected_result)
print("Inclusive scan object example completed successfully")
