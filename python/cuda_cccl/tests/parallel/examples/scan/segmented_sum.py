# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# example-begin
"""
Implement segmented scan using zip iterator and ordinary scan.
"""

import cupy as cp
import numpy as np

import cuda.cccl.parallel.experimental as parallel

# Prepare the input data and head flags.
# Segmented inclusive sum on array of values and head-flags
# array demarkating locations of start of segments can be implemented
# using ordinary inclusive scan using Schwarz operator acting
# of value-flag pairs. `ZipIterator` can be used to efficiently
# load data from pair of input arrays, instead of copying them
# to array of structs.
#
# For example, for data = [1, 1, 1, 1, 1, 1, 1, 1] with
# 3 segments encoded by head_flags = [0, 0, 1, 0, 0, 1, 1, 0]
# corresponding to segmented data [[1, 1], [1, 1, 1], [1], [1, 1]],
# the expected prefix-sum values are [1, 2, 1, 2, 3, 1, 1, 2]
data = cp.asarray([1, 1, 1, 1, 1, 1, 1, 1], dtype=cp.int64)
hflg = cp.asarray([0, 0, 1, 0, 0, 1, 1, 0], dtype=cp.int32)

# Define the custom data type and binary operation.


@parallel.gpu_struct
class ValueFlag:
    value: cp.int64
    flag: cp.int32


def schwartz_sum(op1: ValueFlag, op2: ValueFlag) -> ValueFlag:
    f1: cp.int32 = 1 if op1.flag else 0
    f2: cp.int32 = 1 if op2.flag else 0
    f: cp.int32 = f1 | f2
    v: cp.int64 = op2.value if f2 else op1.value + op2.value
    return ValueFlag(v, f)


# Prepare the output array and initial value.
zip_it = parallel.ZipIterator(data, hflg)
d_output = cp.empty(data.shape, dtype=ValueFlag.dtype)
h_init = ValueFlag(0, 0)

# Perform the segmented scan.
parallel.inclusive_scan(zip_it, d_output, schwartz_sum, h_init, data.size)

# Verify the result.
expected_prefix = np.asarray([1, 2, 1, 2, 3, 1, 1, 2], dtype=np.int64)
result = d_output.get()

assert np.array_equal(result["value"], expected_prefix)
print(f"Segmented sum result: {result}")
