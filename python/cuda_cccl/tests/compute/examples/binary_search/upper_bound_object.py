# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# example-begin
import cupy as cp
import numpy as np

import cuda.compute

h_data = np.array([1, 3, 3, 5, 7, 9], dtype=np.int32)
h_values = np.array([0, 3, 4, 10], dtype=np.int32)

d_data = cp.asarray(h_data)
d_values = cp.asarray(h_values)
d_out = cp.empty(len(h_values), dtype=np.uintp)

searcher = cuda.compute.make_upper_bound(d_data, d_values, d_out)
searcher(d_data, d_values, d_out, None, len(d_data), len(d_values))

expected = np.searchsorted(h_data, h_values, side="right").astype(np.uintp)
got = cp.asnumpy(d_out)

assert np.array_equal(got, expected)
