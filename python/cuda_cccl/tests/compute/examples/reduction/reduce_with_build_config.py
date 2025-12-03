# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# example-begin
"""
Perform reduction with custom NVRTC compile options using BuildConfig.
"""

import cupy as cp
import numpy as np

import cuda.compute
from cuda.compute import BuildConfig, OpKind

# Prepare the input and output arrays.
dtype = np.float32
h_init = np.array([0.0], dtype=dtype)
d_input = cp.array([1.5, 2.5, 3.5, 4.5, 5.5], dtype=dtype)
d_output = cp.empty(1, dtype=dtype)

# Create BuildConfig with NVRTC compile options.
# Common options include:
# - "-fmad=true": Enable fused multiply-add
# - "-use_fast_math": Enable fast math operations
# - "-lineinfo": Generate line number information for profiling
build_config = BuildConfig(extra_compile_flags=["-fmad=true", "-lineinfo"])

# Perform the reduction with BuildConfig.
cuda.compute.reduce_into(
    d_input, d_output, OpKind.PLUS, len(d_input), h_init, build_config=build_config
)

# Verify the result.
expected_output = 17.5
assert np.isclose(d_output[0], expected_output)
result = d_output[0]
print(f"Sum reduction with BuildConfig result: {result}")
