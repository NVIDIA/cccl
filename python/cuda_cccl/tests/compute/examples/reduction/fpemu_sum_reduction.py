# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# example-begin
"""
Sum an array using software-emulated double precision (``cuda::experimental::fp64emu``).

``fp64emu`` is bit-identical to a ``double`` but its arithmetic is emulated by
libcu++. ``cuda.compute.fp64emu_high`` (also ``fp64emu_mid`` and ``fp64emu_low``)
are NumPy dtypes with the layout of float64, so float64 data is moved in and out
with ``.view``, without copying. Requires the v2 (HostJIT) backend.
"""

import cupy as cp
import numpy as np

import cuda.compute
from cuda.compute import OpKind

# The emulated dtype. ``fp64emu_high`` is correctly rounded (the default,
# also available as ``fp64emu``); ``fp64emu_mid`` and ``fp64emu_low`` trade
# accuracy for speed.
dtype = cuda.compute.fp64emu_high

# Prepare the input and output arrays.
d_input = cp.arange(1, 1001, dtype=np.float64).view(dtype)
d_output = cp.empty(1, dtype=dtype)
h_init = np.array([0.0], dtype=dtype)

# Perform the reduction using emulated arithmetic.
cuda.compute.reduce_into(
    d_in=d_input, d_out=d_output, num_items=len(d_input), op=OpKind.PLUS, h_init=h_init
)

# View the result as float64 and verify it.
result = d_output.view(np.float64)[0]
expected_output = 500500.0
assert result == expected_output
print(f"Emulated-double sum reduction result: {result}")
