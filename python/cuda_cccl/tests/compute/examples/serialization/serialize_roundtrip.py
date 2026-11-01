# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# example-begin
"""
Serialize a built reduction to bytes, then reconstruct and run it without
recompiling.
"""

import cupy as cp
import numpy as np

import cuda.compute
from cuda.compute import OpKind

# Build a reduction object for the current device, as usual.
dtype = np.int32
h_init = np.array([0], dtype=dtype)
d_input = cp.array([1, 2, 3, 4, 5], dtype=dtype)
d_output = cp.empty(1, dtype=dtype)

reducer = cuda.compute.make_reduce_into(
    d_in=d_input, d_out=d_output, op=OpKind.PLUS, h_init=h_init
)

# Serialize the compiled build result to a blob of bytes. In practice you would
# write this to a file and load it in a later run or on another machine; here we
# keep it in memory to stay self-contained.
blob = cuda.compute.serialize(reducer)

# Reconstruct the reduction from the blob. This performs no JIT compilation.
restored = cuda.compute.deserialize(blob)

# Invoke the restored object exactly as if it had just been built.
temp_storage_size = restored(
    temp_storage=None,
    d_in=d_input,
    d_out=d_output,
    num_items=len(d_input),
    op=OpKind.PLUS,
    h_init=h_init,
)
d_temp_storage = cp.empty(temp_storage_size, dtype=np.uint8)
restored(
    temp_storage=d_temp_storage,
    d_in=d_input,
    d_out=d_output,
    num_items=len(d_input),
    op=OpKind.PLUS,
    h_init=h_init,
)

# The reconstructed object produces the same result as the original.
assert d_output.get()[0] == 15
print("Serialize/deserialize round-trip result:", d_output.get()[0])
