# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# example-begin
"""
Create a custom C++ operator from LTOIR bytecode using RawOp.

This example demonstrates how to compile C++ device code to LTOIR and use it
with CUDA CCCL algorithms via the RawOp class. RawOp is language-agnostic and
accepts pre-compiled LTOIR from any source (C++, Rust, Julia, custom DSLs, etc.).
"""

import cupy as cp
import numpy as np

import cuda.compute
from cuda.compute.op import RawOp
from cuda.core import Device, Program, ProgramOptions


def get_arch():
    """Get the SM architecture string for the current device."""
    device = Device()
    device.set_current()
    cc_major, cc_minor = device.compute_capability
    return f"sm_{cc_major}{cc_minor}"


def compile_cpp_to_ltoir(source: str, arch: str) -> bytes:
    """Compile C++ source to LTOIR using cuda.core."""
    opts = ProgramOptions(
        arch=arch,
        relocatable_device_code=True,
        link_time_optimization=True,
    )
    prog = Program(source, "c++", options=opts)
    return prog.compile("ltoir").code


# Define a C++ custom multiply operator
cpp_source = """
extern "C" __device__ void multiply_op(void* a, void* b, void* result) {
    *static_cast<int*>(result) = *static_cast<int*>(a) * *static_cast<int*>(b);
}
"""

# Compile C++ to LTOIR
arch = get_arch()
ltoir_bytes = compile_cpp_to_ltoir(cpp_source, arch)

# Create a RawOp from the LTOIR bytecode
multiply_op = RawOp(ltoir=ltoir_bytes, name="multiply_op")

# Prepare test data
h_input = np.array([1, 2, 3, 4, 5], dtype=np.int32)
d_input = cp.array(h_input)
d_output = cp.empty(1, dtype=np.int32)
h_init = np.array(1, dtype=np.int32)

# Use the custom operator with reduce_into
cuda.compute.reduce_into(d_input, d_output, multiply_op, len(d_input), h_init)

# Verify the result
result = d_output.get()[0]
expected = np.prod(h_input)  # 1 * 2 * 3 * 4 * 5 = 120
assert result == expected, f"Expected {expected}, got {result}"

print(f"Custom multiply reduction result: {result}")
print("RawOp stateless example completed successfully!")
