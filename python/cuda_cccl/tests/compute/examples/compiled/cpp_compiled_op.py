# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# example-begin
"""
Use a CUDA C++ device function with CompiledOp for reduction.

This example shows how to compile C++ source to LTOIR and use it
with cuda.compute's CompiledOp for custom reduction operations.
"""

import cupy as cp
import numpy as np

from cuda.compute import CompiledOp, reduce_into, types
from cuda.core import Device, Program, ProgramOptions


def get_arch():
    """Get the SM architecture string for the current device."""
    device = Device()
    device.set_current()
    cc_major, cc_minor = device.compute_capability
    return f"sm_{cc_major}{cc_minor}"


def compile_to_ltoir(source: str, arch: str) -> bytes:
    """Compile C++ source to LTOIR using cuda.core."""
    opts = ProgramOptions(
        arch=arch,
        relocatable_device_code=True,
        link_time_optimization=True,
    )
    prog = Program(source, "c++", options=opts)
    return prog.compile("ltoir").code


# C++ source for a custom add operation.
# CCCL ABI requires all arguments as void* pointers.
ADD_OP_SOURCE = """
extern "C" __device__ void my_add(void* a_ptr, void* b_ptr, void* result_ptr) {
    int a = *static_cast<int*>(a_ptr);
    int b = *static_cast<int*>(b_ptr);
    *static_cast<int*>(result_ptr) = a + b;
}
"""

# Compile the C++ source to LTOIR
arch = get_arch()
add_ltoir = compile_to_ltoir(ADD_OP_SOURCE, arch)

# Create a CompiledOp from the LTOIR
add_op = CompiledOp(
    ltoir=add_ltoir,
    name="my_add",
    arg_types=(types.int32, types.int32),
    return_type=types.int32,
)

# Prepare input and output arrays
d_input = cp.array([1, 2, 3, 4, 5], dtype=np.int32)
d_output = cp.array([0], dtype=np.int32)
h_init = np.array([0], dtype=np.int32)

# Perform the reduction using the compiled operator
reduce_into(d_input, d_output, add_op, len(d_input), h_init)

# Verify the result
result = d_output.get()[0]
expected = 15  # 1+2+3+4+5
assert result == expected, f"Expected {expected}, got {result}"
print(f"Sum reduction result: {result}")
# example-end
