# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# example-begin
"""
Create a stateful custom operator using RawOp.

This example demonstrates how to create a stateful operator that maintains
runtime state (in this case, a counter on the device). The operator selects
even numbers and atomically increments a counter for each selected item.
"""

import struct

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


# Create a device counter initialized to 0
d_counter = cp.zeros(1, dtype=np.int32)

# Pack the counter pointer as state bytes
counter_ptr = d_counter.__cuda_array_interface__["data"][0]
state_bytes = struct.pack("P", counter_ptr)
state_alignment = np.dtype(np.intp).alignment

# Define a C++ stateful select operator
# The operator selects even numbers and counts them using atomic operations
cpp_source = """
extern "C" __device__ void select_even_with_count(void* state, void* input, void* result) {
    // Extract counter pointer from state
    int* counter = *reinterpret_cast<int**>(state);

    // Get input value
    int value = *static_cast<int*>(input);

    // Check if even
    bool is_even = (value % 2 == 0);

    // If selected, atomically increment the counter
    if (is_even) {
        atomicAdd(counter, 1);
    }

    // Store result as bool (uint8)
    *static_cast<unsigned char*>(result) = is_even ? 1 : 0;
}
"""

# Compile C++ to LTOIR
arch = get_arch()
ltoir_bytes = compile_cpp_to_ltoir(cpp_source, arch)

# Create a stateful RawOp with the state bytes
select_op = RawOp(
    ltoir=ltoir_bytes,
    name="select_even_with_count",
    state=state_bytes,
    state_alignment=state_alignment,
)

# Prepare test data: numbers 0 to 19
num_items = 20
h_input = np.arange(num_items, dtype=np.int32)
d_input = cp.array(h_input)

# Allocate output arrays
d_output = cp.empty(num_items, dtype=np.int32)
d_num_selected = cp.empty(1, dtype=np.int32)

# Run select with the stateful operator
cuda.compute.select(d_input, d_output, d_num_selected, select_op, num_items)

# Get results
num_selected = d_num_selected.get()[0]
counter_value = d_counter.get()[0]

# Verify: should have selected 10 even numbers (0, 2, 4, ..., 18)
expected_count = 10
assert num_selected == expected_count, (
    f"Expected {expected_count} selected, got {num_selected}"
)
assert counter_value == expected_count, (
    f"Expected counter={expected_count}, got {counter_value}"
)

# Verify the selected values are correct
selected_values = d_output.get()[:num_selected]
expected_selected = np.arange(0, 20, 2, dtype=np.int32)
assert np.array_equal(selected_values, expected_selected), "Selected values don't match"

print(f"Selected {num_selected} even numbers")
print(f"Counter value: {counter_value}")
print(f"Selected values: {selected_values}")
print("RawOp stateful example completed successfully!")
