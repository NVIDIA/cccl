# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# example-begin
"""
This example demonstrates how to use cuda.compute with CuPy JIT'd
Python device functions.
"""

import cupy as cp
import cupyx.jit
import numpy as np
from cupyx.jit import _compile as cupy_compile
from cupyx.jit import _cuda_types

from cuda.compute import CompiledOp, reduce_into, types
from cuda.core import Device, Program, ProgramOptions

# =============================================================================
# Helper functions
# =============================================================================


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


def get_cupy_jit_cpp_code(func, arg_types):
    """
    Get the transpiled C++ code from a CuPy JIT device function.

    Args:
        func: CuPy JIT raw kernel (decorated with @cupyx.jit.rawkernel)
        arg_types: Tuple of numpy types (e.g., np.int32, np.float32)

    Returns:
        Tuple of (function_name, cpp_code, return_type)
    """
    in_types = []
    for x in arg_types:
        if isinstance(x, np.dtype):
            t = _cuda_types.Scalar(x)
        elif isinstance(x, type) and issubclass(x, np.generic):
            t = _cuda_types.Scalar(np.dtype(x))
        else:
            raise TypeError(f"{type(x)} is not supported")
        in_types.append(t)
    in_types = tuple(in_types)

    result = cupy_compile.transpile(
        func._func, ["__device__"], func._mode, in_types, None
    )

    return result.func_name, result.code, result.return_type


def create_cccl_wrapper(
    cupy_func_name, cupy_cpp_code, arg_cpp_types, result_cpp_type, wrapper_name
):
    """
    Create a CCCL-compatible wrapper around CuPy JIT generated code.

    CCCL requires the void pointer ABI:
        extern "C" __device__ void name(void* arg0, void* arg1, ..., void* result)
    """
    wrapper_params = []
    arg_casts = []
    call_args = []

    for i, cpp_type in enumerate(arg_cpp_types):
        wrapper_params.append(f"void* arg{i}_ptr")
        arg_casts.append(
            f"    {cpp_type} arg{i} = *static_cast<{cpp_type}*>(arg{i}_ptr);"
        )
        call_args.append(f"arg{i}")

    wrapper_params.append("void* result_ptr")
    params_str = ", ".join(wrapper_params)
    casts_str = "\n".join(arg_casts)
    call_args_str = ", ".join(call_args)

    return f"""
// CuPy JIT generated code
{cupy_cpp_code}

// CCCL-compatible wrapper with void* ABI
extern "C" __device__ void {wrapper_name}({params_str}) {{
{casts_str}
    {result_cpp_type} result = {cupy_func_name}({call_args_str});
    *static_cast<{result_cpp_type}*>(result_ptr) = result;
}}
"""


def cupy_jit_to_compiled_op(cupy_func, arg_numpy_types, wrapper_name):
    """
    Convert a CuPy JIT device function to a CCCL CompiledOp.

    Args:
        cupy_func: CuPy JIT device function
        arg_numpy_types: Tuple of numpy types (np.int32, np.float32, etc.)
        wrapper_name: Name for the CCCL wrapper function

    Returns:
        CompiledOp ready to use with reduce_into, etc.
    """
    func_name, cpp_code, return_type = get_cupy_jit_cpp_code(cupy_func, arg_numpy_types)

    numpy_to_cpp = {
        np.dtype("int8"): "signed char",
        np.dtype("int16"): "short",
        np.dtype("int32"): "int",
        np.dtype("int64"): "long long",
        np.dtype("uint8"): "unsigned char",
        np.dtype("uint16"): "unsigned short",
        np.dtype("uint32"): "unsigned int",
        np.dtype("uint64"): "unsigned long long",
        np.dtype("float32"): "float",
        np.dtype("float64"): "double",
    }

    numpy_to_cccl_type = {
        np.dtype("int8"): types.int8,
        np.dtype("int16"): types.int16,
        np.dtype("int32"): types.int32,
        np.dtype("int64"): types.int64,
        np.dtype("uint8"): types.uint8,
        np.dtype("uint16"): types.uint16,
        np.dtype("uint32"): types.uint32,
        np.dtype("uint64"): types.uint64,
        np.dtype("float32"): types.float32,
        np.dtype("float64"): types.float64,
    }

    arg_cpp_types = []
    arg_cccl_types = []
    for arg in arg_numpy_types:
        dtype = np.dtype(arg)
        arg_cpp_types.append(numpy_to_cpp[dtype])
        arg_cccl_types.append(numpy_to_cccl_type[dtype])

    result_dtype = return_type.dtype
    result_cpp_type = numpy_to_cpp[result_dtype]
    result_cccl_type = numpy_to_cccl_type[result_dtype]

    wrapper_code = create_cccl_wrapper(
        cupy_func_name=func_name,
        cupy_cpp_code=cpp_code,
        arg_cpp_types=arg_cpp_types,
        result_cpp_type=result_cpp_type,
        wrapper_name=wrapper_name,
    )

    arch = get_arch()
    ltoir = compile_to_ltoir(wrapper_code, arch)

    return CompiledOp(
        ltoir=ltoir,
        name=wrapper_name,
        arg_types=tuple(arg_cccl_types),
        return_type=result_cccl_type,
    )


# =============================================================================
# Example: Sum only even numbers using CuPy JIT
# =============================================================================


@cupyx.jit.rawkernel(device=True)
def sum_even_only(x, y):
    """Sum values, but only count even numbers."""
    x_even = x if (x & 1) == 0 else 0
    y_even = y if (y & 1) == 0 else 0
    return x_even + y_even


# Create CompiledOp from the CuPy JIT function
op = cupy_jit_to_compiled_op(sum_even_only, (np.int32, np.int32), "sum_even_only")

d_input = cp.array([1, 2, 3, 4, 5], dtype=np.int32)
d_output = cp.array([0], dtype=np.int32)
h_init = np.array([0], dtype=np.int32)

reduce_into(d_input, d_output, op, len(d_input), h_init)

result = d_output.get()[0]
expected = 6
assert result == expected, f"Expected {expected}, got {result}"
print(f"Sum of even numbers: {result}")
# example-end
