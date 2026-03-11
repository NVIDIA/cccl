# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import math

from llvmlite import ir
from numba import cuda, types
from numba.core.extending import intrinsic

import cuda.bindings.driver as driver
from cuda import coop
from cuda.core import Device


@intrinsic
def get_smid(typingctx):
    """Read the %smid special register (SM ID)."""
    sig = types.uint32()

    def codegen(context, builder, signature, args):
        ftype = ir.FunctionType(ir.IntType(32), [])
        asm_ir = ir.InlineAsm(ftype, "mov.u32 $0, %smid;", "=r", side_effect=True)
        return builder.call(asm_ir, [])

    return sig, codegen


def make_unrolled_kernel(block_size, algorithm_name, unroll_factor, numba_dtype):
    """Generate a kernel with manually unrolled loop."""

    @intrinsic
    def generate_random_data(typingctx, dtype_type):
        """
        Generate random data of the specified type using the local array + memcpy pattern.

        Equivalent to C++:
            uint32_t data[sizeof(T) / sizeof(uint32_t)];
            for (...) data[i] = clock();
            T ret;
            memcpy(&ret, data, sizeof(T));
            return ret;

        Usage: generate_random_data(numba.float64)
        """
        target_type = dtype_type.dtype  # Extract the actual type from Type[T]

        def codegen(context, builder, signature, args):
            # Get LLVM type info
            target_llvm = context.get_value_type(target_type)
            size_bytes = target_llvm.get_abi_size(context.target_data)
            num_u32s = math.ceil(size_bytes / 4)

            # 1. Allocate local array: uint32_t data[num_u32s]
            u32_type = ir.IntType(32)
            array_type = ir.ArrayType(u32_type, num_u32s)
            data_ptr = builder.alloca(array_type, name="data")

            # 2. Fill array with clock values
            # Clock read inline asm
            asm_ftype = ir.FunctionType(ir.IntType(32), [])
            asm_ir = ir.InlineAsm(
                asm_ftype, "mov.u32 $0, %clock;", "=r", side_effect=True
            )

            for i in range(num_u32s):
                clock_val = builder.call(asm_ir, [])
                # GEP to get &data[i]
                elem_ptr = builder.gep(
                    data_ptr,
                    [ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), i)],
                )
                builder.store(clock_val, elem_ptr)

            # 3. Allocate result: T ret
            ret_ptr = builder.alloca(target_llvm, name="ret")

            # 4. memcpy(&ret, data, sizeof(T))
            # Cast both pointers to i8* for memcpy
            i8_ptr_type = ir.PointerType(ir.IntType(8))
            dest = builder.bitcast(ret_ptr, i8_ptr_type)
            src = builder.bitcast(data_ptr, i8_ptr_type)

            # Call LLVM memcpy intrinsic
            memcpy_fn = builder.module.declare_intrinsic(
                "llvm.memcpy", [i8_ptr_type, i8_ptr_type, ir.IntType(64)]
            )
            builder.call(
                memcpy_fn,
                [
                    dest,
                    src,
                    ir.Constant(ir.IntType(64), size_bytes),
                    # not volatile, means it can be optimized away
                    ir.Constant(ir.IntType(1), 0),
                ],
            )

            # 5. return ret
            return builder.load(ret_ptr)

        sig = target_type(dtype_type)
        return sig, codegen

    @cuda.jit(device=True)
    def sink(value, sink_buffer):
        """Prevent dead code elimination. Condition is always false."""
        if get_smid() == 0xFFFFFFFF:
            sink_buffer[0] = value

    # Generate unrolled code as a string
    unrolled_body = "\n    ".join(
        f"data = {algorithm_name}(data)  # iteration {i}" for i in range(unroll_factor)
    )

    kernel_code = f"""
@cuda.jit(link={algorithm_name}.files, launch_bounds={block_size})
def benchmark_kernel(sink_buffer):
    data = generate_random_data(target_dtype)

    # Manually unrolled {unroll_factor} iterations:
    {unrolled_body}

    sink(data, sink_buffer)
"""

    if algorithm_name == "warp_sum":
        algorithm = coop.warp.make_sum(numba_dtype)
    elif algorithm_name == "warp_min":

        def min_op(a, b):
            return a if a < b else b

        algorithm = coop.warp.make_reduce(numba_dtype, min_op)

    # Create local namespace with required functions
    local_ns = {
        "cuda": cuda,
        algorithm_name: algorithm,
        "generate_random_data": generate_random_data,
        "get_smid": get_smid,
        "sink": sink,
        "target_dtype": numba_dtype,
    }

    exec(kernel_code, local_ns)
    return local_ns["benchmark_kernel"]


def get_grid_size(device_id, block_size, kernel, sink_buffer):
    """Get the grid size for the given kernel and block size."""

    # warmup to force compilation so we can extract occupancy info
    kernel[1, block_size](sink_buffer)

    device = Device(device_id)
    device.sync()
    num_SMs = device.properties.multiprocessor_count

    sig = kernel.signatures[0]
    cufunc = kernel.overloads[sig].library.get_cufunc()

    err, max_blocks_per_sm = driver.cuOccupancyMaxActiveBlocksPerMultiprocessor(
        cufunc.handle, block_size, 0
    )
    if err != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"Failed to get occupancy info: {err}")

    return max_blocks_per_sm * num_SMs
