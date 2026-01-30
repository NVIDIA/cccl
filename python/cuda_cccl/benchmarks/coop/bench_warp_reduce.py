import math
import sys

import numba
import numpy as np
from llvmlite import ir
from numba import config, cuda, types
from numba.core.extending import intrinsic

import cuda.bench as bench
import cuda.bindings.driver as driver
from cuda import coop
from cuda.core.experimental import Device

config.CUDA_LOW_OCCUPANCY_WARNINGS = 0


@intrinsic
def get_smid(typingctx):
    """Read the %smid special register (SM ID)."""
    sig = types.uint32()

    def codegen(context, builder, signature, args):
        ftype = ir.FunctionType(ir.IntType(32), [])
        asm_ir = ir.InlineAsm(ftype, "mov.u32 $0, %smid;", "=r", side_effect=True)
        return builder.call(asm_ir, [])

    return sig, codegen


def make_unrolled_kernel(block_size, warp_sum, unroll_factor, numba_dtype):
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
        f"data = warp_sum(data)  # iteration {i}" for i in range(unroll_factor)
    )

    kernel_code = f"""
@cuda.jit(link=warp_sum.files, launch_bounds={block_size})
def benchmark_kernel(sink_buffer):
    data = generate_random_data(target_dtype)

    # Manually unrolled {unroll_factor} iterations:
    {unrolled_body}

    sink(data, sink_buffer)
"""

    # Create local namespace with required functions
    local_ns = {
        "cuda": cuda,
        "warp_sum": warp_sum,
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


def bench_warp_reduce(state: bench.State):
    dtype_str = state.get_string("T{ct}")

    types_map = {
        "I8": np.int8,
        "I16": np.int16,
        "I32": np.int32,
        "I64": np.int64,
        "F16": np.float16,
        "F32": np.float32,
        "F64": np.float64,
    }

    dtype = types_map[dtype_str]

    numba_dtype = numba.from_dtype(dtype)
    block_size = 256
    unroll_factor = 128

    warp_sum = coop.warp.sum(numba_dtype)

    benchmark_kernel = make_unrolled_kernel(
        block_size, warp_sum, unroll_factor, numba_dtype
    )

    sink_buffer = cuda.device_array(16, dtype=np.int32)

    # This calls the kernel (and then immediately synchronizes the device) to
    # force compilation so we can extract occupancy info.
    grid_size = get_grid_size(
        state.get_device(), block_size, benchmark_kernel, sink_buffer
    )

    def launcher(_: bench.Launch):
        benchmark_kernel[grid_size, block_size](sink_buffer)

    state.exec(launcher, batched=False)


if __name__ == "__main__":
    b = bench.register(bench_warp_reduce)
    b.add_string_axis("T{ct}", ["I8", "I16", "I32", "I64", "F16", "F32", "F64"])
    bench.run_all_benchmarks(sys.argv)
