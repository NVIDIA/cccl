import math
import sys

import numba
import numpy as np
from llvmlite import ir
from numba import config, cuda, types
from numba.core.extending import intrinsic

import cuda.bench as bench
from cuda import coop

config.CUDA_LOW_OCCUPANCY_WARNINGS = 0


def as_cuda_stream(cs: bench.CudaStream) -> cuda.cudadrv.driver.Stream:
    return cuda.external_stream(cs.addressof())


@intrinsic
def read_clock(typingctx):
    """Read the %clock special register."""
    sig = types.uint32()

    def codegen(context, builder, signature, args):
        ftype = ir.FunctionType(ir.IntType(32), [])
        asm_txt = "mov.u32 $0, %clock;"
        constraint = "=r"
        asm_ir = ir.InlineAsm(ftype, asm_txt, constraint, side_effect=True)
        return builder.call(asm_ir, [])

    return sig, codegen


@intrinsic
def get_smid(typingctx):
    """Read the %smid special register (SM ID)."""
    sig = types.uint32()

    def codegen(context, builder, signature, args):
        ftype = ir.FunctionType(ir.IntType(32), [])
        asm_ir = ir.InlineAsm(ftype, "mov.u32 $0, %smid;", "=r", side_effect=True)
        return builder.call(asm_ir, [])

    return sig, codegen


def make_unrolled_kernel(block_size, warp_sum, unroll_factor, numba_dtype, num_u32s):
    """Generate a kernel with manually unrolled loop."""

    @cuda.jit(device=True)
    def generate_random_data():
        """Generate random data as the target dtype."""
        ret = read_clock()
        for i in range(num_u32s - 1):
            current = read_clock()
            ret = (ret << 32) | current
        return numba_dtype(ret)

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
    data = generate_random_data()

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
    }

    exec(kernel_code, local_ns)
    return local_ns["benchmark_kernel"]


def bench_warp_reduce(state: bench.State):
    type_id = state.get_int64("TypeID")

    types_map = {
        0: ("I8", np.int8),
        1: ("I16", np.int16),
        2: ("I32", np.int32),
        3: ("I64", np.int64),
        4: ("F16", np.float16),
        5: ("F32", np.float32),
        6: ("F64", np.float64),
    }

    dtype_str, dtype = types_map[type_id]

    state.add_summary("Type", dtype_str)

    numba_dtype = numba.from_dtype(dtype)
    block_size = 256
    unroll_factor = 128

    warp_sum = coop.warp.sum(numba_dtype)

    size_bytes = np.dtype(dtype).itemsize
    num_u32s = math.ceil(size_bytes / np.dtype(np.uint32).itemsize)

    benchmark_kernel = make_unrolled_kernel(
        block_size, warp_sum, unroll_factor, numba_dtype, num_u32s
    )

    _sink_buffer = cuda.device_array(16, dtype=np.int32)

    def launcher(launch: bench.Launch):
        exec_stream = as_cuda_stream(launch.get_stream())
        benchmark_kernel[1, block_size, exec_stream](_sink_buffer)

    state.exec(launcher, batched=False)


if __name__ == "__main__":
    b = bench.register(bench_warp_reduce)
    b.add_int64_axis("TypeID", range(0, 7))
    bench.run_all_benchmarks(sys.argv)
