# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Device-side benchmark building blocks for Numba-CUDA-MLIR kernels.

This mirrors ``nvbench_helper/nvbench_helper/device_side_benchmark.cuh`` so a
Python kernel and its CUB counterpart differ only in the algorithm call.
"""

import math

import numpy as np
from numba_cuda_mlir import cuda, extending, types
from numba_cuda_mlir._mlir import ir
from numba_cuda_mlir._mlir.dialects import arith, llvm

import cuda.coop.numba_mlir as coop
from cuda.bindings import driver

NUMBA_TYPES = {
    "I8": types.int8,
    "I16": types.int16,
    "I32": types.int32,
    "I64": types.int64,
    "F16": types.float16,
    "F32": types.float32,
    "F64": types.float64,
}

NUMPY_TYPES = {
    "I8": np.int8,
    "I16": np.int16,
    "I32": np.int32,
    "I64": np.int64,
    "F16": np.float16,
    "F32": np.float32,
    "F64": np.float64,
}


def sreg_clock():
    """Read ``%clock`` (``cuda::ptx::get_sreg_clock``). Device only."""
    raise NotImplementedError("only callable inside a Numba-CUDA-MLIR kernel")


def sreg_smid():
    """Read ``%smid`` (``cuda::ptx::get_sreg_smid``). Device only."""
    raise NotImplementedError("only callable inside a Numba-CUDA-MLIR kernel")


def random_value(dtype):
    """Return a ``dtype`` value built from ``%clock`` reads. Device only.

    Matches ``generate_random_data<T>()``: fill ``ceil(sizeof(T) / 4)`` words
    with ``%clock`` and reinterpret their leading ``sizeof(T)`` bytes as ``T``.
    """
    raise NotImplementedError("only callable inside a Numba-CUDA-MLIR kernel")


@extending.type_callable(sreg_clock)
def _type_sreg_clock(context):
    def typer():
        return types.uint32

    return typer


@extending.type_callable(sreg_smid)
def _type_sreg_smid(context):
    def typer():
        return types.uint32

    return typer


@extending.type_callable(random_value)
def _type_random_value(context):
    def typer(dtype):
        if isinstance(dtype, types.NumberClass) and isinstance(
            dtype.instance_type, (types.Integer, types.Float)
        ):
            return dtype.instance_type

    return typer


def _read_sreg(name):
    i32 = ir.IntegerType.get_signless(32)
    return llvm.inline_asm(
        i32, [], f"mov.u32 $0, %{name};", "=r", has_side_effects=True
    )


@extending.lowering_registry.lower(sreg_clock)
def _lower_sreg_clock(builder, target, args, kwargs):
    builder.store_var(target, _read_sreg("clock"))


@extending.lowering_registry.lower(sreg_smid)
def _lower_sreg_smid(builder, target, args, kwargs):
    builder.store_var(target, _read_sreg("smid"))


@extending.lowering_registry.lower(random_value, types.NumberClass)
def _lower_random_value(builder, target, args, kwargs):
    numba_type = builder.get_numba_type(target.name)
    bits = numba_type.bitwidth
    words = math.ceil(bits / 32)
    int_type = ir.IntegerType.get_signless(bits)
    if words == 1:
        raw = _read_sreg("clock")
        if bits < 32:
            raw = arith.trunci(int_type, raw)
    else:
        # Little-endian memcpy: word i supplies bits [32 * i, 32 * i + 32).
        raw = None
        for i in range(words):
            word = arith.extui(int_type, _read_sreg("clock"))
            if i:
                shift = arith.constant(int_type, 32 * i)
                word = arith.shli(word, shift)
            raw = word if raw is None else arith.ori(raw, word)
    if isinstance(numba_type, types.Float):
        raw = arith.bitcast(builder.get_mlir_type(numba_type), raw)
    builder.store_var(target, raw)


extending.refresh_registries()


@cuda.jit(device=True, forceinline=True)
def sink(value, sink_buffer):
    """Keep ``value`` live; the guarded store never executes."""
    if sreg_smid() == np.uint32(0xFFFFFFFF):
        sink_buffer[0] = value


@cuda.jit(device=True, forceinline=True)
def add_op(left, right):
    """User callback equivalent to ``cuda::std::plus<>``."""
    return left + right


ALGORITHMS = {
    # coop.sum lowers to cub::WarpReduce<T>::Sum.
    "sum": "coop.sum(coop.this_warp(), data)",
    # A Python callback lowers to cub::WarpReduce<T>::Reduce(x, op).
    "reduce_add": "coop.reduce(coop.this_warp(), data, binary_op=add_op)",
}

_KERNEL_TEMPLATE = """
@cuda.jit(launch_bounds={block_size})
def benchmark_kernel(sink_buffer):
    # cuda.coop plans groups before Numba typing and cannot see through the
    # intrinsic, so the same-type cast states the payload dtype.
    data = T(random_value(T))
{body}
    sink(data, sink_buffer)
"""


def make_benchmark_kernel(
    algorithm, type_name, *, block_size=256, unroll_factor=128, unroll=False
):
    """Build the ``benchmark_kernel<block_size, unroll_factor, op, T, unroll>``
    counterpart for ``algorithm`` and ``type_name``.

    ``unroll=False`` emits a constant-trip-count loop, like the C++ kernel's
    ``Unroll=false`` path. ``unroll=True`` repeats the call ``unroll_factor``
    times in the source, like ``cuda::static_for``.
    """
    call = ALGORITHMS[algorithm]
    if unroll:
        body = "\n".join(f"    data = {call}" for _ in range(unroll_factor))
    else:
        body = f"    for _ in range({unroll_factor}):\n        data = {call}"
    source = _KERNEL_TEMPLATE.format(block_size=block_size, body=body)
    namespace = {
        "add_op": add_op,
        "coop": coop,
        "cuda": cuda,
        "random_value": random_value,
        "sink": sink,
        "T": NUMBA_TYPES[type_name],
    }
    filename = f"<benchmark_kernel {algorithm} {type_name}>"
    exec(compile(source, filename, "exec"), namespace)  # noqa: S102
    return namespace["benchmark_kernel"]


def latest_compile_result(kernel):
    """Return the most recently compiled overload of ``kernel``.

    cuda.coop kernels are specialized on the whole launch configuration,
    including the grid, so a kernel has one overload per launch shape.
    """
    overloads = kernel.launch_config_overloads or kernel.overloads
    return list(overloads.values())[-1]


def compiled_function(kernel):
    """Return the ``CUfunction`` of the most recently compiled overload."""
    return latest_compile_result(kernel)._codelibrary.get_cufunc()._handle


def _check(result):
    err, *values = result
    if err != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA driver error: {err}")
    return values[0] if len(values) == 1 else values


def occupancy_grid_size(kernel, block_size, *args):
    """Compile ``kernel`` and size its grid like the CUB benchmarks.

    The grid is ``cudaOccupancyMaxActiveBlocksPerMultiprocessor * num_SMs``
    with no dynamic shared memory. The probe launch uses one block; callers
    must launch once more at the returned grid before timing, because that
    launch shape compiles a separate overload.
    """
    kernel[1, block_size](*args)
    cuda.synchronize()
    blocks_per_sm = _check(
        driver.cuOccupancyMaxActiveBlocksPerMultiprocessor(
            compiled_function(kernel), block_size, 0
        )
    )
    num_sms = cuda.get_current_device().MULTIPROCESSOR_COUNT
    return blocks_per_sm * num_sms


_FUNC_ATTRIBUTE = driver.CUfunction_attribute
_ATTRIBUTES = {
    "registers": _FUNC_ATTRIBUTE.CU_FUNC_ATTRIBUTE_NUM_REGS,
    "static_shared_bytes": _FUNC_ATTRIBUTE.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
    "local_bytes": _FUNC_ATTRIBUTE.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
    "max_threads": _FUNC_ATTRIBUTE.CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
}


def kernel_attributes(kernel):
    """Report the resource usage that determines occupancy."""
    function = compiled_function(kernel)
    return {
        name: _check(driver.cuFuncGetAttribute(attribute, function))
        for name, attribute in _ATTRIBUTES.items()
    }


def kernel_cubin(kernel):
    """Return the cubin of the most recently compiled overload."""
    return latest_compile_result(kernel).metadata["cubin"]
