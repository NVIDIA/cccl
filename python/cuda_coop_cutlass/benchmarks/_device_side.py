# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Device-side benchmark harness matching CUB's ``device_side_benchmark.cuh``.

C++ equivalent: ``nvbench_helper/nvbench_helper/device_side_benchmark.cuh``
(``benchmark_kernel`` with ``Unroll = false``) and the host code in
``cub/benchmarks/bench/reduce/warp_reduce_base.cuh``.

Each thread seeds a value from ``%clock``, applies ``action`` ``unroll_factor``
times in a loop that feeds every result into the next call, and passes the
result to a sink that stores only when ``%smid == 0xFFFFFFFF``. The grid holds
as many blocks as fit on the device at once.
"""

import sys

import cuda.bench as bench
import cutlass
from cutlass import cute
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import make_ptr

from cuda.bindings import driver

_FLOAT_TYPES = (cutlass.Float32, cutlass.Float64)
_SINK_BYTES = 64


def _checked(result):
    if int(result[0]):
        raise RuntimeError(f"CUDA Driver call failed: {result[0]}")
    return result[1] if len(result) == 2 else result[1:]


def _random_data(value_type):
    """Fill ``value_type`` with ``%clock`` bits, as ``generate_random_data``."""
    bits = cute.arch.clock()
    if value_type.width > 32:
        high = cute.arch.clock()
        bits = (cutlass.Int64(bits) & 0xFFFFFFFF) | (cutlass.Int64(high) << 32)
    if value_type in _FLOAT_TYPES:
        return value_type(llvm.bitcast(value_type.mlir_type, bits.ir_value()))
    return value_type(bits)


def _make_launcher(action, value_type, block_size, unroll_factor):
    @cute.kernel
    def kernel(sink: cute.Pointer):
        data = _random_data(value_type)
        for _ in cutlass.range(unroll_factor):
            data = action(data)
        if cute.arch.smid() == cutlass.Int32(-1):
            cute.make_tensor(sink, cute.make_layout(1))[0] = data

    @cute.jit
    def launcher(
        sink: cute.Pointer, grid: cutlass.Int32, stream: driver.CUstream
    ):
        kernel(sink).launch(
            grid=(grid, 1, 1), block=(block_size, 1, 1), stream=stream
        )

    return launcher


def _kernel_launch_parameters(stream, launch):
    """Return the function and dynamic shared memory of the captured launch."""
    _checked(
        driver.cuStreamBeginCapture(
            stream,
            driver.CUstreamCaptureMode.CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
        )
    )
    try:
        launch()
    except BaseException:
        ended = driver.cuStreamEndCapture(stream)
        if int(ended[0]) == 0 and int(ended[1]) != 0:
            _checked(driver.cuGraphDestroy(ended[1]))
        raise
    graph = _checked(driver.cuStreamEndCapture(stream))
    try:
        _, count = _checked(driver.cuGraphGetNodes(graph, 0))
        if count != 1:
            raise RuntimeError(f"Expected one kernel node, found {count}")
        nodes, _ = _checked(driver.cuGraphGetNodes(graph, count))
        parameters = _checked(driver.cuGraphKernelNodeGetParams(nodes[0]))
        return parameters.func, parameters.sharedMemBytes
    finally:
        _checked(driver.cuGraphDestroy(graph))


def run_device_side_benchmark(
    state: bench.State,
    action,
    value_type,
    *,
    block_size,
    unroll_factor,
    grid_size=None,
):
    """Compile, size the grid, and time one device-side kernel per state.

    ``action`` is traced inside the kernel and maps one ``value_type`` value to
    another. ``grid_size=None`` uses the occupancy-sized grid.
    """
    stream = driver.CUstream(state.get_stream().addressof())
    sink = _checked(driver.cuMemAlloc(_SINK_BYTES))
    try:
        sink_ptr = make_ptr(
            value_type, int(sink), cute.AddressSpace.gmem, assumed_align=16
        )
        launcher = _make_launcher(action, value_type, block_size, unroll_factor)
        compiled = cute.compile(launcher, sink_ptr, cutlass.Int32(1), stream)

        # Module loading must finish before the capture below.
        compiled(sink_ptr, cutlass.Int32(1), stream)
        _checked(driver.cuStreamSynchronize(stream))
        function, shared_bytes = _kernel_launch_parameters(
            stream, lambda: compiled(sink_ptr, cutlass.Int32(1), stream)
        )

        if grid_size is None:
            blocks_per_sm = _checked(
                driver.cuOccupancyMaxActiveBlocksPerMultiprocessor(
                    function, block_size, shared_bytes
                )
            )
            if blocks_per_sm == 0:
                state.skip(
                    "Skipping: no resident blocks for this type on this device."
                )
                return
            num_sms = _checked(
                driver.cuDeviceGetAttribute(
                    driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                    state.get_device(),
                )
            )
            grid_size = blocks_per_sm * num_sms
        grid = cutlass.Int32(grid_size)

        def launcher_fn(_: bench.Launch):
            compiled(sink_ptr, grid, stream)

        state.exec(launcher_fn, batched=False)
    finally:
        _checked(driver.cuStreamSynchronize(stream))
        _checked(driver.cuMemFree(sink))


def run_all_benchmarks(argv):
    """Run the registered benchmarks with ``argv`` hidden from the CuTe DSL."""
    # The DSL parses sys.argv during compilation, and argparse would read
    # NVBench's "-d" as an abbreviation of the DSL's "-diagnostic" option.
    saved = sys.argv
    sys.argv = argv[:1]
    try:
        bench.run_all_benchmarks(argv)
    finally:
        sys.argv = saved
