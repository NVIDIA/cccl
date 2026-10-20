# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Reuse shared bytes between native CuTe code and a cooperative reduction."""

import cutlass
import numpy as np
from cutlass import cute
from cutlass.cute.runtime import make_ptr
from cutlass.memory import SmemAllocator

from cuda.bindings import driver


def _check(result):
    if int(result[0]):
        raise RuntimeError(f"CUDA Driver call failed: {result[0]}")
    return result[1] if len(result) == 2 else result[1:]


def run_example(api="common"):
    """Check sequential workspace reuse and a separate live native buffer."""

    if api == "common":
        from cuda import coop
    elif api == "qualified":
        import cuda.coop.cutlass as coop
    else:
        raise ValueError("api must be 'common' or 'qualified'")

    # docs: start cutlass-shared-workspace
    @cute.jit
    def prepare(values: cute.Tensor):
        thread, _, _ = cute.arch.thread_idx()
        values[thread] = cutlass.Float32(2 * thread + 1)

    @cute.kernel
    def kernel(destination: cute.Pointer):
        block = coop.this_block()
        thread = block.rank()
        storage = coop.TempStorage(sharing="shared", auto_sync=False)
        workspace = storage.reserve(64, cutlass.Float32, alignment=64)
        persistent = SmemAllocator().allocate_tensor(
            cutlass.Float32, cute.make_layout(64), byte_alignment=64
        )
        persistent[thread] = cutlass.Float32(3 * thread + 7)

        # reserve() returns an ordinary tensor consumed by native CuTe code.
        prepare(workspace)
        storage.sync()
        value = workspace[63 - thread]
        # Every thread must finish reading before the collective can overwrite
        # these same bytes with its scratch. The input now lives in registers.
        storage.sync()
        total = coop.sum(block, value, temp_storage=storage)
        storage.sync()

        # Reinitialize the shared view for the next native phase. The separate
        # native allocation retains its values across the collective.
        workspace[thread] = cutlass.Float32(thread + 101)
        storage.sync()
        output = cute.make_tensor(destination, cute.make_layout(129))
        if thread == 0:
            output[0] = total
        output[1 + thread] = persistent[63 - thread]
        output[65 + thread] = workspace[63 - thread]

    @cute.jit
    def launch(destination: cute.Pointer):
        kernel(destination).launch(grid=1, block=64)

    # docs: end cutlass-shared-workspace

    output = np.zeros(129, dtype=np.float32)
    cutlass.cuda.initialize_cuda_context()
    allocation = _check(driver.cuMemAlloc(output.nbytes))
    try:
        pointer = make_ptr(
            cutlass.Float32,
            int(allocation),
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        launch(pointer)
        _check(driver.cuCtxSynchronize())
        _check(
            driver.cuMemcpyDtoH(output.ctypes.data, allocation, output.nbytes)
        )
    finally:
        _check(driver.cuMemFree(allocation))
    assert output[0] == 4096
    reverse = np.arange(63, -1, -1, dtype=np.float32)
    np.testing.assert_array_equal(output[1:65], 3 * reverse + 7)
    np.testing.assert_array_equal(output[65:], reverse + 101)
    return output


if __name__ == "__main__":
    run_example()
