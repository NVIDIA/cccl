# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compute the running sum of 512 integers with one block of 128 threads."""

import cutlass
import numpy as np
from cutlass import cute
from cutlass.cute.runtime import make_ptr

import cuda.coop.cutlass as coop
from cuda.bindings import driver


# docs: start cutlass-prefix-sum
@cute.kernel
def prefix_sum(
    source: cute.Pointer,
    destination: cute.Pointer,
    items_per_thread: cutlass.Constexpr,
):
    block = coop.this_block()
    values = coop.load(block, source, items_per_thread=items_per_thread)
    sums = coop.inclusive_sum(block, values)
    coop.store(block, destination, sums)


@cute.jit
def launch(
    source: cute.Pointer,
    destination: cute.Pointer,
    items_per_thread: cutlass.Constexpr,
):
    kernel = prefix_sum(source, destination, items_per_thread)
    kernel.launch(grid=1, block=128)
    # docs: end cutlass-prefix-sum


def _check(result):
    """Raise on Driver errors and unwrap a single returned handle or value."""

    if int(result[0]):
        raise RuntimeError(f"CUDA Driver call failed: {result[0]}")
    return result[1] if len(result) == 2 else result[1:]


def run_example(items_per_thread=4):
    """Launch the kernel and compare its output with numpy.cumsum."""

    source = np.arange(128 * items_per_thread, dtype=np.int32)
    result = np.zeros_like(source)
    cutlass.cuda.initialize_cuda_context()
    src = _check(driver.cuMemAlloc(source.nbytes))
    dst = _check(driver.cuMemAlloc(result.nbytes))
    try:
        _check(driver.cuMemcpyHtoD(src, source.ctypes.data, source.nbytes))
        # The source pointer's type tells load that the items are Int32.
        launch(
            make_ptr(cutlass.Int32, int(src), cute.AddressSpace.gmem),
            make_ptr(cutlass.Int32, int(dst), cute.AddressSpace.gmem),
            items_per_thread,
        )
        _check(driver.cuCtxSynchronize())
        _check(driver.cuMemcpyDtoH(result.ctypes.data, dst, result.nbytes))
    finally:
        _check(driver.cuMemFree(src))
        _check(driver.cuMemFree(dst))
    np.testing.assert_array_equal(result, np.cumsum(source, dtype=np.int32))
    return result


if __name__ == "__main__":
    run_example()
