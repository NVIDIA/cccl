# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Sum each feature independently across each warp."""

import cutlass
import numpy as np
from cutlass import cute
from cutlass.cute.runtime import make_ptr

from cuda.bindings import driver


def _check(result):
    """Raise on Driver errors and unwrap a single returned handle or value."""

    if int(result[0]):
        raise RuntimeError(f"CUDA Driver call failed: {result[0]}")
    return result[1] if len(result) == 2 else result[1:]


def run_example(api="common", items_per_thread=3):
    """Sum matching features in two independent warps and check against NumPy.

    Reshaping the input as warp, lane, feature makes the CPU reference reduce
    only the lane axis. The kernel writes each defined striped result once,
    leaving the two warps' feature totals separate.
    """

    if api == "common":
        from cuda import coop
    elif api == "qualified":
        import cuda.coop.cutlass as coop
    else:
        raise ValueError("api must be 'common' or 'qualified'")

    # docs: start cutlass-reduce-batched
    @cute.kernel
    def feature_sums(
        samples: cute.Pointer,
        totals: cute.Pointer,
        items_per_thread: cutlass.Constexpr,
    ):
        warp = coop.this_warp()
        features = coop.ThreadData(items_per_thread)
        coop.load(warp, samples, features)
        sums = coop.reduce_batched(warp, features)
        outputs = cute.make_tensor(
            totals, cute.make_layout(2 * items_per_thread)
        )
        # Output slots are striped across the warp lanes.
        warp_index = coop.this_block().rank() // 32
        for item in cutlass.range_constexpr((items_per_thread + 31) // 32):
            feature = warp.rank() + item * 32
            if feature < items_per_thread:
                outputs[warp_index * items_per_thread + feature] = sums[item]

    @cute.jit
    def launch(
        samples: cute.Pointer,
        totals: cute.Pointer,
        items_per_thread: cutlass.Constexpr,
    ):
        feature_sums(samples, totals, items_per_thread).launch(
            grid=1, block=(8, 4, 2)
        )

    # docs: end cutlass-reduce-batched

    samples = np.arange(64 * items_per_thread, dtype=np.int32)
    totals = np.zeros(2 * items_per_thread, dtype=np.int32)
    cutlass.cuda.initialize_cuda_context()
    src = _check(driver.cuMemAlloc(samples.nbytes))
    try:
        dst = _check(driver.cuMemAlloc(totals.nbytes))
        try:
            _check(
                driver.cuMemcpyHtoD(src, samples.ctypes.data, samples.nbytes)
            )
            _check(driver.cuMemcpyHtoD(dst, totals.ctypes.data, totals.nbytes))
            pointers = [
                make_ptr(
                    cutlass.Int32,
                    int(pointer),
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                for pointer in (src, dst)
            ]
            compiled = cute.compile(launch, *pointers, items_per_thread)
            compiled(*pointers)
            _check(driver.cuCtxSynchronize())
            _check(driver.cuMemcpyDtoH(totals.ctypes.data, dst, totals.nbytes))
        finally:
            _check(driver.cuMemFree(dst))
    finally:
        _check(driver.cuMemFree(src))
    expected = (
        samples.reshape(2, 32, items_per_thread)
        .sum(axis=1, dtype=np.int32)
        .ravel()
    )
    np.testing.assert_array_equal(totals, expected)
    return totals


if __name__ == "__main__":
    run_example()
