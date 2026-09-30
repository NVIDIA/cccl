# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Sum each feature independently across each warp."""

import cutlass
import numpy as np
from cutlass import cute
from cutlass.cute.runtime import make_ptr

from cuda import coop
from cuda.bindings import driver
from cuda.coop import cutlass as cutlass_coop


def _check(result):
    if int(result[0]):
        raise RuntimeError(f"CUDA Driver call failed: {result[0]}")
    return result[1] if len(result) == 2 else result[1:]


def run_example(api="common", items_per_thread=3):
    if api not in {"common", "qualified"}:
        raise ValueError("api must be 'common' or 'qualified'")
    module = coop if api == "common" else cutlass_coop

    # docs: start cutlass-reduce-batched
    @cute.kernel
    def feature_sums(
        samples: cute.Pointer,
        totals: cute.Pointer,
        items_per_thread: cutlass.Constexpr,
    ):
        warp = module.this_warp()
        features = module.ThreadData(items_per_thread)
        module.load(warp, samples, features)
        sums = module.reduce_batched(warp, features)
        outputs = cute.make_tensor(
            totals, cute.make_layout(2 * items_per_thread)
        )
        # Output slots are striped across the warp lanes.
        warp_index = module.this_block().rank() // 32
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
