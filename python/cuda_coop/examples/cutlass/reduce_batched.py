# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Sum three independent features across each warp with common or qualified calls."""

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


def run_example(api="common"):
    if api not in {"common", "qualified"}:
        raise ValueError("api must be 'common' or 'qualified'")
    module = coop if api == "common" else cutlass_coop

    # docs: start cutlass-reduce-batched
    @cute.kernel
    def feature_sums(samples: cute.Pointer, totals: cute.Pointer):
        warp = module.this_warp()
        features = module.ThreadData(3)
        module.load(warp, samples, features)
        sums = module.reduce_batched(warp, features)
        outputs = cute.make_tensor(totals, cute.make_layout(6))
        # Lane 0 owns feature 0's sum, lane 1 feature 1's, and so on.
        if warp.rank() < 3:
            warp_index = module.this_block().rank() // 32
            outputs[warp_index * 3 + warp.rank()] = sums[0]

    @cute.jit
    def launch(samples: cute.Pointer, totals: cute.Pointer):
        feature_sums(samples, totals).launch(grid=1, block=(8, 4, 2))

    # docs: end cutlass-reduce-batched

    samples = np.arange(64 * 3, dtype=np.int32)
    totals = np.zeros(6, dtype=np.int32)
    cutlass.cuda.initialize_cuda_context()
    src = _check(driver.cuMemAlloc(samples.nbytes))
    try:
        dst = _check(driver.cuMemAlloc(totals.nbytes))
        try:
            _check(driver.cuMemcpyHtoD(src, samples.ctypes.data, samples.nbytes))
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
            compiled = cute.compile(launch, *pointers)
            compiled(*pointers)
            _check(driver.cuCtxSynchronize())
            _check(driver.cuMemcpyDtoH(totals.ctypes.data, dst, totals.nbytes))
        finally:
            _check(driver.cuMemFree(dst))
    finally:
        _check(driver.cuMemFree(src))
    expected = samples.reshape(2, 32, 3).sum(axis=1, dtype=np.int32).ravel()
    np.testing.assert_array_equal(totals, expected)
    return totals


if __name__ == "__main__":
    run_example()
