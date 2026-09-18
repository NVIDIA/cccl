# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Exchange a block payload layout and shift its elements by one position."""

import cutlass
import numpy as np
from cutlass import cute
from cutlass.cute.runtime import make_ptr

from cuda import coop
from cuda.bindings import driver
from cuda.coop import cutlass as cutlass_coop

_BLOCK = (8, 4, 2)
_THREADS = 64
_ITEMS = 3
_BLOCK_TILE = _THREADS * _ITEMS


def _check(result):
    if int(result[0]):
        raise RuntimeError(f"CUDA Driver call failed: {result[0]}")
    return result[1] if len(result) == 2 else result[1:]


def run_example(api="common"):
    """Run Exchange and Shuffle and check their independent layout oracle."""

    if api not in {"common", "qualified"}:
        raise ValueError("api must be 'common' or 'qualified'")
    module = coop if api == "common" else cutlass_coop

    # docs: start cutlass-exchange-shuffle
    @cute.kernel
    def rearrange(source: cute.Pointer, destination: cute.Pointer):
        group = module.this_block()
        thread = group.rank()
        payload = module.ThreadData(_ITEMS)
        module.load(group, source, payload)
        striped = module.exchange(group, payload, mode="blocked_to_striped")
        shifted = module.shuffle(group, striped, mode="down")
        # Shuffle leaves one boundary undefined; initialize it before storing.
        if thread == _THREADS - 1:
            shifted[_ITEMS - 1] = cutlass.Int32(0)
        module.store(group, destination, shifted)

    @cute.jit
    def launch(source: cute.Pointer, destination: cute.Pointer):
        rearrange(source, destination).launch(grid=1, block=_BLOCK)

    # docs: end cutlass-exchange-shuffle

    source = np.arange(_BLOCK_TILE, dtype=np.int32)
    destination = np.full(_BLOCK_TILE, -101, dtype=np.int32)
    cutlass.cuda.initialize_cuda_context()
    src = _check(driver.cuMemAlloc(source.nbytes))
    try:
        dst = _check(driver.cuMemAlloc(destination.nbytes))
        try:
            _check(driver.cuMemcpyHtoD(src, source.ctypes.data, source.nbytes))
            _check(
                driver.cuMemcpyHtoD(dst, destination.ctypes.data, destination.nbytes)
            )
            src_pointer = make_ptr(
                cutlass.Int32, int(src), cute.AddressSpace.gmem, assumed_align=16
            )
            dst_pointer = make_ptr(
                cutlass.Int32, int(dst), cute.AddressSpace.gmem, assumed_align=16
            )
            launch(src_pointer, dst_pointer)
            _check(driver.cuCtxSynchronize())
            _check(
                driver.cuMemcpyDtoH(destination.ctypes.data, dst, destination.nbytes)
            )
        finally:
            _check(driver.cuMemFree(dst))
    finally:
        _check(driver.cuMemFree(src))
    striped = source.reshape(_ITEMS, _THREADS).T.reshape(-1)
    expected = np.concatenate((striped[1:], np.zeros(1, dtype=np.int32)))
    np.testing.assert_array_equal(destination, expected)
    return destination


if __name__ == "__main__":
    run_example()
