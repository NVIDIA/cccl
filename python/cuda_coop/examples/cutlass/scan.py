# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Scan a block tile while reusing scratch with Load and Store."""

import cutlass
import numpy as np
from cutlass import cute
from cutlass.cute.runtime import make_ptr

from cuda import coop
from cuda.bindings import driver
from cuda.coop import cutlass as cutlass_coop

_BLOCK = (8, 4, 2)
_THREADS = 64
_ITEMS = 2
_BLOCK_TILE = _THREADS * _ITEMS


def _check(result):
    if int(result[0]):
        raise RuntimeError(f"CUDA Driver call failed: {result[0]}")
    return result[1] if len(result) == 2 else result[1:]


def run_example(api="common"):
    """Run a seeded block scan and verify its ordered prefixes."""

    if api not in {"common", "qualified"}:
        raise ValueError("api must be 'common' or 'qualified'")
    module = coop if api == "common" else cutlass_coop

    # docs: start cutlass-scan
    @cute.kernel
    def scan_tiles(source: cute.Pointer, destination: cute.Pointer):
        group = module.this_block()
        storage = module.TempStorage(sharing="shared", alignment=64)
        payload = module.ThreadData(_ITEMS)
        module.load(group, source, payload, algorithm="transpose", temp_storage=storage)
        scanned = module.exclusive_scan(
            group, payload, initial_value=7, temp_storage=storage
        )
        # Scan returns a new payload. The input retains its original values.
        module.store(
            group, destination, scanned, algorithm="transpose", temp_storage=storage
        )

    @cute.jit
    def launch(source: cute.Pointer, destination: cute.Pointer):
        scan_tiles(source, destination).launch(grid=1, block=_BLOCK)

    # docs: end cutlass-scan

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
    expected = np.concatenate(
        (np.array([7], dtype=np.int32), 7 + source[:-1].cumsum(dtype=np.int32))
    )
    np.testing.assert_array_equal(destination, expected)
    return destination


if __name__ == "__main__":
    run_example()
