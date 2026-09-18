# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Scan a block tile while reusing scratch with Load and Store."""

import cutlass
import numpy as np
from cutlass import cute
from cutlass.cute.runtime import make_ptr

from cuda.bindings import driver

_BLOCK = (8, 4, 2)
_THREADS = 64


def _check(result):
    """Raise on Driver errors and unwrap a single returned handle or value."""

    if int(result[0]):
        raise RuntimeError(f"CUDA Driver call failed: {result[0]}")
    return result[1] if len(result) == 2 else result[1:]


def run_example(api="common", items_per_thread=2):
    """Check exclusive prefix sums that start from seven across a block tile.

    Load, Scan and Store share one descriptor with automatic synchronization.
    The CPU reference gives each item seven plus the sum of all earlier
    inputs in blocked order. The stored values come from Scan's separate
    output payload. Common and qualified calls use the same comparison.
    """

    tile_size = _THREADS * items_per_thread
    if api == "common":
        from cuda import coop
    elif api == "qualified":
        import cuda.coop.cutlass as coop
    else:
        raise ValueError("api must be 'common' or 'qualified'")

    # docs: start cutlass-scan
    @cute.kernel
    def scan_tiles(
        source: cute.Pointer,
        destination: cute.Pointer,
        items_per_thread: cutlass.Constexpr,
    ):
        group = coop.this_block()
        storage = coop.TempStorage(
            sharing="shared", alignment=64, auto_sync=True
        )
        payload = coop.ThreadData(items_per_thread)
        coop.load(
            group, source, payload, algorithm="transpose", temp_storage=storage
        )
        scanned = coop.exclusive_scan(
            group, payload, initial_value=7, temp_storage=storage
        )
        # Scan returns a new payload. The input retains its original values.
        coop.store(
            group,
            destination,
            scanned,
            algorithm="transpose",
            temp_storage=storage,
        )

    @cute.jit
    def launch(
        source: cute.Pointer,
        destination: cute.Pointer,
        items_per_thread: cutlass.Constexpr,
    ):
        scan_tiles(source, destination, items_per_thread).launch(
            grid=1, block=_BLOCK
        )

    # docs: end cutlass-scan

    source = np.arange(tile_size, dtype=np.int32)
    destination = np.full(tile_size, -101, dtype=np.int32)
    cutlass.cuda.initialize_cuda_context()
    src = _check(driver.cuMemAlloc(source.nbytes))
    try:
        dst = _check(driver.cuMemAlloc(destination.nbytes))
        try:
            _check(driver.cuMemcpyHtoD(src, source.ctypes.data, source.nbytes))
            _check(
                driver.cuMemcpyHtoD(
                    dst, destination.ctypes.data, destination.nbytes
                )
            )
            src_pointer = make_ptr(
                cutlass.Int32,
                int(src),
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            dst_pointer = make_ptr(
                cutlass.Int32,
                int(dst),
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            launch(src_pointer, dst_pointer, items_per_thread)
            _check(driver.cuCtxSynchronize())
            _check(
                driver.cuMemcpyDtoH(
                    destination.ctypes.data, dst, destination.nbytes
                )
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
