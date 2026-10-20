# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Exchange a block payload layout and shift its elements by one position."""

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


def run_example(api="common", items_per_thread=3):
    """Check a layout conversion followed by a one-item downward shift.

    The CPU reference builds striped order: thread t holds source[t],
    source[t + 64], and so on. It shifts that flattened sequence down by one
    and appends zero. The kernel explicitly repairs Shuffle's undefined final
    item before Store. The comparison checks the composition of two
    operations. Common and qualified calls use the same reference.
    """

    tile_size = _THREADS * items_per_thread
    if api == "common":
        from cuda import coop
    elif api == "qualified":
        import cuda.coop.cutlass as coop
    else:
        raise ValueError("api must be 'common' or 'qualified'")

    # docs: start cutlass-exchange-shuffle
    @cute.kernel
    def rearrange(
        source: cute.Pointer,
        destination: cute.Pointer,
        items_per_thread: cutlass.Constexpr,
    ):
        group = coop.this_block()
        thread = group.rank()
        payload = coop.ThreadData(items_per_thread)
        coop.load(group, source, payload)
        striped = coop.exchange(group, payload, mode="blocked_to_striped")
        shifted = coop.shuffle(group, striped, mode="down")
        # Shuffle leaves one boundary undefined; initialize it before storing.
        if thread == _THREADS - 1:
            shifted[items_per_thread - 1] = cutlass.Int32(0)
        coop.store(group, destination, shifted)

    @cute.jit
    def launch(
        source: cute.Pointer,
        destination: cute.Pointer,
        items_per_thread: cutlass.Constexpr,
    ):
        rearrange(source, destination, items_per_thread).launch(
            grid=1, block=_BLOCK
        )

    # docs: end cutlass-exchange-shuffle

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
    striped = source.reshape(items_per_thread, _THREADS).T.reshape(-1)
    expected = np.concatenate((striped[1:], np.zeros(1, dtype=np.int32)))
    np.testing.assert_array_equal(destination, expected)
    return destination


if __name__ == "__main__":
    run_example()
