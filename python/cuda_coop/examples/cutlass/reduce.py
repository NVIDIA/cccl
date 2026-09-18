# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Reduce a full block, eight-thread groups, and a valid input prefix."""

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
    """Check full block and logical-warp results plus a scalar input prefix.

    The output has one block sum, one maximum per logical group, and one
    prefix sum, each stored by its group leader. The expected prefix
    sum uses each thread's first item; the count is threads, not contiguous
    input elements. The same checks cover common and qualified calls.
    """

    tile_size = _THREADS * items_per_thread
    if api == "common":
        from cuda import coop
    elif api == "qualified":
        import cuda.coop.cutlass as coop
    else:
        raise ValueError("api must be 'common' or 'qualified'")

    # docs: start cutlass-reduce
    @cute.kernel
    def reduce_tiles(
        source: cute.Pointer,
        destination: cute.Pointer,
        items_per_thread: cutlass.Constexpr,
    ):
        tile_size = _THREADS * items_per_thread
        block = coop.this_block()
        thread = block.rank()
        lanes = coop.this_warp().group_by(8)
        inputs = cute.make_tensor(source, cute.make_layout(tile_size))
        outputs = cute.make_tensor(
            destination, cute.make_layout(_THREADS // 8 + 2)
        )
        payload = coop.ThreadData(items_per_thread)
        for item in cutlass.range_constexpr(items_per_thread):
            payload[item] = inputs[thread * items_per_thread + item]
        total = coop.sum(block, payload)
        maximum = coop.reduce(lanes, payload[0], binary_op="max")
        prefix = coop.sum(block, payload[0], valid_items=23)
        if thread == 0:
            outputs[0] = total
            outputs[_THREADS // 8 + 1] = prefix
        if lanes.rank() == 0:
            outputs[1 + thread // 8] = maximum

    @cute.jit
    def launch(
        source: cute.Pointer,
        destination: cute.Pointer,
        items_per_thread: cutlass.Constexpr,
    ):
        reduce_tiles(source, destination, items_per_thread).launch(
            grid=1, block=_BLOCK
        )

    # docs: end cutlass-reduce

    source = np.arange(tile_size, dtype=np.int32)
    destination = np.full(_THREADS // 8 + 2, -101, dtype=np.int32)
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
        (
            np.array([source.sum(dtype=np.int32)], dtype=np.int32),
            source[::items_per_thread].reshape(-1, 8).max(axis=1),
            np.array(
                [
                    source[0 : 23 * items_per_thread : items_per_thread].sum(
                        dtype=np.int32
                    )
                ],
                dtype=np.int32,
            ),
        )
    )
    np.testing.assert_array_equal(destination, expected)
    return destination


if __name__ == "__main__":
    run_example()
