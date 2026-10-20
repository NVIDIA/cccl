# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Copy independent partial tiles with eight logical warps in one CuTe block."""

import cutlass
import numpy as np
from cutlass import cute
from cutlass.cute.runtime import make_ptr

from cuda.bindings import driver

_BLOCK = (8, 4, 2)
_WIDTH = 8
_GROUPS = 8
_SOURCE_OFFSET = 3
_DESTINATION_OFFSET = 5


def _check(result):
    """Raise on Driver errors and unwrap a single returned handle or value."""

    if int(result[0]):
        raise RuntimeError(f"CUDA Driver call failed: {result[0]}")
    return result[1] if len(result) == 2 else result[1:]


def run_example(api="common", items_per_thread=4):
    """Check eight logical-group tiles with different valid prefixes.

    Each group loads a different count, then stores its full tile. The CPU
    comparison checks copied values, fill values and automatic tile origins
    within both physical warps. Sentinels outside the stored interval must
    remain untouched. Common and qualified calls use the same check.
    """

    group_tile = _WIDTH * items_per_thread
    block_tile = _GROUPS * group_tile
    if api == "common":
        from cuda import coop
    elif api == "qualified":
        import cuda.coop.cutlass as coop
    else:
        raise ValueError("api must be 'common' or 'qualified'")

    # docs: start cutlass-logical-warp-load-store
    @cute.kernel
    def copy_logical_tiles(
        source: cute.Pointer,
        destination: cute.Pointer,
        items_per_thread: cutlass.Constexpr,
    ):
        group_tile = _WIDTH * items_per_thread
        x, y, z = cute.arch.thread_idx()
        thread = x + _BLOCK[0] * (y + _BLOCK[1] * z)
        group_index = thread // _WIDTH
        group = coop.this_warp().group_by(_WIDTH)
        payload = coop.ThreadData(items_per_thread)
        # The compiler assigns each eight-lane group its own consecutive tile.
        # valid_items counts elements in that group's tile.
        coop.load(
            group,
            source,
            payload,
            algorithm="transpose",
            valid_items=cutlass.Int32(group_tile - 1 - group_index),
            oob_default=-1,
            offset=_SOURCE_OFFSET,
        )
        coop.store(
            group,
            destination,
            payload,
            algorithm="transpose",
            offset=_DESTINATION_OFFSET,
        )

    @cute.jit
    def launch(
        source: cute.Pointer,
        destination: cute.Pointer,
        items_per_thread: cutlass.Constexpr,
    ):
        copy_logical_tiles(source, destination, items_per_thread).launch(
            grid=1, block=_BLOCK
        )

    # docs: end cutlass-logical-warp-load-store

    source = np.arange(block_tile + _SOURCE_OFFSET, dtype=np.int32)
    destination = np.full(
        block_tile + _DESTINATION_OFFSET + 3,
        -101,
        dtype=np.int32,
    )
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
    expected = np.full_like(destination, -101)
    for group_index in range(_GROUPS):
        origin = group_index * group_tile
        count = group_tile - 1 - group_index
        expected[
            _DESTINATION_OFFSET + origin : _DESTINATION_OFFSET
            + origin
            + group_tile
        ] = -1
        expected[
            _DESTINATION_OFFSET + origin : _DESTINATION_OFFSET + origin + count
        ] = source[_SOURCE_OFFSET + origin : _SOURCE_OFFSET + origin + count]
    np.testing.assert_array_equal(destination, expected)
    return destination


if __name__ == "__main__":
    run_example()
