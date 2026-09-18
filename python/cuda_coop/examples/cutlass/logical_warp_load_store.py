# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Copy independent partial tiles with eight logical warps in one CuTe block."""

import cutlass
import numpy as np
from cutlass import cute
from cutlass.cute.runtime import make_ptr

from cuda import coop
from cuda.bindings import driver
from cuda.coop import cutlass as cutlass_coop

_BLOCK = (8, 4, 2)
_ITEMS = 4
_WIDTH = 8
_GROUPS = 8
_GROUP_TILE = _WIDTH * _ITEMS
_BLOCK_TILE = _GROUPS * _GROUP_TILE
_SOURCE_OFFSET = 3
_DESTINATION_OFFSET = 5


def _check(result):
    if int(result[0]):
        raise RuntimeError(f"CUDA Driver call failed: {result[0]}")
    return result[1] if len(result) == 2 else result[1:]


def run_example(api="common"):
    """Run eight independent logical tiles and verify their prefixes and defaults."""

    if api not in {"common", "qualified"}:
        raise ValueError("api must be 'common' or 'qualified'")
    module = coop if api == "common" else cutlass_coop

    # docs: start cutlass-logical-warp-load-store
    @cute.kernel
    def copy_logical_tiles(source: cute.Pointer, destination: cute.Pointer):
        x, y, z = cute.arch.thread_idx()
        thread = x + _BLOCK[0] * (y + _BLOCK[1] * z)
        group_index = thread // _WIDTH
        group = module.this_warp().group_by(_WIDTH)
        payload = module.ThreadData(_ITEMS)
        # The compiler assigns each eight-lane group its own consecutive tile.
        # valid_items counts elements in that group's tile.
        module.load(
            group,
            source,
            payload,
            algorithm="transpose",
            valid_items=cutlass.Int32(_GROUP_TILE - 1 - group_index * 3),
            oob_default=-1,
            offset=_SOURCE_OFFSET,
        )
        module.store(
            group,
            destination,
            payload,
            algorithm="transpose",
            offset=_DESTINATION_OFFSET,
        )

    @cute.jit
    def launch(source: cute.Pointer, destination: cute.Pointer):
        copy_logical_tiles(source, destination).launch(grid=1, block=_BLOCK)

    # docs: end cutlass-logical-warp-load-store

    source = np.arange(_BLOCK_TILE + _SOURCE_OFFSET, dtype=np.int32)
    destination = np.full(_BLOCK_TILE + _DESTINATION_OFFSET + 3, -101, dtype=np.int32)
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
    expected = np.full_like(destination, -101)
    for group_index in range(_GROUPS):
        origin = group_index * _GROUP_TILE
        count = _GROUP_TILE - 1 - group_index * 3
        expected[
            _DESTINATION_OFFSET + origin : _DESTINATION_OFFSET + origin + _GROUP_TILE
        ] = -1
        expected[
            _DESTINATION_OFFSET + origin : _DESTINATION_OFFSET + origin + count
        ] = source[_SOURCE_OFFSET + origin : _SOURCE_OFFSET + origin + count]
    np.testing.assert_array_equal(destination, expected)
    return destination


if __name__ == "__main__":
    run_example()
