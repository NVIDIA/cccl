# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Reuse block scratch while transforming several tiles in a CuTe loop."""

import cutlass
import numpy as np
from cutlass import cute
from cutlass.cute.runtime import make_ptr

from cuda import coop
from cuda.bindings import driver

_BLOCK = (8, 4, 2)
_ITEMS = 4
_TILE = 256


def _check(result):
    if int(result[0]):
        raise RuntimeError(f"CUDA Driver call failed: {result[0]}")
    return result[1] if len(result) == 2 else result[1:]


def run_example(*, sharing="shared", manual_sync=False):
    """Run eight tiles and verify every output item against NumPy."""

    # docs: start cutlass-block-storage
    @cute.kernel
    def transform(
        source: cute.Pointer, destination: cute.Pointer, tiles: cutlass.Int32
    ):
        storage = coop.TempStorage(
            sharing=sharing, alignment=1, auto_sync=not manual_sync
        )
        for tile in range(tiles):
            payload = coop.ThreadData(_ITEMS)
            coop.load(
                coop.this_block(),
                source,
                payload,
                algorithm="transpose",
                offset=tile * _TILE,
                temp_storage=storage,
            )
            if cutlass.const_expr(manual_sync):
                storage.sync()
            for item in cutlass.range_constexpr(_ITEMS):
                payload[item] = payload[item] + cutlass.Int32(tile + 1)
            coop.store(
                coop.this_block(),
                destination,
                payload,
                algorithm="transpose",
                offset=tile * _TILE,
                temp_storage=storage,
            )
            if cutlass.const_expr(manual_sync):
                storage.sync()

    @cute.jit
    def launch(source: cute.Pointer, destination: cute.Pointer, tiles: cutlass.Int32):
        transform(source, destination, tiles).launch(grid=1, block=_BLOCK)

    # docs: end cutlass-block-storage

    tiles = 8
    source = np.arange(tiles * _TILE, dtype=np.int32)
    destination = np.full_like(source, -101)
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
            launch(src_pointer, dst_pointer, tiles)
            _check(driver.cuCtxSynchronize())
            _check(
                driver.cuMemcpyDtoH(destination.ctypes.data, dst, destination.nbytes)
            )
        finally:
            _check(driver.cuMemFree(dst))
    finally:
        _check(driver.cuMemFree(src))
    expected = (
        source.reshape(tiles, _TILE) + np.arange(1, tiles + 1, dtype=np.int32)[:, None]
    )
    np.testing.assert_array_equal(destination, expected.reshape(-1))
    return destination


if __name__ == "__main__":
    run_example()
