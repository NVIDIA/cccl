# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Copy a partial tile with CUTLASS and ``cuda.coop.load``/``store``."""

import cutlass
import numpy as np
from cutlass import cute
from cutlass.cute.runtime import make_ptr

from cuda import coop
from cuda.bindings import driver
from cuda.coop import cutlass as cutlass_coop

_BLOCK = (8, 4, 1)
_LOAD_OFFSET = 3
_STORE_OFFSET = 5
_DEFAULT = -7
_SENTINEL = -101


def _check(result):
    if int(result[0]):
        raise RuntimeError(f"CUDA Driver call failed: {result[0]}")
    return result[1] if len(result) == 2 else result[1:]


def run_example(api="common", items_per_thread=2):
    """Run the partial copy and compare its complete output with a CPU
    oracle.
    """

    tile_size = 32 * items_per_thread
    load_valid_items = tile_size - 19
    store_valid_items = tile_size - 11
    if api not in {"common", "qualified"}:
        raise ValueError("api must be 'common' or 'qualified'")
    module = coop if api == "common" else cutlass_coop

    # docs: start cutlass-block-load-store
    @cute.kernel
    def block_copy(
        source: cute.Pointer,
        destination: cute.Pointer,
        items_per_thread: cutlass.Constexpr,
    ):
        tile_size = 32 * items_per_thread
        load_valid_items = tile_size - 19
        store_valid_items = tile_size - 11
        block = module.this_block()
        payload = module.ThreadData(items_per_thread)
        module.load(
            block,
            source,
            payload,
            valid_items=load_valid_items,
            oob_default=_DEFAULT,
            offset=_LOAD_OFFSET,
        )
        module.store(
            block,
            destination,
            payload,
            valid_items=store_valid_items,
            offset=_STORE_OFFSET,
        )

    @cute.jit
    def launch(
        source: cute.Pointer,
        destination: cute.Pointer,
        items_per_thread: cutlass.Constexpr,
    ):
        block_copy(source, destination, items_per_thread).launch(
            grid=1, block=_BLOCK
        )

    # docs: end cutlass-block-load-store

    cutlass.cuda.initialize_cuda_context()
    source = np.arange(tile_size + _LOAD_OFFSET, dtype=np.int32)
    destination = np.full(
        tile_size + _STORE_OFFSET + 3, _SENTINEL, dtype=np.int32
    )
    source_device = _check(driver.cuMemAlloc(source.nbytes))
    try:
        destination_device = _check(driver.cuMemAlloc(destination.nbytes))
        try:
            _check(
                driver.cuMemcpyHtoD(
                    source_device, source.ctypes.data, source.nbytes
                )
            )
            _check(
                driver.cuMemcpyHtoD(
                    destination_device,
                    destination.ctypes.data,
                    destination.nbytes,
                )
            )
            source_pointer = make_ptr(
                cutlass.Int32,
                int(source_device),
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            destination_pointer = make_ptr(
                cutlass.Int32,
                int(destination_device),
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            launch(source_pointer, destination_pointer, items_per_thread)
            _check(driver.cuCtxSynchronize())
            _check(
                driver.cuMemcpyDtoH(
                    destination.ctypes.data,
                    destination_device,
                    destination.nbytes,
                )
            )
        finally:
            _check(driver.cuMemFree(destination_device))
    finally:
        _check(driver.cuMemFree(source_device))
    expected = np.full_like(destination, _SENTINEL)
    expected[_STORE_OFFSET : _STORE_OFFSET + store_valid_items] = _DEFAULT
    expected[_STORE_OFFSET : _STORE_OFFSET + load_valid_items] = source[
        _LOAD_OFFSET : _LOAD_OFFSET + load_valid_items
    ]
    np.testing.assert_array_equal(destination, expected)
    return destination


if __name__ == "__main__":
    run_example()
