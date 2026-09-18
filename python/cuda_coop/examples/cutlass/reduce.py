# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Reduce per-thread payloads over a block, logical warps, and a valid prefix."""

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
    """Run built-in reductions and verify group results and root ownership."""

    if api not in {"common", "qualified"}:
        raise ValueError("api must be 'common' or 'qualified'")
    module = coop if api == "common" else cutlass_coop

    # docs: start cutlass-reduce
    @cute.kernel
    def reduce_tiles(source: cute.Pointer, destination: cute.Pointer):
        block = module.this_block()
        thread = block.rank()
        lanes = module.this_warp().group_by(8)
        inputs = cute.make_tensor(source, cute.make_layout(_BLOCK_TILE))
        outputs = cute.make_tensor(destination, cute.make_layout(2 * _THREADS + 1))
        payload = module.ThreadData(_ITEMS, dtype=cutlass.Int32)
        for item in cutlass.range_constexpr(_ITEMS):
            payload[item] = inputs[thread * _ITEMS + item]
        # Full reductions broadcast a scalar to every member of the group.
        outputs[thread] = module.sum(block, payload)
        outputs[_THREADS + thread] = module.reduce(lanes, payload, binary_op="max")
        # A valid prefix counts contributing threads. Every thread calls;
        # only rank zero consumes the result when broadcast is disabled.
        prefix = module.sum(block, payload[0], broadcast=False, valid_items=23)
        if thread == 0:
            outputs[2 * _THREADS] = prefix

    @cute.jit
    def launch(source: cute.Pointer, destination: cute.Pointer):
        reduce_tiles(source, destination).launch(grid=1, block=_BLOCK)

    # docs: end cutlass-reduce

    source = np.arange(_BLOCK_TILE, dtype=np.int32)
    destination = np.full(2 * _THREADS + 1, -101, dtype=np.int32)
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
        (
            np.full(_THREADS, source.sum(dtype=np.int32), dtype=np.int32),
            np.repeat(source.reshape(-1, 8 * _ITEMS).max(axis=1), 8),
            np.array(
                [source[0 : 23 * _ITEMS : _ITEMS].sum(dtype=np.int32)], dtype=np.int32
            ),
        )
    )
    np.testing.assert_array_equal(destination, expected)
    return destination


if __name__ == "__main__":
    run_example()
