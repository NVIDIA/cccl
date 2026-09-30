# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Reduce a full block, eight-thread groups, and a valid input prefix."""

import cutlass
import numpy as np
from cutlass import cute
from cutlass.cute.runtime import make_ptr

from cuda import coop
from cuda.bindings import driver
from cuda.coop import cutlass as cutlass_coop

_BLOCK = (8, 4, 2)
_THREADS = 64


def _check(result):
    if int(result[0]):
        raise RuntimeError(f"CUDA Driver call failed: {result[0]}")
    return result[1] if len(result) == 2 else result[1:]


def run_example(api="common", items_per_thread=2):
    """Run built-in reductions and verify group results and root ownership."""

    tile_size = _THREADS * items_per_thread
    if api not in {"common", "qualified"}:
        raise ValueError("api must be 'common' or 'qualified'")
    module = coop if api == "common" else cutlass_coop

    # docs: start cutlass-reduce
    @cute.kernel
    def reduce_tiles(
        source: cute.Pointer,
        destination: cute.Pointer,
        items_per_thread: cutlass.Constexpr,
    ):
        tile_size = _THREADS * items_per_thread
        block = module.this_block()
        thread = block.rank()
        lanes = module.this_warp().group_by(8)
        inputs = cute.make_tensor(source, cute.make_layout(tile_size))
        outputs = cute.make_tensor(
            destination, cute.make_layout(2 * _THREADS + 1)
        )
        payload = module.ThreadData(items_per_thread)
        for item in cutlass.range_constexpr(items_per_thread):
            payload[item] = inputs[thread * items_per_thread + item]
        # Full reductions broadcast a scalar to every member of the group.
        outputs[thread] = module.sum(block, payload)
        outputs[_THREADS + thread] = module.reduce(
            lanes, payload, binary_op="max"
        )
        # A valid prefix counts contributing threads. Every thread calls;
        # only rank zero consumes the result when broadcast is disabled.
        prefix = module.sum(block, payload[0], broadcast=False, valid_items=23)
        if thread == 0:
            outputs[2 * _THREADS] = prefix

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
    destination = np.full(2 * _THREADS + 1, -101, dtype=np.int32)
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
            np.full(_THREADS, source.sum(dtype=np.int32), dtype=np.int32),
            np.repeat(source.reshape(-1, 8 * items_per_thread).max(axis=1), 8),
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
