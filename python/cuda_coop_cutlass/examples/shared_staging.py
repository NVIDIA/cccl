# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Stage a masked global tile with CuTe, then load shared values cooperatively.

The example uses separate Int32 source, shared, and output storage. Every
logical group has its own contiguous tile. Native copy uses striped thread
ownership; returning Load can then select blocked or striped ownership.
"""

import cutlass
import numpy as np
from cutlass import cute
from cutlass.cute.runtime import make_ptr
from cutlass.memory import SmemAllocator

import cuda.coop.cutlass as coop
from cuda.bindings import driver

_THREADS = 64
_OFFSET = 5
_DEFAULT = -7


# docs: start cutlass-native-shared-staging
@cute.kernel
def stage_and_reload(
    source: cute.Pointer,
    destination: cute.Pointer,
    valid_items: cutlass.Int32,
    width: cutlass.Constexpr,
    items_per_thread: cutlass.Constexpr,
    striped: cutlass.Constexpr,
):
    tid, _, _ = cute.arch.thread_idx()
    tile_size = width * items_per_thread
    group_origin = (tid // width) * tile_size
    rank = tid % width
    shared = SmemAllocator().allocate_tensor(
        cutlass.Int32, cute.make_layout(_THREADS * items_per_thread)
    )
    layout = cute.make_layout((width, items_per_thread), stride=(1, width))
    source_tile = cute.make_tensor(source + _OFFSET + group_origin, layout)
    shared_tile = cute.make_tensor(shared.iterator + group_origin, layout)
    source_row = source_tile[rank, None]
    shared_row = shared_tile[rank, None]
    predicates = cute.make_rmem_tensor(items_per_thread, cutlass.Boolean)
    for item in cutlass.range_constexpr(items_per_thread):
        # A false native copy predicate leaves this initialized value intact.
        shared_row[item] = cutlass.Int32(_DEFAULT)
        predicates[item] = rank + item * width < valid_items
    cute.basic_copy_if(predicates, source_row, shared_row)
    # Returning Load may read shared values written by other threads.
    cute.arch.sync_threads()

    if cutlass.const_expr(width == _THREADS):
        group = coop.this_block()
    else:
        group = coop.this_warp().group_by(width)
    if cutlass.const_expr(striped):
        values = coop.load(
            group,
            shared,
            items_per_thread=items_per_thread,
            algorithm="striped",
        )
    else:
        values = coop.load(group, shared, items_per_thread=items_per_thread)
    coop.store(group, destination, values)


@cute.jit
def launch(
    source: cute.Pointer,
    destination: cute.Pointer,
    valid_items: cutlass.Int32,
    width: cutlass.Constexpr,
    items_per_thread: cutlass.Constexpr,
    striped: cutlass.Constexpr,
):
    stage_and_reload(
        source, destination, valid_items, width, items_per_thread, striped
    ).launch(grid=1, block=_THREADS)


# docs: end cutlass-native-shared-staging


def _check(result):
    if int(result[0]):
        raise RuntimeError(f"CUDA Driver call failed: {result[0]}")
    return result[1] if len(result) == 2 else result[1:]


def run_example(*, width=8, items_per_thread=4, striped=False, tail=3):
    """Check source offsets, per-group tails, and item ownership."""
    tile_size = width * items_per_thread
    valid_items = tile_size - tail
    source = np.arange(_OFFSET + _THREADS * items_per_thread, dtype=np.int32)
    output = np.zeros(_THREADS * items_per_thread, dtype=np.int32)
    cutlass.cuda.initialize_cuda_context()
    src = _check(driver.cuMemAlloc(source.nbytes))
    try:
        dst = _check(driver.cuMemAlloc(output.nbytes))
        try:
            _check(driver.cuMemcpyHtoD(src, source.ctypes.data, source.nbytes))
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
            launch(
                src_pointer,
                dst_pointer,
                valid_items,
                width,
                items_per_thread,
                striped,
            )
            _check(driver.cuCtxSynchronize())
            _check(driver.cuMemcpyDtoH(output.ctypes.data, dst, output.nbytes))
        finally:
            _check(driver.cuMemFree(dst))
    finally:
        _check(driver.cuMemFree(src))
    expected = source[_OFFSET:].copy().reshape(-1, tile_size)
    expected[:, valid_items:] = _DEFAULT
    if striped:
        expected = expected.reshape(-1, items_per_thread, width).transpose(
            0, 2, 1
        )
    np.testing.assert_array_equal(output, expected.ravel())
    return output


if __name__ == "__main__":
    run_example()
