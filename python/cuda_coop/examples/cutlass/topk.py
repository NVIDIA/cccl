# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Select a partial tile's smallest keys and largest key/value pairs."""

from contextlib import ExitStack

import cutlass
import numpy as np
from cutlass import cute
from cutlass.cute.runtime import make_ptr

from cuda.bindings import driver

_THREADS = 64
_K = 31


def _check(result):
    """Raise on Driver errors and unwrap a single returned handle or value."""

    if int(result[0]):
        raise RuntimeError(f"CUDA Driver call failed: {result[0]}")
    return result[1] if len(result) == 2 else result[1:]


def run_example(api="common", items_per_thread=2):
    """Check selected keys and pair identity without assuming output order.

    Repeated keys allow several correct selections at the boundary. Sorting
    only the selected outputs checks membership while permitting those ties.
    Original-position values identify each chosen input. Comparing each
    selected key's bits with the source key at that position checks the pair
    and also distinguishes positive and negative zero. The kernel also stores
    the original payloads to confirm that TopK left them unchanged. Sentinel
    tails check that Store writes only the defined result prefix.
    """
    tile_size = _THREADS * items_per_thread
    valid_items = tile_size - 35
    selected_items = min(_K, valid_items)
    if api == "common":
        from cuda import coop
    elif api == "qualified":
        import cuda.coop.cutlass as coop
    else:
        raise ValueError("api must be 'common' or 'qualified'")

    # docs: start cutlass-topk
    @cute.kernel
    def select_tile(
        source: cute.Pointer,
        smallest: cute.Pointer,
        largest: cute.Pointer,
        selected_positions: cute.Pointer,
        original_keys: cute.Pointer,
        original_positions: cute.Pointer,
        items_per_thread: cutlass.Constexpr,
    ):
        tile_size = _THREADS * items_per_thread
        valid_items = tile_size - 35
        selected_items = min(_K, valid_items)
        block = coop.this_block()
        keys = coop.ThreadData(items_per_thread)
        positions = coop.ThreadData(items_per_thread)
        coop.load(block, source, keys)
        for item in cutlass.range_constexpr(items_per_thread):
            positions[item] = (
                cutlass.Int32(block.rank()) * items_per_thread + item
            )
        if cutlass.const_expr(api == "qualified"):
            input_keys = keys.to_register_tensor()
            input_positions = positions.to_tensor_ssa(dtype=cutlass.Int32)
        else:
            input_keys = keys
            input_positions = positions
        scratch = coop.TempStorage(alignment=16, auto_sync=True)
        min_keys = coop.topk_min_keys(
            block,
            input_keys,
            k=_K,
            valid_items=valid_items,
            temp_storage=scratch,
        )
        max_keys, max_positions = coop.topk_max_pairs(
            block,
            input_keys,
            input_positions,
            k=_K,
            valid_items=valid_items,
            temp_storage=scratch,
        )
        # Only min(k, valid_items) output positions are defined, in any order.
        coop.store(
            block,
            smallest,
            min_keys,
            valid_items=selected_items,
        )
        coop.store(
            block,
            largest,
            max_keys,
            valid_items=selected_items,
        )
        coop.store(
            block,
            selected_positions,
            max_positions,
            valid_items=selected_items,
        )
        coop.store(block, original_keys, keys)
        coop.store(block, original_positions, positions)

    @cute.jit
    def launch(
        source: cute.Pointer,
        smallest: cute.Pointer,
        largest: cute.Pointer,
        selected_positions: cute.Pointer,
        original_keys: cute.Pointer,
        original_positions: cute.Pointer,
        items_per_thread: cutlass.Constexpr,
    ):
        select_tile(
            source,
            smallest,
            largest,
            selected_positions,
            original_keys,
            original_positions,
            items_per_thread,
        ).launch(grid=1, block=_THREADS)

    # docs: end cutlass-topk

    source = (
        np.random.default_rng(42)
        .integers(-1, 2, size=tile_size)
        .astype(np.float32)
    )
    source[2::3] = 0.0
    source[1::5] = -0.0
    outputs = [
        np.full(tile_size, -999, dtype=dtype)
        for dtype in (np.float32, np.float32, np.int32, np.float32, np.int32)
    ]
    arrays = [source, *outputs]
    cutlass.cuda.initialize_cuda_context()
    with ExitStack() as cleanup:
        allocations = []
        pointers = []
        for array in arrays:
            allocation = _check(driver.cuMemAlloc(array.nbytes))
            cleanup.callback(
                lambda ptr=allocation: _check(driver.cuMemFree(ptr))
            )
            allocations.append(allocation)
            _check(
                driver.cuMemcpyHtoD(allocation, array.ctypes.data, array.nbytes)
            )
            dtype = (
                cutlass.Float32 if array.dtype == np.float32 else cutlass.Int32
            )
            pointers.append(
                make_ptr(
                    dtype,
                    int(allocation),
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
            )
        launch(*pointers, items_per_thread)
        _check(driver.cuCtxSynchronize())
        for array, allocation in zip(outputs, allocations[1:]):
            _check(
                driver.cuMemcpyDtoH(array.ctypes.data, allocation, array.nbytes)
            )

    smallest, largest, positions, original_keys, original_positions = outputs
    ordered = np.sort(source[:valid_items])
    np.testing.assert_array_equal(
        np.sort(smallest[:selected_items]),
        ordered[:selected_items],
    )
    np.testing.assert_array_equal(
        np.sort(largest[:selected_items]),
        ordered[-selected_items:],
    )
    selected_ids = positions[:selected_items]
    assert len(np.unique(selected_ids)) == selected_items
    assert np.all((selected_ids >= 0) & (selected_ids < valid_items))
    # Check pair association and the sign bit of tied floating-point zeros.
    np.testing.assert_array_equal(
        largest[:selected_items].view(np.uint32),
        source[selected_ids].view(np.uint32),
    )
    np.testing.assert_array_equal(
        original_keys.view(np.uint32), source.view(np.uint32)
    )
    np.testing.assert_array_equal(
        original_positions,
        np.arange(tile_size, dtype=np.int32),
    )
    for output in outputs[:3]:
        np.testing.assert_array_equal(output[selected_items:], -999)
    return outputs


if __name__ == "__main__":
    run_example()
