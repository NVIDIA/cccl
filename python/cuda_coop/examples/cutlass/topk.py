# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Select a partial tile's smallest keys and largest key/value pairs."""

from contextlib import ExitStack

import cutlass
import numpy as np
from cutlass import cute
from cutlass.cute.runtime import make_ptr

from cuda import coop
from cuda.bindings import driver
from cuda.coop import cutlass as cutlass_coop

_THREADS = 64
_ITEMS = 2
_TILE = _THREADS * _ITEMS
_K = 31
_VALID = 93
_SELECTED = min(_K, _VALID)


def _check(result):
    if int(result[0]):
        raise RuntimeError(f"CUDA Driver call failed: {result[0]}")
    return result[1] if len(result) == 2 else result[1:]


def run_example(api="common"):
    """Check selected multisets and pair identity without assuming output order."""
    if api not in {"common", "qualified"}:
        raise ValueError("api must be 'common' or 'qualified'")
    module = coop if api == "common" else cutlass_coop

    # docs: start cutlass-topk
    @cute.kernel
    def select_tile(
        source: cute.Pointer,
        smallest: cute.Pointer,
        largest: cute.Pointer,
        selected_positions: cute.Pointer,
        original_keys: cute.Pointer,
        original_positions: cute.Pointer,
    ):
        block = module.this_block()
        keys = module.ThreadData(_ITEMS)
        positions = module.ThreadData(_ITEMS)
        module.load(block, source, keys)
        for item in cutlass.range_constexpr(_ITEMS):
            positions[item] = cutlass.Int32(block.rank()) * _ITEMS + item
        if cutlass.const_expr(api == "qualified"):
            input_keys = keys.to_register_tensor()
            input_positions = positions.to_tensor_ssa(dtype=cutlass.Int32)
        else:
            input_keys = keys
            input_positions = positions
        scratch = module.TempStorage(alignment=16)
        min_keys = module.topk_min_keys(
            block,
            input_keys,
            k=_K,
            valid_items=_VALID,
            temp_storage=scratch,
        )
        max_keys, max_positions = module.topk_max_pairs(
            block,
            input_keys,
            input_positions,
            k=_K,
            valid_items=_VALID,
            temp_storage=scratch,
        )
        # Only min(k, valid_items) output positions are defined, in any order.
        module.store(block, smallest, min_keys, valid_items=_SELECTED)
        module.store(block, largest, max_keys, valid_items=_SELECTED)
        module.store(block, selected_positions, max_positions, valid_items=_SELECTED)
        module.store(block, original_keys, keys)
        module.store(block, original_positions, positions)

    @cute.jit
    def launch(
        source: cute.Pointer,
        smallest: cute.Pointer,
        largest: cute.Pointer,
        selected_positions: cute.Pointer,
        original_keys: cute.Pointer,
        original_positions: cute.Pointer,
    ):
        select_tile(
            source,
            smallest,
            largest,
            selected_positions,
            original_keys,
            original_positions,
        ).launch(grid=1, block=_THREADS)

    # docs: end cutlass-topk

    source = np.random.default_rng(42).integers(-1, 2, size=_TILE).astype(np.float32)
    source[2::3] = 0.0
    source[1::5] = -0.0
    outputs = [
        np.full(_TILE, -999, dtype=dtype)
        for dtype in (np.float32, np.float32, np.int32, np.float32, np.int32)
    ]
    arrays = [source, *outputs]
    cutlass.cuda.initialize_cuda_context()
    with ExitStack() as cleanup:
        allocations = []
        pointers = []
        for array in arrays:
            allocation = _check(driver.cuMemAlloc(array.nbytes))
            cleanup.callback(lambda ptr=allocation: _check(driver.cuMemFree(ptr)))
            allocations.append(allocation)
            _check(driver.cuMemcpyHtoD(allocation, array.ctypes.data, array.nbytes))
            dtype = cutlass.Float32 if array.dtype == np.float32 else cutlass.Int32
            pointers.append(
                make_ptr(
                    dtype, int(allocation), cute.AddressSpace.gmem, assumed_align=16
                )
            )
        launch(*pointers)
        _check(driver.cuCtxSynchronize())
        for array, allocation in zip(outputs, allocations[1:]):
            _check(driver.cuMemcpyDtoH(array.ctypes.data, allocation, array.nbytes))

    smallest, largest, positions, original_keys, original_positions = outputs
    ordered = np.sort(source[:_VALID])
    np.testing.assert_array_equal(np.sort(smallest[:_SELECTED]), ordered[:_SELECTED])
    np.testing.assert_array_equal(np.sort(largest[:_SELECTED]), ordered[-_SELECTED:])
    selected_ids = positions[:_SELECTED]
    assert len(np.unique(selected_ids)) == _SELECTED
    assert np.all((selected_ids >= 0) & (selected_ids < _VALID))
    # Preserve the source bits and association even when tied keys are signed zeros.
    np.testing.assert_array_equal(
        largest[:_SELECTED].view(np.uint32), source[selected_ids].view(np.uint32)
    )
    np.testing.assert_array_equal(original_keys.view(np.uint32), source.view(np.uint32))
    np.testing.assert_array_equal(original_positions, np.arange(_TILE, dtype=np.int32))
    for output in outputs[:3]:
        np.testing.assert_array_equal(output[_SELECTED:], -999)
    return outputs


if __name__ == "__main__":
    run_example()
