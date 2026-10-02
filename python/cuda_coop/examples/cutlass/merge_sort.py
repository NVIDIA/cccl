# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Sort a partial block tile while preserving keys and associated values."""

from contextlib import ExitStack

import cutlass
import numpy as np
from cutlass import cute
from cutlass.cute.runtime import make_ptr

from cuda import coop
from cuda.bindings import driver
from cuda.coop import cutlass as cutlass_coop

_BLOCK = (8, 4, 2)


def _check(result):
    if int(result[0]):
        raise RuntimeError(f"CUDA Driver call failed: {result[0]}")
    return result[1] if len(result) == 2 else result[1:]


def run_example(api="common", items_per_thread=3):
    """Check key ordering, pair association, and unchanged input payloads."""
    tile_size = 64 * items_per_thread
    valid_items = tile_size - 7
    if api not in {"common", "qualified"}:
        raise ValueError("api must be 'common' or 'qualified'")
    module = coop if api == "common" else cutlass_coop

    # docs: start cutlass-merge-sort
    @cute.kernel
    def sort_tile(
        source_keys: cute.Pointer,
        source_values: cute.Pointer,
        ascending: cute.Pointer,
        descending: cute.Pointer,
        sorted_values: cute.Pointer,
        original_keys: cute.Pointer,
        original_values: cute.Pointer,
        items_per_thread: cutlass.Constexpr,
    ):
        tile_size = 64 * items_per_thread
        valid_items = tile_size - 7
        group = module.this_block()
        keys = module.ThreadData(items_per_thread)
        values = module.ThreadData(items_per_thread)
        module.load(
            group,
            source_keys,
            keys,
            valid_items=valid_items,
            oob_default=1000,
        )
        module.load(
            group,
            source_values,
            values,
            valid_items=valid_items,
            oob_default=0.0,
        )
        scratch = module.TempStorage(alignment=16, auto_sync=True)
        sorted_keys = module.merge_sort_keys(
            group,
            keys,
            valid_items=valid_items,
            oob_default=1000,
            temp_storage=scratch,
        )
        pair_keys, pair_values = module.merge_sort_pairs(
            group,
            keys,
            values,
            descending=True,
            valid_items=valid_items,
            oob_default=-1000,
            temp_storage=scratch,
        )
        module.store(
            group,
            ascending,
            sorted_keys,
            valid_items=valid_items,
        )
        module.store(
            group,
            descending,
            pair_keys,
            valid_items=valid_items,
        )
        module.store(
            group,
            sorted_values,
            pair_values,
            valid_items=valid_items,
        )
        module.store(
            group,
            original_keys,
            keys,
            valid_items=valid_items,
        )
        module.store(
            group,
            original_values,
            values,
            valid_items=valid_items,
        )

    @cute.jit
    def launch(
        source_keys: cute.Pointer,
        source_values: cute.Pointer,
        ascending: cute.Pointer,
        descending: cute.Pointer,
        sorted_values: cute.Pointer,
        original_keys: cute.Pointer,
        original_values: cute.Pointer,
        items_per_thread: cutlass.Constexpr,
    ):
        sort_tile(
            source_keys,
            source_values,
            ascending,
            descending,
            sorted_values,
            original_keys,
            original_values,
            items_per_thread,
        ).launch(grid=1, block=_BLOCK)

    # docs: end cutlass-merge-sort

    rng = np.random.default_rng(42)
    source_keys = rng.integers(-31, 32, size=valid_items, dtype=np.int32)
    source_values = np.arange(valid_items, dtype=np.float64) + 0.25
    outputs = [
        np.full(tile_size, -999, dtype=dtype)
        for dtype in (np.int32, np.int32, np.float64, np.int32, np.float64)
    ]
    arrays = [source_keys, source_values, *outputs]
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
                cutlass.Int32 if array.dtype == np.int32 else cutlass.Float64
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
        for array, allocation in zip(outputs, allocations[2:]):
            _check(
                driver.cuMemcpyDtoH(array.ctypes.data, allocation, array.nbytes)
            )

    ascending, descending, sorted_values, original_keys, original_values = (
        outputs
    )
    expected_keys = np.sort(source_keys)
    np.testing.assert_array_equal(ascending[:valid_items], expected_keys)
    np.testing.assert_array_equal(descending[:valid_items], expected_keys[::-1])
    # Equal keys have no stability guarantee; compare complete key/value pairs.
    assert sorted(
        zip(
            descending[:valid_items],
            sorted_values[:valid_items],
        )
    ) == sorted(zip(source_keys, source_values))
    np.testing.assert_array_equal(original_keys[:valid_items], source_keys)
    np.testing.assert_array_equal(original_values[:valid_items], source_values)
    for output in outputs:
        np.testing.assert_array_equal(output[valid_items:], -999)
    return outputs


if __name__ == "__main__":
    run_example()
