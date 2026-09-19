# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Sort signed keys and stably rank a selected digit in a block tile."""

from contextlib import ExitStack

import cutlass
import numpy as np
from cutlass import cute
from cutlass.cute.runtime import make_ptr

from cuda.bindings import driver

_BLOCK = (8, 4, 2)
_BINS = 16


def _check(result):
    """Raise on Driver errors and unwrap a single returned handle or value."""

    if int(result[0]):
        raise RuntimeError(f"CUDA Driver call failed: {result[0]}")
    return result[1] if len(result) == 2 else result[1:]


def run_example(api="common", items_per_thread=2):
    """Check full-key Sort, stable signed-digit pairs, ranks, and prefixes.

    The host flips the sign bit before extracting the high digit, then uses a
    stable descending order to track ties by their original positions. Rank
    must return that permutation's inverse at the original input positions.
    The qualified path stores the striped pair-sort results with a striped
    Store and counts greater digits for each bin prefix. Both paths check
    that the input keys are unchanged.
    """
    tile_size = 64 * items_per_thread
    if api == "common":
        from cuda import coop
    elif api == "qualified":
        import cuda.coop.cutlass as coop
    else:
        raise ValueError("api must be 'common' or 'qualified'")

    # docs: start cutlass-radix
    @cute.kernel
    def order_tile(
        source: cute.Pointer,
        full_keys: cute.Pointer,
        digit_keys: cute.Pointer,
        digit_positions: cute.Pointer,
        ranks_out: cute.Pointer,
        prefixes_out: cute.Pointer,
        original_keys: cute.Pointer,
        items_per_thread: cutlass.Constexpr,
    ):
        block = coop.this_block()
        keys = coop.ThreadData(items_per_thread)
        positions = coop.ThreadData(items_per_thread)
        coop.load(block, source, keys)
        for item in cutlass.range_constexpr(items_per_thread):
            positions[item] = (
                cutlass.Int32(block.rank()) * items_per_thread + item
            )
        scratch = coop.TempStorage(alignment=32, auto_sync=True)
        ordered = coop.radix_sort_keys(block, keys, temp_storage=scratch)
        coop.store(block, full_keys, ordered)
        if cutlass.const_expr(api == "qualified"):
            pair_keys, pair_positions = coop.radix_sort_pairs(
                block,
                keys,
                positions,
                begin_bit=28,
                end_bit=32,
                descending=True,
                temp_storage=scratch,
                blocked_to_striped=True,
            )
            # Match the Store layout to the striped result registers.
            coop.store(block, digit_keys, pair_keys, algorithm="striped")
            coop.store(
                block, digit_positions, pair_positions, algorithm="striped"
            )
            prefixes = coop.ThreadData(items_per_thread=1)
            ranks = coop.radix_rank_keys(
                block,
                keys,
                begin_bit=28,
                radix_bits=4,
                descending=True,
                exclusive_digit_prefix=prefixes,
            )
            # Only the first 16 bin slots are defined for this four-bit digit.
            coop.store(block, prefixes_out, prefixes, valid_items=_BINS)
        else:
            pair_keys, pair_positions = coop.radix_sort_pairs(
                block,
                keys,
                positions,
                begin_bit=28,
                end_bit=32,
                descending=True,
                temp_storage=scratch,
            )
            coop.store(block, digit_keys, pair_keys)
            coop.store(block, digit_positions, pair_positions)
            ranks = coop.radix_rank_keys(
                block,
                keys,
                begin_bit=28,
                radix_bits=4,
                descending=True,
            )
        coop.store(block, ranks_out, ranks)
        coop.store(block, original_keys, keys)

    @cute.jit
    def launch(
        source: cute.Pointer,
        full_keys: cute.Pointer,
        digit_keys: cute.Pointer,
        digit_positions: cute.Pointer,
        ranks_out: cute.Pointer,
        prefixes_out: cute.Pointer,
        original_keys: cute.Pointer,
        items_per_thread: cutlass.Constexpr,
    ):
        order_tile(
            source,
            full_keys,
            digit_keys,
            digit_positions,
            ranks_out,
            prefixes_out,
            original_keys,
            items_per_thread,
        ).launch(grid=1, block=_BLOCK)

    # docs: end cutlass-radix

    source = (
        np.random.default_rng(42)
        .integers(0, 1 << 32, size=tile_size, dtype=np.uint32)
        .view(np.int32)
    )
    outputs = [np.full(tile_size, -999, dtype=np.int32) for _ in range(6)]
    outputs[4] = np.full(_BINS, -999, dtype=np.int32)
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
            pointers.append(
                make_ptr(
                    cutlass.Int32,
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

    full_keys, digit_keys, positions, ranks, prefixes, original_keys = outputs
    # Signed radix keys invert the sign bit before selecting the digit.
    digit = (source.view(np.uint32) ^ np.uint32(1 << 31)) >> np.uint32(28)
    order = np.argsort(-digit.astype(np.int32), kind="stable")
    expected_ranks = np.empty(tile_size, dtype=np.int32)
    expected_ranks[order] = np.arange(tile_size, dtype=np.int32)
    np.testing.assert_array_equal(full_keys, np.sort(source))
    np.testing.assert_array_equal(digit_keys, source[order])
    np.testing.assert_array_equal(positions, order)
    np.testing.assert_array_equal(ranks, expected_ranks)
    np.testing.assert_array_equal(original_keys, source)
    if api == "qualified":
        expected_prefixes = np.array(
            [np.count_nonzero(digit > bin_index) for bin_index in range(_BINS)],
            dtype=np.int32,
        )
        np.testing.assert_array_equal(prefixes, expected_prefixes)
    else:
        np.testing.assert_array_equal(prefixes, -999)
    return outputs


if __name__ == "__main__":
    run_example()
