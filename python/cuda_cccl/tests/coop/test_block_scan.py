# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
test_block_scan.py

This file contains unit tests for cuda.coop.block_scan. It covers both
valid and invalid usage scenarios, tests sum-based scans, user-defined operators
and types, prefix callback operators, and known operators such as min, max, and
bitwise XOR.
"""

import textwrap
from dataclasses import dataclass
from functools import reduce
from operator import mul

import numba
import numpy as np
import pytest
from helpers import (
    NUMBA_TYPES_TO_NP,
    Complex,
    complex_type,
    random_int,
    row_major_tid,
)
from numba import cuda, types
from numba.core import cgutils
from numba.core.extending import (
    lower_builtin,
    make_attribute_wrapper,
    models,
    register_model,
    type_callable,
    typeof_impl,
)

from cuda import coop
from cuda.coop import BlockLoadAlgorithm, BlockScanAlgorithm, BlockStoreAlgorithm
from cuda.coop.block._block_scan import (
    ScanOp,
)

numba.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0


class BlockPrefixCallbackOp:
    """
    A sample prefix callback operator that stores and updates a running total.
    """

    def __init__(self, running_total):
        self.running_total = running_total

    def __call__(self_ptr, block_aggregate):
        old_prefix = self_ptr[0].running_total
        self_ptr[0] = BlockPrefixCallbackOp(old_prefix + block_aggregate)
        return old_prefix


class BlockPrefixCallbackOpType(types.Type):
    def __init__(self):
        super().__init__(name="BlockPrefixCallbackOp")


block_prefix_callback_op_type = BlockPrefixCallbackOpType()


@typeof_impl.register(BlockPrefixCallbackOp)
def typeof_block_prefix_callback_op(val, c):
    return block_prefix_callback_op_type


@type_callable(BlockPrefixCallbackOp)
def type__block_prefix_callback_op(context):
    def typer(running_total):
        if isinstance(running_total, types.Integer):
            return block_prefix_callback_op_type

    return typer


@register_model(BlockPrefixCallbackOpType)
class BlockPrefixCallbackOpModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [("running_total", types.int64)]
        models.StructModel.__init__(self, dmm, fe_type, members)


make_attribute_wrapper(BlockPrefixCallbackOpType, "running_total", "running_total")


@lower_builtin(BlockPrefixCallbackOp, types.Integer)
def impl_block_prefix_callback_op(context, builder, sig, args):
    typ = sig.return_type
    [running_total] = args
    state = cgutils.create_struct_proxy(typ)(context, builder)
    state.running_total = running_total
    return state._getvalue()


@pytest.mark.parametrize("T", [types.uint32])
@pytest.mark.parametrize("threads_per_block", [32, (4, 16), (4, 8, 8)])
@pytest.mark.parametrize("items_per_thread", [1, 4])
@pytest.mark.parametrize("mode", ["inclusive", "exclusive"])
@pytest.mark.parametrize("algorithm", [BlockScanAlgorithm.RAKING])
def test_block_sum(T, threads_per_block, items_per_thread, mode, algorithm):
    """
    Tests block-wide sums with either inclusive or exclusive scans.
    Checks correctness of results and verifies no device memory ops
    occur in generated SASS.
    """
    num_threads = (
        threads_per_block
        if isinstance(threads_per_block, int)
        else reduce(mul, threads_per_block)
    )

    # Avoid resource issues in some configurations for raking_memoize.
    if algorithm == BlockScanAlgorithm.RAKING_MEMOIZE and num_threads >= 512:
        pytest.skip("raking_memoize can exceed resources for >= 512 threads.")

    @cuda.jit
    def kernel(input_arr, output_arr, items_per_thread):
        tid = row_major_tid()
        thread_in = coop.local.array(items_per_thread, dtype=T)
        thread_out = coop.local.array(items_per_thread, dtype=T)

        for i in range(items_per_thread):
            thread_in[i] = input_arr[tid * items_per_thread + i]

        coop.block.scan(
            thread_in,
            thread_out,
            items_per_thread,
            mode=mode,
            scan_op="+",
            algorithm=algorithm,
        )

        for i in range(items_per_thread):
            output_arr[tid * items_per_thread + i] = thread_out[i]

    dtype_np = NUMBA_TYPES_TO_NP[T]
    items_per_tile = num_threads * items_per_thread
    h_input = random_int(items_per_tile, dtype_np)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(items_per_tile, dtype=dtype_np)

    kernel[1, threads_per_block](d_input, d_output, items_per_thread)
    cuda.synchronize()

    output = d_output.copy_to_host()
    if mode == "inclusive":
        reference = np.cumsum(h_input)
    else:
        reference = np.cumsum(h_input) - h_input

    np.testing.assert_array_equal(output, reference)

    sig = (T[::1], T[::1], types.int64)
    sass = kernel.inspect_sass(sig)
    # Check that no device memory loads/stores appear in SASS.
    assert "LDL" not in sass
    assert "STL" not in sass


def test_block_scan_scalar_return():
    threads_per_block = 128
    dtype = np.int32

    @cuda.jit
    def kernel(d_in, d_out_exclusive, d_out_inclusive):
        tid = row_major_tid()
        value = d_in[tid]
        out_exclusive = coop.block.scan(value, mode="exclusive", scan_op="+")
        out_inclusive = coop.block.scan(value, mode="inclusive", scan_op="+")
        d_out_exclusive[tid] = out_exclusive
        d_out_inclusive[tid] = out_inclusive

    h_input = np.random.randint(0, 64, threads_per_block, dtype=dtype)
    d_input = cuda.to_device(h_input)
    d_out_exclusive = cuda.device_array_like(d_input)
    d_out_inclusive = cuda.device_array_like(d_input)

    kernel[1, threads_per_block](d_input, d_out_exclusive, d_out_inclusive)
    h_out_exclusive = d_out_exclusive.copy_to_host()
    h_out_inclusive = d_out_inclusive.copy_to_host()

    reference_inclusive = np.cumsum(h_input)
    reference_exclusive = reference_inclusive - h_input

    np.testing.assert_array_equal(h_out_exclusive, reference_exclusive)
    np.testing.assert_array_equal(h_out_inclusive, reference_inclusive)


def test_block_scan_scalar_block_aggregate():
    threads_per_block = 64

    @cuda.jit
    def kernel(output, aggregates):
        tid = cuda.threadIdx.x
        value = numba.int32(tid + 1)
        out_value = cuda.local.array(1, numba.int32)
        block_aggregate = cuda.local.array(1, numba.int32)
        out_value[0] = value
        coop.block.scan(
            out_value,
            out_value,
            items_per_thread=1,
            mode="exclusive",
            scan_op="+",
            block_aggregate=block_aggregate,
        )
        output[tid] = out_value[0]
        aggregates[tid] = block_aggregate[0]

    d_output = cuda.device_array(threads_per_block, dtype=np.int32)
    d_aggregates = cuda.device_array(threads_per_block, dtype=np.int32)
    kernel[1, threads_per_block](d_output, d_aggregates)
    h_output = d_output.copy_to_host()
    h_aggregates = d_aggregates.copy_to_host()

    expected_aggregate = (threads_per_block * (threads_per_block + 1)) // 2
    expected_exclusive = np.arange(threads_per_block, dtype=np.int32)
    expected_exclusive = expected_exclusive * (expected_exclusive + 1) // 2

    np.testing.assert_array_equal(h_output, expected_exclusive)
    np.testing.assert_array_equal(
        h_aggregates, np.full(threads_per_block, expected_aggregate, dtype=np.int32)
    )


def test_block_scan_array_block_aggregate():
    threads_per_block = 64
    items_per_thread = 2
    total_items = threads_per_block * items_per_thread

    @cuda.jit
    def kernel(output, aggregates):
        tid = cuda.threadIdx.x
        items = cuda.local.array(items_per_thread, numba.int32)
        out_items = cuda.local.array(items_per_thread, numba.int32)
        block_aggregate = cuda.local.array(1, numba.int32)

        for i in range(items_per_thread):
            items[i] = 1

        coop.block.scan(
            items,
            out_items,
            items_per_thread=items_per_thread,
            mode="exclusive",
            scan_op="+",
            block_aggregate=block_aggregate,
        )

        for i in range(items_per_thread):
            output[tid * items_per_thread + i] = out_items[i]
        aggregates[tid] = block_aggregate[0]

    d_output = cuda.device_array(total_items, dtype=np.int32)
    d_aggregates = cuda.device_array(threads_per_block, dtype=np.int32)
    kernel[1, threads_per_block](d_output, d_aggregates)
    h_output = d_output.copy_to_host()
    h_aggregates = d_aggregates.copy_to_host()

    expected_aggregate = total_items
    expected_output = np.arange(total_items, dtype=np.int32)

    np.testing.assert_array_equal(h_output, expected_output)
    np.testing.assert_array_equal(
        h_aggregates, np.full(threads_per_block, expected_aggregate, dtype=np.int32)
    )


@pytest.mark.parametrize("threads_per_block", [32, (4, 16), (4, 8, 8)])
@pytest.mark.parametrize("items_per_thread", [1, 4])
@pytest.mark.parametrize("mode", ["inclusive", "exclusive"])
@pytest.mark.parametrize("algorithm", [BlockScanAlgorithm.RAKING])
def test_block_sum_prefix_op(threads_per_block, items_per_thread, mode, algorithm):
    """
    Tests block-wide sums with a user-supplied prefix callback operator.
    Each tile of data is scanned and the prefix operator updates a running
    total for each tile within a segment.
    """
    num_threads = (
        threads_per_block
        if isinstance(threads_per_block, int)
        else reduce(mul, threads_per_block)
    )

    if algorithm == "raking_memoize" and num_threads >= 512:
        pytest.skip("raking_memoize can exceed resources for >= 512 threads.")

    segment_size = 2 * 1024
    num_segments = 128
    num_elements = segment_size * num_segments

    @cuda.jit
    def kernel(input_arr, output_arr, items_per_thread):
        segment_offset = cuda.blockIdx.x * segment_size
        block_prefix_op = coop.local.array(1, dtype=block_prefix_callback_op_type)
        block_prefix_op[0] = BlockPrefixCallbackOp(0)
        thread_in = coop.local.array(items_per_thread, dtype=numba.int32)
        thread_out = coop.local.array(items_per_thread, dtype=numba.int32)

        tid = row_major_tid()
        threads_per_block = cuda.blockDim.x * cuda.blockDim.y * cuda.blockDim.z
        tile_items = threads_per_block * items_per_thread
        tile_offset = 0

        while tile_offset < segment_size:
            for item in range(items_per_thread):
                item_offset = tile_offset + tid * items_per_thread + item
                if item_offset < segment_size:
                    thread_in[item] = input_arr[segment_offset + item_offset]
                else:
                    thread_in[item] = 0

            coop.block.scan(
                thread_in,
                thread_out,
                items_per_thread,
                mode=mode,
                scan_op="+",
                block_prefix_callback_op=block_prefix_op,
                algorithm=algorithm,
            )

            for item in range(items_per_thread):
                item_offset = tile_offset + tid * items_per_thread + item
                if item_offset < segment_size:
                    output_arr[segment_offset + item_offset] = thread_out[item]

            tile_offset += tile_items

    h_input = np.arange(num_elements, dtype=np.int32)
    d_input = cuda.to_device(h_input)
    d_output = cuda.to_device(np.zeros(num_elements, dtype=np.int32))

    kernel[num_segments, threads_per_block](d_input, d_output, items_per_thread)
    cuda.synchronize()

    h_output = d_output.copy_to_host()
    ref = np.zeros_like(h_input)

    # Build the reference result for each segment.
    if mode == "inclusive":
        for seg_id in range(num_segments):
            seg_start = seg_id * segment_size
            for i in range(segment_size):
                if i == 0:
                    ref[seg_start + i] = h_input[seg_start + i]
                else:
                    ref[seg_start + i] = ref[seg_start + i - 1] + h_input[seg_start + i]
    else:
        for seg_id in range(num_segments):
            seg_start = seg_id * segment_size
            ref[seg_start] = 0
            for i in range(1, segment_size):
                ref[seg_start + i] = ref[seg_start + i - 1] + h_input[seg_start + i - 1]

    np.testing.assert_array_equal(h_output, ref)

    sig = (types.int32[::1], types.int32[::1], types.int64)
    sass = kernel.inspect_sass(sig)
    assert "LDL" not in sass
    assert "STL" not in sass


@pytest.mark.parametrize("threads_per_block", [32, (8, 16), (2, 4, 8)])
@pytest.mark.parametrize("items_per_thread", [0, -1, -127])
@pytest.mark.parametrize("mode", ["inclusive", "exclusive"])
def test_block_scan_sum_invalid_items_per_thread(
    threads_per_block, items_per_thread, mode
):
    """
    Tests that invalid items_per_thread (< 1) raises a ValueError for both
    inclusive_sum and exclusive_sum.
    """
    if mode == "inclusive":
        sum_func = coop.block.inclusive_sum
    else:
        sum_func = coop.block.exclusive_sum

    with pytest.raises(ValueError):
        sum_func(numba.int32, threads_per_block, items_per_thread)


@pytest.mark.parametrize("mode", ["inclusive", "exclusive"])
def test_block_scan_sum_invalid_algorithm(mode):
    """
    Tests that supplying an unsupported algorithm to the sum-based scans
    raises a ValueError.
    """
    if mode == "inclusive":
        sum_func = coop.block.inclusive_sum
    else:
        sum_func = coop.block.exclusive_sum

    with pytest.raises(ValueError):
        sum_func(numba.int32, 128, items_per_thread=1, algorithm="invalid_algorithm")


@pytest.mark.parametrize("initial_value", [None, Complex(0, 0), Complex(1, 1)])
@pytest.mark.parametrize("threads_per_block", [32, (4, 16), (4, 8, 8)])
@pytest.mark.parametrize("items_per_thread", [1, 4])
@pytest.mark.parametrize("mode", ["inclusive", "exclusive"])
@pytest.mark.parametrize("algorithm", [BlockScanAlgorithm.RAKING])
def test_block_scan_user_defined_type(
    initial_value, items_per_thread, threads_per_block, mode, algorithm
):
    """
    Tests block-wide scans for a user-defined (Complex) type. Uses an addition
    operator to sum real and imaginary parts respectively.
    """
    if items_per_thread > 1 and initial_value is None:
        pytest.skip("initial_value is required for items_per_thread > 1")

    num_threads = (
        threads_per_block
        if isinstance(threads_per_block, int)
        else reduce(mul, threads_per_block)
    )
    num_elements = num_threads * items_per_thread

    if algorithm == "raking_memoize" and num_threads >= 512:
        pytest.skip("raking_memoize can exceed resources for >= 512 threads.")

    # Our custom operator (add complex).
    @cuda.jit(device=True)
    def op(result_ptr, lhs_ptr, rhs_ptr):
        # We need the explicit cast to prevent automatic up-casting to i64.
        real_val = numba.int32(lhs_ptr[0].real + rhs_ptr[0].real)
        imag_val = numba.int32(lhs_ptr[0].imag + rhs_ptr[0].imag)
        result_ptr[0] = Complex(real_val, imag_val)

    if mode == "inclusive":
        if items_per_thread == 1 and initial_value is not None:
            pytest.skip(
                "initial_value not supported for inclusive "
                "scans with items_per_thread=1"
            )
    else:
        if initial_value is not None:
            pytest.skip("initial_value not supported for exclusive scans")

    # N.B. I had to use two separate kernels here, because having a single
    #      kernel with `if initial_value is not None` did not yield a kernel
    #      that Numba could compile.
    if initial_value is not None:

        @cuda.jit
        def kernel(input_arr, output_arr, items_per_thread):
            tid = row_major_tid()
            thread_in = coop.local.array(items_per_thread, dtype=complex_type)
            thread_out = coop.local.array(items_per_thread, dtype=complex_type)

            real_idx_base = tid * items_per_thread
            complex_idx_base = num_elements + tid * items_per_thread
            for i in range(items_per_thread):
                thread_in[i] = Complex(
                    input_arr[real_idx_base + i],
                    input_arr[complex_idx_base + i],
                )

            coop.block.scan(
                thread_in,
                thread_out,
                items_per_thread,
                mode=mode,
                scan_op=op,
                initial_value=initial_value,
                algorithm=algorithm,
            )

            for i in range(items_per_thread):
                output_arr[real_idx_base + i] = thread_out[i].real
                output_arr[complex_idx_base + i] = thread_out[i].imag

    else:

        @cuda.jit
        def kernel(input_arr, output_arr, items_per_thread):
            tid = row_major_tid()
            thread_in = coop.local.array(items_per_thread, dtype=complex_type)
            thread_out = coop.local.array(items_per_thread, dtype=complex_type)

            real_idx_base = tid * items_per_thread
            complex_idx_base = num_elements + tid * items_per_thread
            for i in range(items_per_thread):
                thread_in[i] = Complex(
                    input_arr[real_idx_base + i],
                    input_arr[complex_idx_base + i],
                )

            coop.block.scan(
                thread_in,
                thread_out,
                items_per_thread,
                mode=mode,
                scan_op=op,
                algorithm=algorithm,
            )

            for i in range(items_per_thread):
                output_arr[real_idx_base + i] = thread_out[i].real
                output_arr[complex_idx_base + i] = thread_out[i].imag

    # Account for a Complex type containing two int32 values.
    total_items = num_threads * items_per_thread * 2
    h_input = random_int(total_items, "int32")
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(total_items, dtype=np.int32)
    kernel[1, threads_per_block](d_input, d_output, items_per_thread)
    cuda.synchronize()

    h_output = d_output.copy_to_host()
    real_vals = h_input[:num_elements]
    imag_vals = h_input[num_elements:]

    if mode == "inclusive":
        real_ref = np.cumsum(real_vals)
        imag_ref = np.cumsum(imag_vals)

        if initial_value is not None:
            real_ref = real_ref + initial_value.real
            imag_ref = imag_ref + initial_value.imag
    else:
        real_ref = np.zeros_like(real_vals)
        imag_ref = np.zeros_like(imag_vals)

        if len(real_vals) > 1:
            real_ref[1:] = np.cumsum(real_vals)[:-1]
            imag_ref[1:] = np.cumsum(imag_vals)[:-1]

        if initial_value is None:
            real_ref[0] = h_output[0]
            imag_ref[0] = h_output[num_threads]

    np.testing.assert_array_equal(h_output[:num_elements], real_ref)
    np.testing.assert_array_equal(h_output[num_elements:], imag_ref)

    sig = (numba.int32[::1], numba.int32[::1], types.int64)
    sass = kernel.inspect_sass(sig)
    assert "LDL" not in sass
    assert "STL" not in sass


@pytest.mark.parametrize("T", [types.uint32, types.float64])
@pytest.mark.parametrize("threads_per_block", [32, (4, 16), (4, 8, 8)])
@pytest.mark.parametrize("items_per_thread", [1, 4])
@pytest.mark.parametrize("mode", ["inclusive", "exclusive"])
@pytest.mark.parametrize("algorithm", [BlockScanAlgorithm.RAKING])
def test_block_scan_with_callable(
    T, threads_per_block, items_per_thread, mode, algorithm
):
    """
    Tests block-wide scans with a user-supplied Python callable as the scan
    operator. Verifies correctness for inclusive and exclusive scans.
    """
    num_threads = (
        threads_per_block
        if isinstance(threads_per_block, int)
        else reduce(mul, threads_per_block)
    )

    if algorithm == "raking_memoize" and num_threads >= 512:
        pytest.skip("raking_memoize can exceed resources for >= 512 threads.")

    # Example custom operator that just adds two operands.
    @cuda.jit(device=True, inline="always", forceinline=True)
    def op(a: T, b: T) -> T:
        return a + b

    @cuda.jit
    def kernel(input_arr, output_arr, items_per_thread):
        # Get the correct thread ID based on thread configuration
        tid = row_major_tid()
        thread_in = coop.local.array(items_per_thread, dtype=T)
        thread_out = coop.local.array(items_per_thread, dtype=T)

        # Load the input data
        for i in range(items_per_thread):
            thread_in[i] = input_arr[tid * items_per_thread + i]

        coop.block.scan(
            thread_in,
            thread_out,
            items_per_thread,
            mode=mode,
            scan_op=op,
            algorithm=algorithm,
        )

        # Store the output data
        for i in range(items_per_thread):
            output_arr[tid * items_per_thread + i] = thread_out[i]

    dtype_np = NUMBA_TYPES_TO_NP[T]
    total_items = num_threads * items_per_thread
    h_input = random_int(total_items, dtype_np)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(total_items, dtype=dtype_np)

    kernel[1, threads_per_block](d_input, d_output, items_per_thread)
    cuda.synchronize()

    output = d_output.copy_to_host()

    # Calculate reference result
    if mode == "inclusive":
        ref = np.cumsum(h_input)
    else:
        # For exclusive scan:
        # - First thread gets first value directly (CUB behavior)
        # - Other elements are shifted by 1
        ref = np.zeros_like(h_input)
        # Copy the first value from the actual output
        # since CUB's behavior for first element can vary
        ref[0] = output[0]
        # For all other elements, it's the cumulative sum up to the previous element
        if len(h_input) > 1:
            ref[1:] = np.cumsum(h_input[:-1])

    # If T is an integer type, use assert_array_equal.  Otherwise, if
    # floating point, use assert_array_almost_equal.
    if isinstance(T, types.Integer):
        np.testing.assert_array_equal(output, ref)
    else:
        np.testing.assert_array_almost_equal(output, ref)

    sig = (T[::1], T[::1], types.int64)
    sass = kernel.inspect_sass(sig)
    if not (T == types.float64 and items_per_thread > 1):
        assert "LDL" not in sass
        assert "STL" not in sass


@pytest.mark.parametrize("mode", ["inclusive", "exclusive"])
def test_block_scan_invariants(mode):
    """
    Tests invariants:
      1) Initial value unsupported for inclusive scans with 1 item/thread.
      2) Initial value unsupported for exclusive scans with 1 item/thread and
         a block prefix callback.
      3) When items_per_thread > 1 and no prefix callback is supplied, an
         initial value is required.
    """
    if mode == "inclusive":
        scan_func = coop.block.inclusive_scan
    else:
        scan_func = coop.block.exclusive_scan

    # 1) For inclusive scans with items_per_thread=1, initial_value is invalid.
    if mode == "inclusive":
        with pytest.raises(
            ValueError,
            match=(
                "initial_value is not supported for inclusive scans "
                "with items_per_thread == 1"
            ),
        ):
            scan_func(
                dtype=numba.int32,
                threads_per_block=128,
                scan_op="*",
                items_per_thread=1,
                initial_value=0,
            )

    # 2) For exclusive scans with items_per_thread=1 and a prefix callback,
    #    initial_value is invalid.
    if mode == "exclusive":
        prefix_op = coop.StatefulFunction(
            BlockPrefixCallbackOp,
            block_prefix_callback_op_type,
            name="block_prefix_callback_op",
        )
        with pytest.raises(
            ValueError,
            match=(
                "initial_value is not supported for exclusive scans "
                "with items_per_thread == 1 and a block prefix "
                "callback operator"
            ),
        ):
            scan_func(
                dtype=numba.int32,
                threads_per_block=128,
                scan_op="*",
                items_per_thread=1,
                initial_value=0,
                prefix_op=prefix_op,
            )

    # 3) For items_per_thread>1 and no prefix callback, initial_value is
    #    required.  We use `complex_type` here instead of a simpler type
    #    (like an int32), because the latter will be auto-defaulted to a
    #    value of 0.
    with pytest.raises(
        ValueError,
        match=(
            "initial_value is required for both inclusive and exclusive "
            "scans when items_per_thread > 1 and no block prefix callback "
            "operator has been supplied"
        ),
    ):
        scan_func(
            dtype=complex_type,
            threads_per_block=128,
            scan_op="*",
            items_per_thread=2,
        )


@pytest.mark.parametrize("T", [types.int32])
@pytest.mark.parametrize("threads_per_block", [32, 64])
@pytest.mark.parametrize("items_per_thread", [2, 3])
@pytest.mark.parametrize("initial_value", [-1, 0, 100])
@pytest.mark.parametrize("mode", ["inclusive", "exclusive"])
def test_block_scan_with_prefix_op_multi_items(
    T, threads_per_block, items_per_thread, initial_value, mode
):
    """
    Tests scans with a prefix callback operator in a multi-items-per-thread
    setup. The prefix callback operator is initialized with 'initial_value'
    which is applied as the starting prefix for the entire block.
    """
    num_threads = (
        threads_per_block
        if isinstance(threads_per_block, int)
        else reduce(mul, threads_per_block)
    )

    @cuda.jit(device=True)
    def add_op(a, b):
        return a + b

    @cuda.jit
    def kernel(input_arr, output_arr, items_per_thread):
        tid = row_major_tid()
        thread_data = coop.local.array(items_per_thread, dtype=T)
        for i in range(items_per_thread):
            thread_data[i] = input_arr[tid * items_per_thread + i]

        # Initialize the prefix callback operator with 'initial_value'.
        block_prefix_op = coop.local.array(1, dtype=block_prefix_callback_op_type)
        block_prefix_op[0] = BlockPrefixCallbackOp(initial_value)

        coop.block.scan(
            thread_data,
            thread_data,
            items_per_thread,
            mode=mode,
            scan_op=add_op,
            block_prefix_callback_op=block_prefix_op,
            algorithm=BlockScanAlgorithm.RAKING,
        )

        for i in range(items_per_thread):
            output_arr[tid * items_per_thread + i] = thread_data[i]

    dtype_np = NUMBA_TYPES_TO_NP[T]
    total_items = num_threads * items_per_thread
    h_input = random_int(total_items, dtype_np)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(total_items, dtype=dtype_np)

    kernel[1, threads_per_block](d_input, d_output, items_per_thread)
    cuda.synchronize()

    output = d_output.copy_to_host()

    # Build reference with the prefix included.
    if mode == "inclusive":
        ref = np.zeros_like(h_input)
        running = initial_value
        for i in range(total_items):
            running = running + h_input[i]
            ref[i] = running
    else:
        ref = np.zeros_like(h_input)
        running = initial_value
        for i in range(total_items):
            ref[i] = running
            running = running + h_input[i]

    np.testing.assert_array_equal(output, ref)

    sig = (T[::1], T[::1], types.int64)
    sass = kernel.inspect_sass(sig)
    assert "LDL" not in sass
    assert "STL" not in sass


@pytest.mark.parametrize("mode", ["inclusive", "exclusive"])
@pytest.mark.parametrize(
    "scan_op", ["max", "min", "multiplies", "bit_and", "bit_or", "bit_xor"]
)
@pytest.mark.parametrize("T", [types.uint32])
@pytest.mark.parametrize("threads_per_block", [32, (4, 8, 8)])
@pytest.mark.parametrize("items_per_thread", [1, 4])
@pytest.mark.parametrize("algorithm", [BlockScanAlgorithm.RAKING])
def test_block_scan_known_ops(
    mode, scan_op, T, threads_per_block, items_per_thread, algorithm
):
    """
    Tests block-wide scans with known operators:
      - max
      - min
      - multiplies
      - bit_and
      - bit_or
      - bit_xor
    Verifies correctness against a Python-based reference for both
    inclusive and exclusive scans.
    """
    num_threads = (
        threads_per_block
        if isinstance(threads_per_block, int)
        else reduce(mul, threads_per_block)
    )

    op = ScanOp(scan_op)
    assert op.is_known

    @cuda.jit
    def kernel(input_arr, output_arr, items_per_thread):
        tid = row_major_tid()
        thread_in = coop.local.array(items_per_thread, dtype=T)
        thread_out = coop.local.array(items_per_thread, dtype=T)
        for i in range(items_per_thread):
            thread_in[i] = input_arr[tid * items_per_thread + i]

        coop.block.scan(
            thread_in,
            thread_out,
            items_per_thread,
            mode=mode,
            scan_op=scan_op,
            algorithm=algorithm,
        )

        for i in range(items_per_thread):
            output_arr[tid * items_per_thread + i] = thread_out[i]

    # Prepare host data.
    dtype_np = NUMBA_TYPES_TO_NP[T]
    total_items = num_threads * items_per_thread

    # Generate appropriate test data based on operation.
    if scan_op == "multiplies":
        # For multiplication, use very small values (1-2) to avoid overflow.
        h_input = np.ones(total_items, dtype=dtype_np)
        # Set a few values to 2 for a meaningful test (only ~10% of elements)
        rng = np.random.default_rng(42)
        indices = rng.choice(
            total_items, size=min(10, total_items // 10), replace=False
        )
        h_input[indices] = 2
    else:
        h_input = random_int(total_items, dtype_np)

    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(total_items, dtype=dtype_np)

    k = kernel[1, threads_per_block]
    k(d_input, d_output, items_per_thread)
    cuda.synchronize()

    output = d_output.copy_to_host()

    # Choose the appropriate operator based on scan_op.
    if scan_op == "multiplies":
        py_op = np.multiply
    elif scan_op == "bit_and":
        py_op = np.bitwise_and
    elif scan_op == "bit_or":
        py_op = np.bitwise_or
    elif scan_op == "bit_xor":
        py_op = np.bitwise_xor
    elif scan_op == "min":
        py_op = np.minimum
    elif scan_op == "max":
        py_op = np.maximum
    else:
        raise ValueError(f"Unexpected scan_op: {scan_op}")

    # Calculate reference results
    ref = np.zeros_like(h_input)
    if mode == "inclusive":
        if scan_op == "multiplies":
            # Use numpy's cumprod for inclusive scan with multiplication.
            ref = np.cumprod(h_input, dtype=dtype_np).astype(dtype_np)
        else:
            # For other operations, use our own loop-based implementation.
            accum = h_input[0]
            ref[0] = accum
            for i in range(1, total_items):
                accum = py_op(accum, h_input[i])
                ref[i] = accum
    else:
        # Initial value will default to 0 for exclusive scan when we provide
        # no alternate initial value.
        ref[0] = output[0]

        if scan_op == "multiplies":
            accum = np.array(h_input[0], dtype=dtype_np)
            for i in range(1, total_items):
                ref[i] = accum
                accum = py_op(accum, h_input[i])
        else:
            accum = h_input[0]
            for i in range(1, total_items):
                ref[i] = accum
                accum = py_op(accum, h_input[i])

    np.testing.assert_array_equal(output, ref)

    sig = (T[::1], T[::1], types.int64)
    sass = kernel.inspect_sass(sig)
    assert "LDL" not in sass
    assert "STL" not in sass


def test_inclusive_sum_alignment():
    block_scan1 = coop.block.inclusive_sum(
        dtype=types.int32,
        threads_per_block=256,
        items_per_thread=1,
    )

    block_scan2 = coop.block.inclusive_sum(
        dtype=types.float64,
        threads_per_block=256,
        items_per_thread=1,
    )

    assert block_scan1.temp_storage_alignment == 16
    assert block_scan2.temp_storage_alignment == 16


class BlockPrefixCallbackOpWithCount:
    """
    A sample prefix callback operator that stores and updates a running total.
    """

    def __init__(self, running_total, call_count):
        self.running_total = running_total
        self.call_count = call_count

    def __call__(self_ptr, block_aggregate):
        old_prefix = self_ptr[0].running_total
        call_count = self_ptr[0].call_count + 1
        new_running_total = old_prefix + block_aggregate
        self_ptr[0] = BlockPrefixCallbackOpWithCount(new_running_total, call_count)
        return old_prefix


class BlockPrefixCallbackOpWithCountType(types.Type):
    def __init__(self):
        super().__init__(name="BlockPrefixCallbackOpWithCount")


block_prefix_callback_op_with_count_type = BlockPrefixCallbackOpWithCountType()


@typeof_impl.register(BlockPrefixCallbackOpWithCount)
def typeof_block_prefix_callback_op_with_count(val, c):
    return block_prefix_callback_op_with_count_type


@type_callable(BlockPrefixCallbackOpWithCount)
def type__block_prefix_callback_op_with_count(context):
    def typer(running_total, call_count):
        return block_prefix_callback_op_with_count_type
        valid_args = isinstance(running_total, types.Integer) and isinstance(
            call_count, types.Integer
        )
        if valid_args:
            return block_prefix_callback_op_with_count_type

    return typer


@register_model(BlockPrefixCallbackOpWithCountType)
class BlockPrefixCallbackOpWithCountModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("running_total", types.int64),
            ("call_count", types.int64),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


make_attribute_wrapper(
    BlockPrefixCallbackOpWithCountType, "running_total", "running_total"
)
make_attribute_wrapper(BlockPrefixCallbackOpWithCountType, "call_count", "call_count")


# @lower_builtin(BlockPrefixCallbackOpWithCount, types.VarArg(types.Any))
@lower_builtin(BlockPrefixCallbackOpWithCount, types.Integer, types.Integer)
def impl_block_prefix_callback_op_with_count(context, builder, sig, args):
    typ = sig.return_type
    # llvtype = context.get_value_type(typ)
    # lldtype = context.get_data_type(typ)
    # abi_alignment = context.get_abi_alignment(lldtype)
    # itemsize = context.get_abi_sizeof(lldtype)
    [running_total, call_count] = args
    state = cgutils.create_struct_proxy(typ)(context, builder)
    state.running_total = running_total
    state.call_count = call_count
    return state._getvalue()


def test_block_load_store_scan_simple1():
    @cuda.jit
    def kernel(d_in, d_out, items_per_thread):
        threads_per_block = cuda.blockDim.x * cuda.blockDim.y * cuda.blockDim.z
        items_per_block = items_per_thread * threads_per_block

        block_offset = cuda.blockIdx.x * items_per_block
        # thread_offset = cuda.threadIdx.x * items_per_thread

        thread_data = coop.local.array(items_per_thread, dtype=d_in.dtype)

        # Load with padding
        coop.block.load(
            d_in[block_offset:],
            thread_data,
            items_per_thread=items_per_thread,
            algorithm=BlockLoadAlgorithm.WARP_TRANSPOSE,
        )

        coop.block.scan(
            thread_data,
            thread_data,
            items_per_thread,
        )

        coop.block.store(
            d_out[block_offset:],
            thread_data,
            items_per_thread=items_per_thread,
            algorithm=BlockStoreAlgorithm.DIRECT,
        )

    dtype = np.int32
    threads_per_block = 128
    num_total_items = threads_per_block * 4  # Total items to process
    items_per_thread = 4

    h_input = np.random.randint(0, 42, num_total_items, dtype=dtype)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array_like(d_input)

    items_per_block = threads_per_block * items_per_thread
    blocks_per_grid = (num_total_items + items_per_block - 1) // items_per_block

    kernel[blocks_per_grid, threads_per_block](d_input, d_output, items_per_thread)

    cuda.synchronize()

    # Compute the correct prefix-sum on host to compare results
    h_reference = np.zeros_like(h_input)
    if len(h_input) > 0:
        h_reference[0] = 0
        h_reference[1:] = np.cumsum(h_input[:-1])

    # h_reference2 = np.cumsum(h_input)
    # h_reference3 = np.cumsum(h_input) - h_input

    h_output = d_output.copy_to_host()
    np.testing.assert_array_equal(h_output, h_reference)


@pytest.mark.parametrize("items_per_thread", [1, 4, 8])
def test_block_load_store_scan_thread_data(items_per_thread):
    @cuda.jit
    def kernel(d_in, d_out, items_per_thread):
        thread_data = coop.ThreadData(items_per_thread)
        coop.block.load(d_in, thread_data)
        coop.block.scan(thread_data, thread_data)
        coop.block.store(d_out, thread_data)

    dtype = np.int32
    threads_per_block = 128
    num_total_items = threads_per_block * items_per_thread

    h_input = np.random.randint(0, 42, num_total_items, dtype=dtype)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array_like(d_input)

    kernel[1, threads_per_block](d_input, d_output, items_per_thread)
    cuda.synchronize()

    h_reference = np.zeros_like(h_input)
    if len(h_input) > 0:
        h_reference[0] = 0
        h_reference[1:] = np.cumsum(h_input[:-1])

    h_output = d_output.copy_to_host()
    np.testing.assert_array_equal(h_output, h_reference)


def test_block_load_store_scan_temp_storage_placeholder():
    dtype = np.int32
    threads_per_block = 128
    items_per_thread = 4

    block_scan = coop.block.scan(dtype, threads_per_block, items_per_thread)
    temp_storage_bytes = block_scan.temp_storage_bytes
    temp_storage_alignment = block_scan.temp_storage_alignment

    @cuda.jit
    def kernel(d_in, d_out):
        temp_storage = coop.TempStorage(
            temp_storage_bytes,
            temp_storage_alignment,
        )
        thread_data = coop.ThreadData(items_per_thread)
        coop.block.load(d_in, thread_data)
        coop.block.scan(thread_data, thread_data, temp_storage=temp_storage)
        coop.block.store(d_out, thread_data)

    num_total_items = threads_per_block * items_per_thread
    h_input = np.random.randint(0, 42, num_total_items, dtype=dtype)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array_like(d_input)

    kernel[1, threads_per_block](d_input, d_output)
    cuda.synchronize()

    h_reference = np.zeros_like(h_input)
    if len(h_input) > 0:
        h_reference[0] = 0
        h_reference[1:] = np.cumsum(h_input[:-1])

    h_output = d_output.copy_to_host()
    np.testing.assert_array_equal(h_output, h_reference)


def test_block_load_store_scan_two_phase_temp_storage():
    dtype = np.int32
    threads_per_block = 128
    items_per_thread = 1

    block_scan = coop.block.scan(dtype, threads_per_block, items_per_thread)
    temp_storage_bytes = block_scan.temp_storage_bytes
    temp_storage_alignment = block_scan.temp_storage_alignment

    @cuda.jit
    def kernel(d_in, d_out):
        temp_storage = coop.TempStorage(
            temp_storage_bytes,
            temp_storage_alignment,
        )
        thread_data = coop.ThreadData(items_per_thread)
        coop.block.load(d_in, thread_data)
        block_scan(thread_data, thread_data, temp_storage=temp_storage)
        coop.block.store(d_out, thread_data)

    num_total_items = threads_per_block * items_per_thread
    h_input = np.random.randint(0, 42, num_total_items, dtype=dtype)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array_like(d_input)

    kernel[1, threads_per_block](d_input, d_output)
    cuda.synchronize()

    h_reference = np.zeros_like(h_input)
    if len(h_input) > 0:
        h_reference[0] = 0
        h_reference[1:] = np.cumsum(h_input[:-1])

    h_output = d_output.copy_to_host()
    np.testing.assert_array_equal(h_output, h_reference)


def test_block_load_store_scan_gpu_dataclass_temp_storage():
    dtype = np.int32
    threads_per_block = 128
    items_per_thread = 1

    block_load = coop.block.load(
        dtype,
        threads_per_block,
        items_per_thread,
        algorithm=BlockLoadAlgorithm.DIRECT,
    )
    block_scan = coop.block.scan(dtype, threads_per_block, items_per_thread)
    block_store = coop.block.store(
        dtype,
        threads_per_block,
        items_per_thread,
        algorithm=BlockStoreAlgorithm.DIRECT,
    )

    @dataclass
    class KernelParams:
        items_per_thread: int
        block_load: coop.block.load
        block_scan: coop.block.scan
        block_store: coop.block.store

    kp = KernelParams(
        items_per_thread=items_per_thread,
        block_load=block_load,
        block_scan=block_scan,
        block_store=block_store,
    )
    kp = coop.gpu_dataclass(kp)

    temp_storage_bytes = kp.temp_storage_bytes_max
    temp_storage_alignment = kp.temp_storage_alignment
    expected_max = max(
        block_load.temp_storage_bytes,
        block_scan.temp_storage_bytes,
        block_store.temp_storage_bytes,
    )
    expected_alignment = max(
        block_load.temp_storage_alignment,
        block_scan.temp_storage_alignment,
        block_store.temp_storage_alignment,
    )
    assert temp_storage_bytes == expected_max
    assert temp_storage_alignment == expected_alignment

    @cuda.jit
    def kernel(d_in, d_out, kp):
        items_per_thread = kp.items_per_thread
        threads_per_block = cuda.blockDim.x * cuda.blockDim.y * cuda.blockDim.z
        items_per_block = items_per_thread * threads_per_block
        block_offset = cuda.blockIdx.x * items_per_block

        temp_storage = coop.TempStorage(
            temp_storage_bytes,
            temp_storage_alignment,
        )
        thread_data = coop.local.array(items_per_thread, dtype=d_in.dtype)
        kp.block_load(d_in[block_offset:], thread_data, temp_storage=temp_storage)
        kp.block_scan(thread_data, thread_data, temp_storage=temp_storage)
        kp.block_store(d_out[block_offset:], thread_data, temp_storage=temp_storage)

    num_total_items = threads_per_block * items_per_thread
    h_input = np.random.randint(0, 42, num_total_items, dtype=dtype)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array_like(d_input)

    kernel[1, threads_per_block](d_input, d_output, kp)
    cuda.synchronize()

    h_reference = np.zeros_like(h_input)
    if len(h_input) > 0:
        h_reference[0] = 0
        h_reference[1:] = np.cumsum(h_input[:-1])

    h_output = d_output.copy_to_host()
    np.testing.assert_array_equal(h_output, h_reference)


def test_block_load_store_scan_gpu_dataclass_temp_storage_bit_or():
    dtype = np.int32
    threads_per_block = 128
    items_per_thread = 4

    block_load = coop.block.load(
        dtype,
        threads_per_block,
        items_per_thread,
        algorithm=BlockLoadAlgorithm.DIRECT,
    )
    block_scan = coop.block.scan(
        dtype,
        threads_per_block,
        items_per_thread,
        scan_op="bit_or",
    )
    block_store = coop.block.store(
        dtype,
        threads_per_block,
        items_per_thread,
        algorithm=BlockStoreAlgorithm.DIRECT,
    )

    @dataclass
    class KernelParams:
        items_per_thread: int
        block_load: coop.block.load
        block_scan: coop.block.scan
        block_store: coop.block.store

    kp = KernelParams(
        items_per_thread=items_per_thread,
        block_load=block_load,
        block_scan=block_scan,
        block_store=block_store,
    )
    kp = coop.gpu_dataclass(kp)

    temp_storage_bytes = kp.temp_storage_bytes_max
    temp_storage_alignment = kp.temp_storage_alignment

    @cuda.jit
    def kernel(d_in, d_out, kp):
        items_per_thread = kp.items_per_thread
        threads_per_block = cuda.blockDim.x * cuda.blockDim.y * cuda.blockDim.z
        items_per_block = items_per_thread * threads_per_block
        block_offset = cuda.blockIdx.x * items_per_block

        temp_storage = coop.TempStorage(
            temp_storage_bytes,
            temp_storage_alignment,
        )
        thread_data = coop.local.array(items_per_thread, dtype=d_in.dtype)
        kp.block_load(d_in[block_offset:], thread_data, temp_storage=temp_storage)
        kp.block_scan(thread_data, thread_data, temp_storage=temp_storage)
        kp.block_store(d_out[block_offset:], thread_data, temp_storage=temp_storage)

    num_total_items = threads_per_block * items_per_thread
    h_input = np.random.randint(0, 16, num_total_items, dtype=dtype)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array_like(d_input)

    kernel[1, threads_per_block](d_input, d_output, kp)
    cuda.synchronize()

    h_reference = np.empty_like(h_input)
    running = np.int32(0)
    for i in range(len(h_input)):
        h_reference[i] = running
        running = np.int32(running | h_input[i])

    h_output = d_output.copy_to_host()
    np.testing.assert_array_equal(h_output, h_reference)


def test_block_load_store_scan_simple2():
    @cuda.jit
    def kernel(d_in, d_out, items_per_thread, num_total_items):
        threads_per_block = cuda.blockDim.x * cuda.blockDim.y * cuda.blockDim.z
        items_per_block = items_per_thread * threads_per_block

        block_offset = cuda.blockIdx.x * items_per_block
        thread_offset = cuda.threadIdx.x * items_per_thread

        thread_data = coop.local.array(items_per_thread, dtype=d_in.dtype)

        num_valid_items = min(
            items_per_block,
            num_total_items - block_offset,
        )

        # Load with padding
        coop.block.load(
            d_in[block_offset:],
            thread_data,
            items_per_thread=items_per_thread,
            algorithm=BlockLoadAlgorithm.WARP_TRANSPOSE,
            num_valid_items=num_valid_items,
        )

        # Zero-pad invalid thread items explicitly.
        for i in range(items_per_thread):
            global_idx = block_offset + thread_offset + i
            if global_idx >= num_total_items:
                thread_data[i] = 0

        # cuda.syncthreads()

        coop.block.scan(
            thread_data,
            thread_data,
            items_per_thread,
        )

        # Store only valid items back to global memory
        coop.block.store(
            d_out[block_offset:],
            thread_data,
            items_per_thread=items_per_thread,
            algorithm=BlockStoreAlgorithm.DIRECT,
            num_valid_items=num_valid_items,
        )

    dtype = np.int32
    threads_per_block = 128
    num_total_items = threads_per_block * 4  # Total items to process
    items_per_thread = 4

    h_input = np.random.randint(0, 42, num_total_items, dtype=dtype)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array_like(d_input)

    items_per_block = threads_per_block * items_per_thread
    blocks_per_grid = (num_total_items + items_per_block - 1) // items_per_block

    kernel[blocks_per_grid, threads_per_block](
        d_input, d_output, items_per_thread, num_total_items
    )

    cuda.synchronize()

    # Compute the correct prefix-sum on host to compare results
    h_reference = np.zeros_like(h_input)
    if len(h_input) > 0:
        h_reference[0] = 0
        h_reference[1:] = np.cumsum(h_input[:-1])

    # h_reference2 = np.cumsum(h_input)
    # h_reference3 = np.cumsum(h_input) - h_input

    h_output = d_output.copy_to_host()
    np.testing.assert_array_equal(h_output, h_reference)


def test_block_load_store_scan_simple3():
    @cuda.jit
    def kernel(d_in, d_out, items_per_thread, num_total_items):
        threads_per_block = cuda.blockDim.x * cuda.blockDim.y * cuda.blockDim.z
        items_per_block = items_per_thread * threads_per_block

        block_offset = cuda.blockIdx.x * items_per_block
        thread_offset = cuda.threadIdx.x * items_per_thread

        thread_data = coop.local.array(items_per_thread, dtype=d_in.dtype)

        num_valid_items = min(
            items_per_block,
            num_total_items - block_offset,
        )

        # Load with padding
        coop.block.load(
            d_in[block_offset:],
            thread_data,
            items_per_thread=items_per_thread,
            algorithm=BlockLoadAlgorithm.WARP_TRANSPOSE,
            num_valid_items=num_valid_items,
        )

        # Zero-pad invalid thread items explicitly.
        for i in range(items_per_thread):
            global_idx = block_offset + thread_offset + i
            if global_idx >= num_total_items:
                thread_data[i] = 0

        # cuda.syncthreads()

        coop.block.scan(
            thread_data,
            thread_data,
            items_per_thread,
        )

        # Store only valid items back to global memory
        coop.block.store(
            d_out[block_offset:],
            thread_data,
            items_per_thread=items_per_thread,
            algorithm=BlockStoreAlgorithm.DIRECT,
            num_valid_items=num_valid_items,
        )

    dtype = np.int32
    threads_per_block = 128
    num_total_items = threads_per_block * 4  # Total items to process
    items_per_thread = 4

    h_input = np.random.randint(0, 42, num_total_items, dtype=dtype)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array_like(d_input)

    items_per_block = threads_per_block * items_per_thread
    blocks_per_grid = (num_total_items + items_per_block - 1) // items_per_block

    kernel[blocks_per_grid, threads_per_block](
        d_input, d_output, items_per_thread, num_total_items
    )

    cuda.synchronize()

    # Compute the correct prefix-sum on host to compare results
    h_reference = np.zeros_like(h_input)
    if len(h_input) > 0:
        h_reference[0] = 0
        h_reference[1:] = np.cumsum(h_input[:-1])

    # h_reference2 = np.cumsum(h_input)
    # h_reference3 = np.cumsum(h_input) - h_input

    h_output = d_output.copy_to_host()
    np.testing.assert_array_equal(h_output, h_reference)


@pytest.mark.skip(reason="Experimental prefix-op CUSource scaffolding")
def test_block_scan_prefix_op_cusource_experimental():
    _ = cuda.CUSource(
        textwrap.dedent("""
        template<typename T>
        struct BlockPrefixCallbackOpWithCount {
            T running_total;
            T call_count;

            __device__
            BlockPrefixCallbackOpWithCount(
                T running_total,
                T call_count
                ) : running_total(running_total),
                    call_count(call_count) {}

            __device__
            T callback(T block_aggregate)
            {
                T old_prefix = running_total;
                call_count += 1;
                running_total += block_aggregate;
                return old_prefix;
            }

            __device__
            T operator()(T block_aggregate)
            {
                return callback(block_aggregate);
            }
        };

        using BlockPrefixCallbackOpWithCountInt32 = BlockPrefixCallbackOpWithCount<int32_t>;

        extern "C" __device__ int32_t block_prefix_callback_op(
            BlockPrefixCallbackOpWithCountInt32* op,
            int32_t block_aggregate
            )
        {
            *op = (*op)(block_aggregate);
            return 0;
        }
    """)
    )


def test_block_load_store_scan_simple4():
    @cuda.jit
    def kernel(d_in, d_out, items_per_thread, num_total_items, call_counts):
        threads_per_block = cuda.blockDim.x * cuda.blockDim.y * cuda.blockDim.z
        items_per_block = items_per_thread * threads_per_block

        block_offset = cuda.blockIdx.x * items_per_block
        thread_offset = cuda.threadIdx.x * items_per_thread

        thread_in = coop.local.array(items_per_thread, dtype=d_in.dtype)
        thread_out = coop.local.array(items_per_thread, dtype=d_in.dtype)

        num_valid_items = min(
            items_per_block,
            num_total_items - block_offset,
        )

        block_prefix_op = cuda.local.array(
            shape=1,
            dtype=block_prefix_callback_op_with_count_type,
        )
        block_prefix_op[0] = BlockPrefixCallbackOpWithCount(0, 0)
        # block_prefix_callback_op = BlockPrefixCallbackOpWithCount(0)

        # Load with padding
        coop.block.load(
            d_in[block_offset:],
            thread_in,
            items_per_thread=items_per_thread,
            algorithm=BlockLoadAlgorithm.WARP_TRANSPOSE,
            num_valid_items=num_valid_items,
        )

        # Zero-pad invalid thread items explicitly.
        for i in range(items_per_thread):
            global_idx = block_offset + thread_offset + i
            if global_idx >= num_total_items:
                thread_in[i] = 0

        # cuda.syncthreads()

        coop.block.scan(
            thread_in,
            thread_out,
            items_per_thread,
            block_prefix_callback_op=block_prefix_op,
        )

        tid = cuda.grid(1)
        call_counts[tid] = block_prefix_op[0].call_count

        # Store only valid items back to global memory
        coop.block.store(
            d_out[block_offset:],
            thread_out,
            items_per_thread=items_per_thread,
            algorithm=BlockStoreAlgorithm.DIRECT,
            num_valid_items=num_valid_items,
        )

    dtype = np.int32
    threads_per_block = 128
    num_total_items = threads_per_block * 4  # Total items to process
    items_per_thread = 4

    h_input = np.random.randint(0, 42, num_total_items, dtype=dtype)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array_like(d_input)

    items_per_block = threads_per_block * items_per_thread
    blocks_per_grid = (num_total_items + items_per_block - 1) // items_per_block

    num_threads = (
        threads_per_block
        if isinstance(threads_per_block, int)
        else reduce(mul, threads_per_block)
    )

    d_call_counts = cuda.to_device(np.zeros(num_threads, dtype=np.int64))

    print(
        f"blocks_per_grid: {blocks_per_grid}\n"
        f"items_per_block: {items_per_block}\n"
        f"threads_per_block: {threads_per_block}\n"
    )

    blocks_per_grid = 1
    kernel[blocks_per_grid, threads_per_block](
        d_input,
        d_output,
        items_per_thread,
        num_total_items,
        d_call_counts,
    )

    cuda.synchronize()

    h_call_counts = d_call_counts.copy_to_host()
    print(f"call_counts: {h_call_counts}")

    # Compute the correct prefix-sum on host to compare results
    h_reference = np.zeros_like(h_input)
    if len(h_input) > 0:
        h_reference[0] = 0
        h_reference[1:] = np.cumsum(h_input[:-1])

    # h_reference2 = np.cumsum(h_input)
    # h_reference3 = np.cumsum(h_input) - h_input

    h_output = d_output.copy_to_host()
    np.testing.assert_array_equal(h_output, h_reference)


def test_block_load_store_scan_simple5():
    pytest.skip(
        "Experimental prefix-op CUSource scaffolding not supported in single-phase yet."
    )
    from numba.core.typing.templates import Signature

    # ffi = cffi.FFI()

    bp_npy = np.dtype(
        [
            ("running_total", np.int32),
            ("call_count", np.int32),
        ]
    )

    ext = cuda.CUSource(
        textwrap.dedent("""
        extern "C"
        __device__
        struct BlockPrefixCallbackOpWithCount {
            int running_total;
            int call_count;
        };

        extern "C"
        __device__
        int
        block_prefix_callback_op(
            struct BlockPrefixCallbackOpWithCount* op,
            int block_aggregate
            )
        {
            int old_prefix = op->running_total;
            op->call_count += 1;
            op->running_total += block_aggregate;
            printf("block_prefix_callback_op: old_prefix=%d, "
                   "running_total=%d, call_count=%d\n",
                   old_prefix, op->running_total, op->call_count);
            return old_prefix;
        }
    """)
    )
    if ext:
        print(ext)

    args = [types.CPointer(block_prefix_callback_op_with_count_type), types.int32]
    sig = Signature(
        return_type=types.int32,
        args=args,
        recvr=None,
        pysig=None,
    )
    if sig is None:
        assert False

    # cb = cuda.declare_device(
    #     "block_prefix_callback_op",
    #    sig,
    #    link=ext,
    # )

    @cuda.jit
    def kernel1(d_in, d_out1, d_out2):
        block_prefix_op = cuda.local.array(
            shape=1,
            dtype=bp_npy.dtype,
        )
        # block_prefix_op[0] = BlockPrefixCallbackOpWithCount(0, 0)
        block_prefix_op[0].running_total = d_in[cuda.grid(1)]
        block_prefix_op[0].call_count = 0
        # results = cuda.local.array(shape=2, dtype=np.int32)

        # ptr = ffi.from_buffer(block_prefix_op)
        # results[0] = block_prefix_callback_op(ptr, cuda.threadIdx.x)
        # results[1] = block_prefix_callback_op(ptr, results[0])

        # d_out1[cuda.grid(1)] = results[0]
        # d_out2[cuda.grid(1)] = results[1]

    threads_per_block = 128
    num_total_items = threads_per_block * 4  # Total items to process
    blocks_per_grid = (num_total_items + threads_per_block - 1) // threads_per_block

    d_in = cuda.to_device(np.arange(num_total_items, dtype=np.int32))
    d_out1 = cuda.device_array(np.zeros(num_total_items, dtype=np.int32))
    d_out2 = cuda.device_array(np.zeros(num_total_items, dtype=np.int32))

    kernel1[blocks_per_grid, threads_per_block](d_in, d_out1, d_out2)
    cuda.synchronize()

    h_out1 = d_out1.copy_to_host()
    h_out2 = d_out2.copy_to_host()

    print(f"h_out1: {h_out1}")
    print(f"h_out2: {h_out2}")

    return

    @cuda.jit
    def kernel(d_in, d_out, items_per_thread, num_total_items, call_counts):
        threads_per_block = cuda.blockDim.x * cuda.blockDim.y * cuda.blockDim.z
        items_per_block = items_per_thread * threads_per_block

        block_offset = cuda.blockIdx.x * items_per_block
        thread_offset = cuda.threadIdx.x * items_per_thread

        thread_in = coop.local.array(items_per_thread, dtype=d_in.dtype)
        thread_out = coop.local.array(items_per_thread, dtype=d_in.dtype)

        num_valid_items = min(
            items_per_block,
            num_total_items - block_offset,
        )

        block_prefix_op = cuda.local.array(
            shape=1,
            dtype=block_prefix_callback_op_with_count_type,
        )
        block_prefix_op[0] = BlockPrefixCallbackOpWithCount(0, 0)
        # block_prefix_callback_op = BlockPrefixCallbackOpWithCount(0)

        # Load with padding
        coop.block.load(
            d_in[block_offset:],
            thread_in,
            items_per_thread=items_per_thread,
            algorithm=BlockLoadAlgorithm.WARP_TRANSPOSE,
            num_valid_items=num_valid_items,
        )

        # Zero-pad invalid thread items explicitly.
        for i in range(items_per_thread):
            global_idx = block_offset + thread_offset + i
            if global_idx >= num_total_items:
                thread_in[i] = 0

        # cuda.syncthreads()

        coop.block.scan(
            thread_in,
            thread_out,
            items_per_thread,
            block_prefix_callback_op=block_prefix_op,
        )

        tid = cuda.grid(1)
        call_counts[tid] = block_prefix_op[0].call_count

        # Store only valid items back to global memory
        coop.block.store(
            d_out[block_offset:],
            thread_out,
            items_per_thread=items_per_thread,
            algorithm=BlockStoreAlgorithm.DIRECT,
            num_valid_items=num_valid_items,
        )

    dtype = np.int32
    threads_per_block = 128
    num_total_items = threads_per_block * 4  # Total items to process
    items_per_thread = 4

    h_input = np.random.randint(0, 42, num_total_items, dtype=dtype)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array_like(d_input)

    items_per_block = threads_per_block * items_per_thread
    blocks_per_grid = (num_total_items + items_per_block - 1) // items_per_block

    num_threads = (
        threads_per_block
        if isinstance(threads_per_block, int)
        else reduce(mul, threads_per_block)
    )

    d_call_counts = cuda.to_device(np.zeros(num_threads, dtype=np.int64))

    print(
        f"blocks_per_grid: {blocks_per_grid}\n"
        f"items_per_block: {items_per_block}\n"
        f"threads_per_block: {threads_per_block}\n"
    )

    blocks_per_grid = 1
    kernel[blocks_per_grid, threads_per_block](
        d_input,
        d_output,
        items_per_thread,
        num_total_items,
        d_call_counts,
    )

    cuda.synchronize()

    h_call_counts = d_call_counts.copy_to_host()
    print(f"call_counts: {h_call_counts}")

    # Compute the correct prefix-sum on host to compare results
    h_reference = np.zeros_like(h_input)
    if len(h_input) > 0:
        h_reference[0] = 0
        h_reference[1:] = np.cumsum(h_input[:-1])

    # h_reference2 = np.cumsum(h_input)
    # h_reference3 = np.cumsum(h_input) - h_input

    h_output = d_output.copy_to_host()
    np.testing.assert_array_equal(h_output, h_reference)


def test_block_load_store_scan_simple55():
    import cffi
    from numba.core.typing.templates import Signature

    ffi = cffi.FFI()

    # CUDA C extension source
    ext = cuda.CUSource(
        textwrap.dedent("""
        extern "C"
        __device__
        struct BlockPrefixCallbackOpWithCount {
            int running_total;
            int call_count;
        };

        extern "C"
        __device__
        int
        block_prefix_callback_op(
            int* result,
            struct BlockPrefixCallbackOpWithCount* op,
            int block_aggregate
        )
        {
            int old_prefix = op->running_total;
            op->call_count += 1;
            op->running_total += block_aggregate;
            printf("[blockIdx.x=%d/threadIdx.x=%d; tid: %d] "
                   "block_prefix_callback_op(result=%p, op=%p): "
                   "old_prefix=%d, running_total=%d, call_count=%d\\n",
                   blockIdx.x, threadIdx.x,
                   blockIdx.x * blockDim.x + threadIdx.x,
                   result,
                   op,
                   old_prefix,
                   op->running_total,
                   op->call_count);
            *result = old_prefix;
            return 0;
        }
    """)
    )

    # Define the numpy dtype matching your device struct
    bp_npy = np.dtype(
        [
            ("running_total", np.int32),
            ("call_count", np.int32),
        ]
    )
    print(f"bp_npy: {bp_npy}")

    # Define the function signature for device declaration
    block_prefix_callback_op_with_count_type = types.Record.make_c_struct(
        [
            ("running_total", types.int32),
            ("call_count", types.int32),
        ]
    )

    args = [types.CPointer(block_prefix_callback_op_with_count_type), types.int32]
    sig = Signature(
        return_type=types.int32,
        args=args,
        recvr=None,
        pysig=None,
    )
    sig = types.int32(
        types.CPointer(block_prefix_callback_op_with_count_type),
        types.int32,
    )
    # args = [types.CPointer(block_prefix_callback_op_with_count_type), types.int32]
    # sig = Signature(return_type=types.int32, args=args)

    # Declare the external CUDA function
    block_prefix_callback_op = cuda.declare_device(
        "block_prefix_callback_op",
        sig,
        link=ext,
    )

    @cuda.jit
    def kernel1(d_in, d_out1, d_out2):
        idx = cuda.grid(1)
        if idx >= d_in.size:
            return

        # Initialize local struct
        block_prefix_op = cuda.local.array(
            1, dtype=block_prefix_callback_op_with_count_type
        )
        block_prefix_op[0]["running_total"] = d_in[idx]
        block_prefix_op[0]["call_count"] = 0

        op_ptr = ffi.from_buffer(block_prefix_op)

        # Call the external device function
        d_out1[idx] = block_prefix_callback_op(op_ptr, cuda.threadIdx.x)
        d_out2[idx] = block_prefix_callback_op(op_ptr, d_out1[idx])

        # d_out1[idx] = results[0]
        # d_out2[idx] = results[1]

    threads_per_block = 128
    num_total_items = threads_per_block * 4
    blocks_per_grid = (num_total_items + threads_per_block - 1) // threads_per_block

    # Allocate device memory
    d_in = cuda.to_device(np.arange(num_total_items, dtype=np.int32))
    d_out1 = cuda.device_array(num_total_items, dtype=np.int32)
    d_out2 = cuda.device_array(num_total_items, dtype=np.int32)

    kernel1[blocks_per_grid, threads_per_block](d_in, d_out1, d_out2)
    cuda.synchronize()

    h_out1 = d_out1.copy_to_host()
    h_out2 = d_out2.copy_to_host()

    print(f"h_out1: {h_out1}")
    print(f"h_out2: {h_out2}")

    return


def test_block_load_store_scan_simple6():
    import cffi
    from numba.core.typing.templates import Signature

    ffi = cffi.FFI()

    # CUDA C extension source
    ext = cuda.CUSource(
        textwrap.dedent("""
        extern "C"
        __device__
        struct BlockPrefixCallbackOpWithCount {
            int running_total;
            int call_count;
        };

        extern "C"
        __device__
        int
        block_prefix_callback_op(
            int* result,
            struct BlockPrefixCallbackOpWithCount* op,
            int block_aggregate
        )
        {
            int old_prefix = op->running_total;
            op->call_count += 1;
            op->running_total += block_aggregate;
            printf("[blockIdx.x=%d/threadIdx.x=%d; tid: %d] "
                   "block_prefix_callback_op(result=%p, op=%p): "
                   "old_prefix=%d, running_total=%d, call_count=%d\\n",
                   blockIdx.x, threadIdx.x,
                   blockIdx.x * blockDim.x + threadIdx.x,
                   result,
                   op,
                   old_prefix,
                   op->running_total,
                   op->call_count);
            *result = old_prefix;
            return 0;
        }
    """)
    )

    # Define the numpy dtype matching your device struct
    bp_npy = np.dtype(
        [
            ("running_total", np.int32),
            ("call_count", np.int32),
        ]
    )
    print(f"bp_npy: {bp_npy}")

    # Define the function signature for device declaration
    block_prefix_callback_op_with_count_type = types.Record.make_c_struct(
        [
            ("running_total", types.int32),
            ("call_count", types.int32),
        ]
    )

    args = [types.CPointer(block_prefix_callback_op_with_count_type), types.int32]
    sig = Signature(
        return_type=types.int32,
        args=args,
        recvr=None,
        pysig=None,
    )
    sig = types.int32(
        types.CPointer(block_prefix_callback_op_with_count_type),
        types.int32,
    )
    # args = [types.CPointer(block_prefix_callback_op_with_count_type), types.int32]
    # sig = Signature(return_type=types.int32, args=args)

    # Declare the external CUDA function
    block_prefix_callback_op = cuda.declare_device(
        "block_prefix_callback_op",
        sig,
        link=ext,
    )

    @cuda.jit
    def kernel1(d_in, d_out1, d_out2):
        idx = cuda.grid(1)
        if idx >= d_in.size:
            return

        # Initialize local struct
        block_prefix_op = cuda.local.array(
            1, dtype=block_prefix_callback_op_with_count_type
        )
        block_prefix_op[0]["running_total"] = d_in[idx]
        block_prefix_op[0]["call_count"] = 0

        op_ptr = ffi.from_buffer(block_prefix_op)

        # Call the external device function
        d_out1[idx] = block_prefix_callback_op(op_ptr, cuda.threadIdx.x)
        d_out2[idx] = block_prefix_callback_op(op_ptr, d_out1[idx])

        # d_out1[idx] = results[0]
        # d_out2[idx] = results[1]

    threads_per_block = 128
    num_total_items = threads_per_block * 4
    blocks_per_grid = (num_total_items + threads_per_block - 1) // threads_per_block

    # Allocate device memory
    d_in = cuda.to_device(np.arange(num_total_items, dtype=np.int32))
    d_out1 = cuda.device_array(num_total_items, dtype=np.int32)
    d_out2 = cuda.device_array(num_total_items, dtype=np.int32)

    kernel1[blocks_per_grid, threads_per_block](d_in, d_out1, d_out2)
    cuda.synchronize()

    h_out1 = d_out1.copy_to_host()
    h_out2 = d_out2.copy_to_host()

    print(f"h_out1: {h_out1}")
    print(f"h_out2: {h_out2}")

    return


def test_block_load_store_scan_simple7():
    @cuda.jit
    def kernel(d_in, d_out, items_per_thread, num_total_items):
        threads_per_block = cuda.blockDim.x * cuda.blockDim.y * cuda.blockDim.z
        items_per_block = items_per_thread * threads_per_block

        block_offset = cuda.blockIdx.x * items_per_block
        thread_offset = cuda.threadIdx.x * items_per_thread

        thread_in = coop.local.array(items_per_thread, dtype=d_in.dtype)
        thread_out = coop.local.array(items_per_thread, dtype=d_in.dtype)

        num_valid_items = min(
            items_per_block,
            num_total_items - block_offset,
        )

        block_prefix_op = coop.local.array(
            1, dtype=block_prefix_callback_op_with_count_type
        )
        block_prefix_op[0] = BlockPrefixCallbackOpWithCount(0, 0)

        # Load with padding
        coop.block.load(
            d_in[block_offset:],
            thread_in,
            items_per_thread=items_per_thread,
            algorithm=BlockLoadAlgorithm.WARP_TRANSPOSE,
            num_valid_items=num_valid_items,
        )

        # Zero-pad invalid thread items explicitly.
        for i in range(items_per_thread):
            global_idx = block_offset + thread_offset + i
            if global_idx >= num_total_items:
                thread_in[i] = 0

        # cuda.syncthreads()

        coop.block.scan(
            thread_in,
            thread_out,
            items_per_thread,
            block_prefix_callback_op=block_prefix_op,
        )

        # Store only valid items back to global memory
        coop.block.store(
            d_out[block_offset:],
            thread_out,
            items_per_thread=items_per_thread,
            algorithm=BlockStoreAlgorithm.DIRECT,
            num_valid_items=num_valid_items,
        )

    dtype = np.int32
    threads_per_block = 128
    num_total_items = threads_per_block * 4  # Total items to process
    items_per_thread = 4

    h_input = np.random.randint(0, 42, num_total_items, dtype=dtype)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array_like(d_input)

    items_per_block = threads_per_block * items_per_thread
    blocks_per_grid = (num_total_items + items_per_block - 1) // items_per_block

    print(
        f"blocks_per_grid: {blocks_per_grid}\n"
        f"items_per_block: {items_per_block}\n"
        f"threads_per_block: {threads_per_block}\n"
    )

    blocks_per_grid = 1
    kernel[blocks_per_grid, threads_per_block](
        d_input,
        d_output,
        items_per_thread,
        num_total_items,
    )

    cuda.synchronize()

    # Compute the correct prefix-sum on host to compare results
    h_reference = np.zeros_like(h_input)
    if len(h_input) > 0:
        h_reference[0] = 0
        h_reference[1:] = np.cumsum(h_input[:-1])

    # h_reference2 = np.cumsum(h_input)
    # h_reference3 = np.cumsum(h_input) - h_input

    h_output = d_output.copy_to_host()
    np.testing.assert_array_equal(h_output, h_reference)


def test_block_load_store_num_valid_items_with_single_phase_scan():
    @cuda.jit
    def kernel(d_in, d_out, items_per_thread, num_total_items):
        idx = cuda.grid(1)
        if idx >= d_in.size:
            return

        threads_per_block = cuda.blockDim.x * cuda.blockDim.y * cuda.blockDim.z
        items_per_block = items_per_thread * threads_per_block

        block_offset = cuda.blockIdx.x * items_per_block
        thread_offset = cuda.threadIdx.x * items_per_thread

        thread_data = coop.local.array(items_per_thread, dtype=d_in.dtype)

        block_prefix_callback_op = coop.local.array(
            shape=1,
            dtype=block_prefix_callback_op_with_count_type,
        )
        block_prefix_callback_op[0] = BlockPrefixCallbackOpWithCount(0, 0)

        while block_offset < num_total_items:
            # Calculate num_valid_items for current block
            num_valid_items = min(
                items_per_block,
                num_total_items - block_offset,
            )

            # Load with padding
            coop.block.load(
                d_in[block_offset:],
                thread_data,
                items_per_thread=items_per_thread,
                algorithm=BlockLoadAlgorithm.WARP_TRANSPOSE,
                num_valid_items=num_valid_items,
            )

            # Zero-pad invalid thread items explicitly.
            for i in range(items_per_thread):
                global_idx = block_offset + thread_offset + i
                if global_idx >= num_total_items:
                    thread_data[i] = 0

            # cuda.syncthreads()

            coop.block.scan(
                thread_data,
                thread_data,
                items_per_thread,
                block_prefix_callback_op=block_prefix_callback_op,
            )

            # Store only valid items back to global memory
            coop.block.store(
                d_out[block_offset:],
                thread_data,
                items_per_thread=items_per_thread,
                algorithm=BlockStoreAlgorithm.DIRECT,
                num_valid_items=num_valid_items,
            )

            block_offset += items_per_block * cuda.gridDim.x

            cuda.syncthreads()

    dtype = np.int32
    # threads_per_block = 128
    threads_per_block = 32
    num_total_items = 1024
    items_per_thread = 2

    h_input = np.random.randint(0, 42, num_total_items, dtype=dtype)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array_like(d_input)

    items_per_block = threads_per_block * items_per_thread
    blocks_per_grid = (num_total_items + items_per_block - 1) // items_per_block

    blocks_per_grid = 1
    kernel[blocks_per_grid, threads_per_block](
        d_input, d_output, items_per_thread, num_total_items
    )

    cuda.synchronize()

    # Compute the correct prefix-sum on host to compare results
    h_reference = np.zeros_like(h_input)
    if len(h_input) > 0:
        h_reference[0] = 0
        h_reference[1:] = np.cumsum(h_input[:-1])

    # h_reference2 = np.cumsum(h_input)
    # h_reference3 = np.cumsum(h_input) - h_input

    h_output = d_output.copy_to_host()
    np.testing.assert_array_equal(h_output, h_reference)


@pytest.mark.parametrize("threads_per_block", [32, 128])
@pytest.mark.parametrize("items_per_thread", [1, 4])
@pytest.mark.parametrize("mode", ["exclusive"])
def test_block_sum_prefix_op_stateful(threads_per_block, items_per_thread, mode):
    """
    Tests block-wide sums with a user-supplied prefix callback operator.
    Each tile of data is scanned and the prefix operator updates a running
    total for each tile within a segment.
    """
    num_threads = (
        threads_per_block
        if isinstance(threads_per_block, int)
        else reduce(mul, threads_per_block)
    )

    tile_items = num_threads * items_per_thread
    segment_size = 2 * 1024
    num_segments = 128
    num_elements = segment_size * num_segments

    prefix_op = coop.StatefulFunction(
        BlockPrefixCallbackOpWithCount,
        block_prefix_callback_op_with_count_type,
        name="block_prefix_callback_op",
    )

    if mode == "inclusive":
        sum_func = coop.block.inclusive_sum
    else:
        sum_func = coop.block.exclusive_sum

    block_sum = sum_func(
        dtype=numba.int32,
        threads_per_block=threads_per_block,
        items_per_thread=items_per_thread,
        prefix_op=prefix_op,
        algorithm=BlockScanAlgorithm.RAKING,
    )
    temp_storage_bytes = block_sum.temp_storage_bytes
    print(temp_storage_bytes)

    @cuda.jit
    def kernel(input_arr, output_arr, call_counts):
        segment_offset = cuda.blockIdx.x * segment_size
        # temp_storage = cuda.shared.array(shape=temp_storage_bytes, dtype=numba.uint8)
        block_prefix_op = coop.local.array(
            1, dtype=block_prefix_callback_op_with_count_type
        )
        block_prefix_op[0] = BlockPrefixCallbackOpWithCount(0, 0)
        # block_prefix_op = BlockPrefixCallbackOpWithCount(0, 0)
        # block_prefix_callback_op = cuda.local.array(
        #    shape=1,
        #    dtype=block_prefix_callback_op_with_count_type,
        # )
        # block_prefix_callback_op[0] = BlockPrefixCallbackOpWithCount(0)

        thread_in = coop.local.array(items_per_thread, dtype=numba.int32)
        thread_out = coop.local.array(items_per_thread, dtype=numba.int32)

        tid = row_major_tid()
        tile_offset = 0

        while tile_offset < segment_size:
            for item in range(items_per_thread):
                item_offset = tile_offset + tid * items_per_thread + item
                if item_offset < segment_size:
                    thread_in[item] = input_arr[segment_offset + item_offset]
                else:
                    thread_in[item] = 0

            coop.block.scan(
                thread_in,
                thread_out,
                items_per_thread,
                mode=mode,
                block_prefix_callback_op=block_prefix_op,
            )

            cuda.syncthreads()

            # Store the number of calls to the prefix operator.
            call_counts[tid] = block_prefix_op[0].call_count
            # call_counts[tid] = block_prefix_op.call_count

            for item in range(items_per_thread):
                item_offset = tile_offset + tid * items_per_thread + item
                if item_offset < segment_size:
                    output_arr[segment_offset + item_offset] = thread_out[item]

            tile_offset += tile_items

    h_input = np.arange(num_elements, dtype=np.int32)
    d_input = cuda.to_device(h_input)
    d_output = cuda.to_device(np.zeros(num_elements, dtype=np.int32))
    d_call_counts = cuda.to_device(np.zeros(num_threads, dtype=np.int64))

    kernel[num_segments, threads_per_block](d_input, d_output, d_call_counts)
    cuda.synchronize()

    h_output = d_output.copy_to_host()
    h_call_counts = d_call_counts.copy_to_host()
    print(f"h_call_counts: {h_call_counts}")
    ref = np.zeros_like(h_input)

    # Build the reference result for each segment.
    if mode == "inclusive":
        for seg_id in range(num_segments):
            seg_start = seg_id * segment_size
            for i in range(segment_size):
                if i == 0:
                    ref[seg_start + i] = h_input[seg_start + i]
                else:
                    ref[seg_start + i] = ref[seg_start + i - 1] + h_input[seg_start + i]
    else:
        for seg_id in range(num_segments):
            seg_start = seg_id * segment_size
            ref[seg_start] = 0
            for i in range(1, segment_size):
                ref[seg_start + i] = ref[seg_start + i - 1] + h_input[seg_start + i - 1]

    sum_equal = np.sum(h_output == ref)
    print(f"Sum equal: {sum_equal} / {num_elements}")
    np.testing.assert_array_equal(h_output, ref)

    sig = (types.int32[::1], types.int32[::1], types.int64[::1])
    sass = kernel.inspect_sass(sig)
    assert "LDL" not in sass
    assert "STL" not in sass


if __name__ == "__main__":
    # pytest.main([__file__])
    # test_block_load_store_scan_simple1()
    # test_block_load_store_scan_simple2()
    # test_block_load_store_scan_simple3()
    # test_block_load_store_scan_simple4()
    # test_block_load_store_num_valid_items_with_single_phase_scan()
    test_block_sum_prefix_op(128, 4, "exclusive")
    # test_block_sum_prefix_op(128, 4, "inclusive")
