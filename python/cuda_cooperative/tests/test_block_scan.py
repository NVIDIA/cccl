# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
test_block_scan.py

This file contains unit tests for cuda.cooperative.block_scan. It covers both
valid and invalid usage scenarios, tests sum-based scans, user-defined operators
and types, prefix callback operators, and known operators such as min, max, and
bitwise XOR.
"""

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
from pynvjitlink import patch

import cuda.cooperative.experimental as cudax
from cuda.cooperative.experimental.block._block_scan import (
    ScanOp,
)

numba.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0

# Patching the Numba linker to enable LTO as needed.
patch.patch_numba_linker(lto=True)


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
@pytest.mark.parametrize("algorithm", ["raking"])
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
    if algorithm == "raking_memoize" and num_threads >= 512:
        pytest.skip("raking_memoize can exceed resources for >= 512 threads.")

    if mode == "inclusive":
        scan_func = cudax.block.inclusive_sum
    else:
        scan_func = cudax.block.exclusive_sum

    block_sum = scan_func(
        dtype=T,
        threads_per_block=threads_per_block,
        items_per_thread=items_per_thread,
        algorithm=algorithm,
    )
    temp_storage_bytes = block_sum.temp_storage_bytes

    @cuda.jit(link=block_sum.files)
    def kernel(input_arr, output_arr):
        tid = row_major_tid()
        temp_storage = cuda.shared.array(shape=temp_storage_bytes, dtype=numba.uint8)
        thread_data = cuda.local.array(items_per_thread, dtype=T)

        for i in range(items_per_thread):
            thread_data[i] = input_arr[tid * items_per_thread + i]

        if items_per_thread == 1:
            thread_data[0] = block_sum(temp_storage, thread_data[0])
        else:
            block_sum(temp_storage, thread_data, thread_data)

        for i in range(items_per_thread):
            output_arr[tid * items_per_thread + i] = thread_data[i]

    dtype_np = NUMBA_TYPES_TO_NP[T]
    items_per_tile = num_threads * items_per_thread
    h_input = random_int(items_per_tile, dtype_np)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(items_per_tile, dtype=dtype_np)

    kernel[1, threads_per_block](d_input, d_output)
    cuda.synchronize()

    output = d_output.copy_to_host()
    if mode == "inclusive":
        reference = np.cumsum(h_input)
    else:
        reference = np.cumsum(h_input) - h_input

    np.testing.assert_array_equal(output, reference)

    sig = (T[::1], T[::1])
    sass = kernel.inspect_sass(sig)
    # Check that no device memory loads/stores appear in SASS.
    assert "LDL" not in sass
    assert "STL" not in sass


@pytest.mark.parametrize("threads_per_block", [32, (4, 16), (4, 8, 8)])
@pytest.mark.parametrize("items_per_thread", [1, 4])
@pytest.mark.parametrize("mode", ["inclusive", "exclusive"])
@pytest.mark.parametrize("algorithm", ["raking"])
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

    tile_items = num_threads * items_per_thread
    segment_size = 2 * 1024
    num_segments = 128
    num_elements = segment_size * num_segments

    prefix_op = cudax.StatefulFunction(
        BlockPrefixCallbackOp, block_prefix_callback_op_type
    )

    if mode == "inclusive":
        sum_func = cudax.block.inclusive_sum
    else:
        sum_func = cudax.block.exclusive_sum

    block_sum = sum_func(
        dtype=numba.int32,
        threads_per_block=threads_per_block,
        items_per_thread=items_per_thread,
        prefix_op=prefix_op,
        algorithm=algorithm,
    )
    temp_storage_bytes = block_sum.temp_storage_bytes

    @cuda.jit(link=block_sum.files)
    def kernel(input_arr, output_arr):
        segment_offset = cuda.blockIdx.x * segment_size
        temp_storage = cuda.shared.array(shape=temp_storage_bytes, dtype=numba.uint8)
        block_prefix_op = cuda.local.array(shape=1, dtype=block_prefix_callback_op_type)
        block_prefix_op[0] = BlockPrefixCallbackOp(0)
        thread_in = cuda.local.array(items_per_thread, dtype=numba.int32)
        thread_out = cuda.local.array(items_per_thread, dtype=numba.int32)

        tid = row_major_tid()
        tile_offset = 0

        while tile_offset < segment_size:
            for item in range(items_per_thread):
                item_offset = tile_offset + tid * items_per_thread + item
                if item_offset < segment_size:
                    thread_in[item] = input_arr[segment_offset + item_offset]
                else:
                    thread_in[item] = 0

            if items_per_thread == 1:
                thread_out[0] = block_sum(temp_storage, thread_in[0], block_prefix_op)
            else:
                block_sum(temp_storage, thread_in, thread_out, block_prefix_op)

            for item in range(items_per_thread):
                item_offset = tile_offset + tid * items_per_thread + item
                if item_offset < segment_size:
                    output_arr[segment_offset + item_offset] = thread_out[item]

            tile_offset += tile_items

    h_input = np.arange(num_elements, dtype=np.int32)
    d_input = cuda.to_device(h_input)
    d_output = cuda.to_device(np.zeros(num_elements, dtype=np.int32))

    kernel[num_segments, threads_per_block](d_input, d_output)
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

    sig = (types.int32[::1], types.int32[::1])
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
        sum_func = cudax.block.inclusive_sum
    else:
        sum_func = cudax.block.exclusive_sum

    with pytest.raises(ValueError):
        sum_func(numba.int32, threads_per_block, items_per_thread)


@pytest.mark.parametrize("mode", ["inclusive", "exclusive"])
def test_block_scan_sum_invalid_algorithm(mode):
    """
    Tests that supplying an unsupported algorithm to the sum-based scans
    raises a ValueError.
    """
    if mode == "inclusive":
        sum_func = cudax.block.inclusive_sum
    else:
        sum_func = cudax.block.exclusive_sum

    with pytest.raises(ValueError):
        sum_func(numba.int32, 128, algorithm="invalid_algorithm")


@pytest.mark.parametrize("initial_value", [None, Complex(0, 0), Complex(1, 1)])
@pytest.mark.parametrize("threads_per_block", [32, (4, 16), (4, 8, 8)])
@pytest.mark.parametrize("items_per_thread", [1, 4])
@pytest.mark.parametrize("mode", ["inclusive", "exclusive"])
@pytest.mark.parametrize("algorithm", ["raking"])
def test_block_scan_user_defined_type(
    initial_value, items_per_thread, threads_per_block, mode, algorithm
):
    """
    Tests block-wide scans for a user-defined (Complex) type. Uses an addition
    operator to sum real and imaginary parts respectively.
    """
    num_threads = (
        threads_per_block
        if isinstance(threads_per_block, int)
        else reduce(mul, threads_per_block)
    )

    if algorithm == "raking_memoize" and num_threads >= 512:
        pytest.skip("raking_memoize can exceed resources for >= 512 threads.")

    # Numba balks at the `block_op(temp_storage, thread_in, thread_out)`
    # call for items_per_thread > 1 when using user-defined types.
    if items_per_thread > 1:
        pytest.skip("items_per_thread>1 not supported for user defined type.")

    # Our custom operator (add complex).
    def op(result_ptr, lhs_ptr, rhs_ptr):
        # We need the explicit cast to prevent automatic up-casting to i64.
        real_val = numba.int32(lhs_ptr[0].real + rhs_ptr[0].real)
        imag_val = numba.int32(lhs_ptr[0].imag + rhs_ptr[0].imag)
        result_ptr[0] = Complex(real_val, imag_val)

    if mode == "inclusive":
        scan_func = cudax.block.inclusive_scan
        if items_per_thread == 1:
            # Initial values aren't supported for inclusive scans with
            # items_per_thread=1.
            if initial_value is not None:
                pytest.skip(
                    "initial_value not supported for inclusive "
                    "scans with items_per_thread=1"
                )
    else:
        if initial_value is not None:
            pytest.skip("initial_value not supported for exclusive scans")
        scan_func = cudax.block.exclusive_scan

    block_op = scan_func(
        dtype=complex_type,
        scan_op=op,
        threads_per_block=threads_per_block,
        items_per_thread=items_per_thread,
        initial_value=initial_value,
        algorithm=algorithm,
        methods={
            "construct": Complex.construct,
            "assign": Complex.assign,
        },
    )
    temp_storage_bytes = block_op.temp_storage_bytes

    # N.B. I had to use two separate kernels here, because having a single
    #      kernel with `if initial_value is not None` did not yield a kernel
    #      that Numba could compile.
    if initial_value is not None:

        @cuda.jit(link=block_op.files)
        def kernel(input_arr, output_arr):
            tid = row_major_tid()
            temp_storage = cuda.shared.array(
                shape=temp_storage_bytes, dtype=numba.uint8
            )
            thread_in = cuda.local.array(items_per_thread, dtype=complex_type)
            thread_out = cuda.local.array(items_per_thread, dtype=complex_type)

            for i in range(items_per_thread):
                thread_in[i] = Complex(
                    input_arr[tid * items_per_thread + i],
                    input_arr[num_threads + tid * items_per_thread + i],
                )

            if items_per_thread == 1:
                thread_out[0] = block_op(temp_storage, thread_in[0], initial_value)
            else:
                block_op(temp_storage, thread_in, thread_out, initial_value)

            for i in range(items_per_thread):
                output_arr[tid * items_per_thread + i] = thread_out[i].real
                output_arr[num_threads + tid * items_per_thread + i] = thread_out[
                    i
                ].imag

    else:

        @cuda.jit(link=block_op.files)
        def kernel(input_arr, output_arr):
            tid = row_major_tid()
            temp_storage = cuda.shared.array(
                shape=temp_storage_bytes, dtype=numba.uint8
            )
            thread_in = cuda.local.array(items_per_thread, dtype=complex_type)
            thread_out = cuda.local.array(items_per_thread, dtype=complex_type)

            for i in range(items_per_thread):
                thread_in[i] = Complex(
                    input_arr[tid * items_per_thread + i],
                    input_arr[num_threads + tid * items_per_thread + i],
                )

            if items_per_thread == 1:
                thread_out[0] = block_op(temp_storage, thread_in[0])
            else:
                block_op(temp_storage, thread_in, thread_out)

            for i in range(items_per_thread):
                output_arr[tid * items_per_thread + i] = thread_out[i].real
                output_arr[num_threads + tid * items_per_thread + i] = thread_out[
                    i
                ].imag

    h_input = random_int(2 * num_threads, "int32")
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(2 * num_threads, dtype=np.int32)
    kernel[1, threads_per_block](d_input, d_output)
    cuda.synchronize()

    h_output = d_output.copy_to_host()
    real_vals = h_input[:num_threads]
    imag_vals = h_input[num_threads:]

    if mode == "inclusive":
        real_ref = np.cumsum(real_vals)
        imag_ref = np.cumsum(imag_vals)
    else:
        real_ref = np.zeros_like(real_vals)
        imag_ref = np.zeros_like(imag_vals)

        if len(real_vals) > 1:
            real_ref[1:] = np.cumsum(real_vals)[:-1]
            imag_ref[1:] = np.cumsum(imag_vals)[:-1]

        if initial_value is None:
            real_ref[0] = h_output[0]
            imag_ref[0] = h_output[num_threads]

    np.testing.assert_array_equal(h_output[:num_threads], real_ref)
    np.testing.assert_array_equal(h_output[num_threads:], imag_ref)

    sig = (numba.int32[::1], numba.int32[::1])
    sass = kernel.inspect_sass(sig)
    assert "LDL" not in sass
    assert "STL" not in sass


@pytest.mark.parametrize("T", [types.uint32, types.float64])
@pytest.mark.parametrize("threads_per_block", [32, (4, 16), (4, 8, 8)])
@pytest.mark.parametrize("items_per_thread", [1, 4])
@pytest.mark.parametrize("mode", ["inclusive", "exclusive"])
@pytest.mark.parametrize("algorithm", ["raking"])
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

    if mode == "inclusive":
        scan_func = cudax.block.inclusive_scan
    else:
        scan_func = cudax.block.exclusive_scan

    # Example custom operator that just adds two operands.
    def op(a: T, b: T) -> T:
        return T(a + b)  # Casting to match T if needed.

    block_op = scan_func(
        dtype=T,
        threads_per_block=threads_per_block,
        scan_op=op,
        items_per_thread=items_per_thread,
        algorithm=algorithm,
    )
    temp_storage_bytes = block_op.temp_storage_bytes

    @cuda.jit(link=block_op.files)
    def kernel(input_arr, output_arr):
        # Get the correct thread ID based on thread configuration
        tid = row_major_tid()
        temp_storage = cuda.shared.array(shape=temp_storage_bytes, dtype=numba.uint8)
        thread_in = cuda.local.array(items_per_thread, dtype=T)
        thread_out = cuda.local.array(items_per_thread, dtype=T)

        # Load the input data
        for i in range(items_per_thread):
            thread_in[i] = input_arr[tid * items_per_thread + i]

        if items_per_thread == 1:
            thread_out[0] = block_op(temp_storage, thread_in[0])
        else:
            block_op(temp_storage, thread_in, thread_out)

        # Store the output data
        for i in range(items_per_thread):
            output_arr[tid * items_per_thread + i] = thread_out[i]

    dtype_np = NUMBA_TYPES_TO_NP[T]
    total_items = num_threads * items_per_thread
    h_input = random_int(total_items, dtype_np)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(total_items, dtype=dtype_np)

    kernel[1, threads_per_block](d_input, d_output)
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

    sig = (T[::1], T[::1])
    sass = kernel.inspect_sass(sig)
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
      4) User-defined types are not supported for items_per_thread > 1.
    """
    if mode == "inclusive":
        scan_func = cudax.block.inclusive_scan
    else:
        scan_func = cudax.block.exclusive_scan

    # 1) For inclusive scans with items_per_thread=1, initial_value is invalid.
    if mode == "inclusive":
        with pytest.raises(
            ValueError,
            match=(
                "initial_value is not supported for inclusive scans "
                "with items_per_thread == 1"
            ),
        ):
            scan_func(numba.int32, 128, scan_op="*", initial_value=0)

    # 2) For exclusive scans with items_per_thread=1 and a prefix callback,
    #    initial_value is invalid.
    if mode == "exclusive":
        prefix_op = cudax.StatefulFunction(
            BlockPrefixCallbackOp, block_prefix_callback_op_type
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
                numba.int32,
                128,
                scan_op="*",
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
        scan_func(complex_type, 128, scan_op="+", items_per_thread=2)

    # 4) User-defined types are not supported for items_per_thread > 1.
    with pytest.raises(
        ValueError,
        match="user-defined types are not supported for items_per_thread > 1",
    ):
        scan_func(
            complex_type,
            128,
            scan_op="+",
            items_per_thread=2,
            methods={
                "construct": Complex.construct,
                "assign": Complex.assign,
            },
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

    def add_op(a, b):
        return a + b

    prefix_op = cudax.StatefulFunction(
        BlockPrefixCallbackOp, block_prefix_callback_op_type
    )

    if mode == "inclusive":
        scan_func = cudax.block.inclusive_scan
    else:
        scan_func = cudax.block.exclusive_scan

    block_op = scan_func(
        dtype=T,
        threads_per_block=threads_per_block,
        scan_op=add_op,
        items_per_thread=items_per_thread,
        prefix_op=prefix_op,
    )
    temp_storage_bytes = block_op.temp_storage_bytes

    @cuda.jit(link=block_op.files)
    def kernel(input_arr, output_arr):
        tid = row_major_tid()
        temp_storage = cuda.shared.array(shape=temp_storage_bytes, dtype=numba.uint8)
        thread_data = cuda.local.array(items_per_thread, dtype=T)
        for i in range(items_per_thread):
            thread_data[i] = input_arr[tid * items_per_thread + i]

        # Initialize the prefix callback operator with 'initial_value'.
        block_prefix_op = cuda.local.array(shape=1, dtype=block_prefix_callback_op_type)
        block_prefix_op[0] = BlockPrefixCallbackOp(initial_value)

        block_op(temp_storage, thread_data, thread_data, block_prefix_op)

        for i in range(items_per_thread):
            output_arr[tid * items_per_thread + i] = thread_data[i]

    dtype_np = NUMBA_TYPES_TO_NP[T]
    total_items = num_threads * items_per_thread
    h_input = random_int(total_items, dtype_np)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(total_items, dtype=dtype_np)

    kernel[1, threads_per_block](d_input, d_output)
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

    sig = (T[::1], T[::1])
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
@pytest.mark.parametrize("algorithm", ["raking"])
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

    # We skip raking_memoize in this test for brevity, but you can add it if
    # desired and check resource constraints.
    if algorithm not in ["raking", "warp_scans"]:
        pytest.skip(f"Skipping algorithm {algorithm} for known ops test.")

    if mode == "inclusive":
        scan_func = cudax.block.inclusive_scan
    else:
        scan_func = cudax.block.exclusive_scan

    op = ScanOp(scan_op)
    assert op.is_known

    block_op = scan_func(
        dtype=T,
        threads_per_block=threads_per_block,
        items_per_thread=items_per_thread,
        scan_op=scan_op,
        algorithm=algorithm,
    )
    temp_storage_bytes = block_op.temp_storage_bytes

    @cuda.jit(link=block_op.files)
    def kernel(input_arr, output_arr):
        tid = row_major_tid()
        temp_storage = cuda.shared.array(shape=temp_storage_bytes, dtype=numba.uint8)
        thread_data = cuda.local.array(items_per_thread, dtype=T)
        for i in range(items_per_thread):
            thread_data[i] = input_arr[tid * items_per_thread + i]

        if items_per_thread == 1:
            thread_data[0] = block_op(temp_storage, thread_data[0])
        else:
            block_op(temp_storage, thread_data, thread_data)

        for i in range(items_per_thread):
            output_arr[tid * items_per_thread + i] = thread_data[i]

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
    k(d_input, d_output)
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

    sig = (T[::1], T[::1])
    sass = kernel.inspect_sass(sig)
    assert "LDL" not in sass
    assert "STL" not in sass
