# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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

numba.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0


patch.patch_numba_linker(lto=True)


@pytest.mark.parametrize("T", [types.uint32, types.uint64])
@pytest.mark.parametrize(
    "threads_per_block", [32, 64, 128, 256, 512, 1024, (8, 16), (2, 4, 8)]
)
@pytest.mark.parametrize("items_per_thread", [1, 2, 3, 4])
@pytest.mark.parametrize("mode", ["inclusive", "exclusive"])
@pytest.mark.parametrize("algorithm", ["raking", "raking_memoize", "warp_scans"])
def test_block_sum(T, threads_per_block, items_per_thread, mode, algorithm):
    num_threads_per_block = (
        threads_per_block
        if isinstance(threads_per_block, int)
        else reduce(mul, threads_per_block)
    )

    if algorithm == "raking_memoize" and num_threads_per_block >= 512:
        # We can hit CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES with raking_memoize in
        # certain configurations, e.g.: 1024 threads_per_block, or 512
        # threads_per_block, 3 items_per_thread, and T == uint64, etc.
        pytest.skip("raking_memoize: skipping threads_per_block >= 512")

    if mode == "inclusive":
        target_sum = cudax.block.inclusive_sum
    else:
        target_sum = cudax.block.exclusive_sum

    block_sum = target_sum(
        dtype=T,
        threads_per_block=threads_per_block,
        items_per_thread=items_per_thread,
        algorithm=algorithm,
    )
    temp_storage_bytes = block_sum.temp_storage_bytes

    @cuda.jit(link=block_sum.files)
    def kernel(input, output):
        tid = row_major_tid()
        temp_storage = cuda.shared.array(shape=temp_storage_bytes, dtype="uint8")
        thread_data = cuda.local.array(shape=items_per_thread, dtype=T)
        for i in range(items_per_thread):
            thread_data[i] = input[tid * items_per_thread + i]
        if items_per_thread == 1:
            thread_data[0] = block_sum(temp_storage, thread_data[0])
        else:
            block_sum(temp_storage, thread_data, thread_data)
        for i in range(items_per_thread):
            output[tid * items_per_thread + i] = thread_data[i]

    dtype_np = NUMBA_TYPES_TO_NP[T]
    items_per_tile = num_threads_per_block * items_per_thread
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

    for i in range(items_per_tile):
        assert output[i] == reference[i]

    sig = (T[::1], T[::1])
    sass = kernel.inspect_sass(sig)

    assert "LDL" not in sass
    assert "STL" not in sass


class BlockPrefixCallbackOp:
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


@pytest.mark.parametrize(
    "threads_per_block", [32, 64, 128, 256, 512, 1024, (8, 16), (2, 4, 8)]
)
@pytest.mark.parametrize("items_per_thread", [1, 2, 3, 4])
@pytest.mark.parametrize("mode", ["inclusive", "exclusive"])
@pytest.mark.parametrize("algorithm", ["raking", "raking_memoize", "warp_scans"])
def test_block_sum_prefix(threads_per_block, items_per_thread, mode, algorithm):
    num_threads_per_block = (
        threads_per_block
        if isinstance(threads_per_block, int)
        else reduce(mul, threads_per_block)
    )

    if algorithm == "raking_memoize" and num_threads_per_block >= 512:
        # We can hit CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES with raking_memoize in
        # certain configurations, e.g.: 1024 threads_per_block, or 512
        # threads_per_block, 3 items_per_thread, and T == uint64, etc.
        pytest.skip("raking_memoize: skipping threads_per_block >= 512")

    tile_items = num_threads_per_block * items_per_thread
    segment_size = 2 * 1024
    num_segments = 128
    num_elements = segment_size * num_segments

    prefix_op = cudax.StatefulFunction(
        BlockPrefixCallbackOp, block_prefix_callback_op_type
    )

    if mode == "inclusive":
        target_sum = cudax.block.inclusive_sum
    else:
        target_sum = cudax.block.exclusive_sum

    block_sum = target_sum(
        dtype=numba.types.int32,
        threads_per_block=threads_per_block,
        items_per_thread=items_per_thread,
        prefix_op=prefix_op,
        algorithm=algorithm,
    )
    temp_storage_bytes = block_sum.temp_storage_bytes

    @cuda.jit(link=block_sum.files)
    def kernel(input, output):
        segment_offset = cuda.blockIdx.x * segment_size
        temp_storage = cuda.shared.array(shape=temp_storage_bytes, dtype="uint8")
        block_prefix_op = cuda.local.array(shape=1, dtype=block_prefix_callback_op_type)
        block_prefix_op[0] = BlockPrefixCallbackOp(0)
        thread_input = cuda.local.array(shape=items_per_thread, dtype="int32")
        thread_output = cuda.local.array(shape=items_per_thread, dtype="int32")

        tid = row_major_tid()
        tile_offset = 0

        while tile_offset < segment_size:
            for item in range(items_per_thread):
                item_offset = tile_offset + tid * items_per_thread + item
                thread_input[item] = (
                    input[segment_offset + item_offset]
                    if item_offset < segment_size
                    else 0
                )

            if items_per_thread == 1:
                thread_output[0] = block_sum(
                    temp_storage, thread_input[0], block_prefix_op
                )
            else:
                block_sum(temp_storage, thread_input, thread_output, block_prefix_op)

            for item in range(items_per_thread):
                item_offset = tile_offset + tid * items_per_thread + item
                if item_offset < segment_size:
                    output[segment_offset + item_offset] = thread_output[item]

            tile_offset += tile_items

    h_input = np.arange(num_elements, dtype="int32")
    d_input = cuda.to_device(h_input)
    d_output = cuda.to_device(np.zeros(num_elements, dtype="int32"))
    kernel[num_segments, threads_per_block](d_input, d_output)
    cuda.synchronize()

    h_output = d_output.copy_to_host()
    h_reference = np.zeros(segment_size * num_segments, dtype="int32")

    if mode == "inclusive":
        for sid in range(num_segments):
            for i in range(segment_size):
                if i == 0:
                    h_reference[sid * segment_size + i] = h_input[
                        sid * segment_size + i
                    ]
                else:
                    h_reference[sid * segment_size + i] = (
                        h_reference[sid * segment_size + i - 1]
                        + h_input[sid * segment_size + i]
                    )
    else:
        for sid in range(num_segments):
            h_reference[sid * segment_size] = 0
            for i in range(1, segment_size):
                h_reference[sid * segment_size + i] = (
                    h_reference[sid * segment_size + i - 1]
                    + h_input[sid * segment_size + i - 1]
                )

    for sid in range(num_segments):
        for i in range(segment_size):
            if h_output[sid * segment_size + i] != h_reference[sid * segment_size + i]:
                print(
                    sid,
                    i,
                    h_output[sid * segment_size + i],
                    h_reference[sid * segment_size + i],
                )

    sig = (types.int32[::1], types.int32[::1])
    sass = kernel.inspect_sass(sig)

    assert "LDL" not in sass
    assert "STL" not in sass


@pytest.mark.parametrize(
    "threads_per_block", [32, 64, 128, 256, 512, 1024, (8, 16), (2, 4, 8)]
)
@pytest.mark.parametrize("items_per_thread", [0, -1, -127])
@pytest.mark.parametrize("mode", ["inclusive", "exclusive"])
def test_block_scan_exclusive_sum_invalid_items_per_thread(
    threads_per_block, items_per_thread, mode
):
    if mode == "inclusive":
        target_sum = cudax.block.inclusive_sum
    else:
        target_sum = cudax.block.exclusive_sum

    with pytest.raises(ValueError):
        target_sum(numba.int32, threads_per_block, items_per_thread)


@pytest.mark.parametrize("mode", ["inclusive", "exclusive"])
def test_block_scan_sum_invalid_algorithm(mode):
    if mode == "inclusive":
        target_sum = cudax.block.inclusive_sum
    else:
        target_sum = cudax.block.exclusive_sum

    with pytest.raises(ValueError):
        target_sum(numba.int32, 128, algorithm="invalid_algorithm")


@pytest.mark.parametrize("items_per_thread", [1, 2, 3, 4])
@pytest.mark.parametrize(
    "threads_per_block", [32, 64, 128, 256, 512, 1024, (8, 16), (2, 4, 8)]
)
@pytest.mark.parametrize("mode", ["inclusive", "exclusive"])
@pytest.mark.parametrize("algorithm", ["raking", "raking_memoize", "warp_scans"])
def test_block_sum_of_user_defined_type(
    items_per_thread, threads_per_block, mode, algorithm
):
    num_threads_per_block = (
        threads_per_block
        if isinstance(threads_per_block, int)
        else reduce(mul, threads_per_block)
    )

    if algorithm == "raking_memoize" and num_threads_per_block >= 512:
        # We can hit CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES with raking_memoize in
        # certain configurations, e.g.: 1024 threads_per_block, or 512
        # threads_per_block, 3 items_per_thread, and T == uint64, etc.
        pytest.skip("raking_memoize: skipping threads_per_block >= 512")

    def op(result_ptr, lhs_ptr, rhs_ptr):
        real_value = numba.int32(lhs_ptr[0].real + rhs_ptr[0].real)
        imag_value = numba.int32(lhs_ptr[0].imag + rhs_ptr[0].imag)
        result_ptr[0] = Complex(real_value, imag_value)

    if mode == "inclusive":
        target_sum = cudax.block.inclusive_scan
    else:
        target_sum = cudax.block.exclusive_scan

    block_scan = target_sum(
        dtype=complex_type,
        scan_op=op,
        threads_per_block=threads_per_block,
        algorithm=algorithm,
        methods={
            "construct": Complex.construct,
            "assign": Complex.assign,
        },
    )
    temp_storage_bytes = block_scan.temp_storage_bytes

    @cuda.jit(link=block_scan.files)
    def kernel(input, output):
        tid = row_major_tid()
        temp_storage = cuda.shared.array(shape=temp_storage_bytes, dtype="uint8")
        thread_input = Complex(input[tid], input[num_threads_per_block + tid])
        thread_output = block_scan(temp_storage, thread_input)

        output[tid * 2] = thread_output.real
        output[tid * 2 + 1] = thread_output.imag

    h_input = random_int(2 * num_threads_per_block, "int32")
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(2 * num_threads_per_block, dtype="int32")
    kernel[1, threads_per_block](d_input, d_output)
    cuda.synchronize()
    h_output = d_output.copy_to_host()

    # Calculate expected results
    real_values = h_input[:num_threads_per_block]
    imag_values = h_input[num_threads_per_block:]

    if mode == "inclusive":
        real_expected = np.cumsum(real_values)
        imag_expected = np.cumsum(imag_values)
    else:
        real_expected = np.zeros_like(real_values)
        imag_expected = np.zeros_like(imag_values)
        real_expected[1:] = np.cumsum(real_values)[:-1]
        imag_expected[1:] = np.cumsum(imag_values)[:-1]

    # Verify results
    for i in range(num_threads_per_block):
        assert h_output[i * 2] == real_expected[i]
        assert h_output[i * 2 + 1] == imag_expected[i]

    sig = (numba.int32[::1], numba.int32[::1])
    sass = kernel.inspect_sass(sig)

    assert "LDL" not in sass
    assert "STL" not in sass


@pytest.mark.parametrize("items_per_thread", [1, 2, 3, 4])
@pytest.mark.parametrize("T", [types.uint32, types.uint64])
@pytest.mark.parametrize(
    "threads_per_block", [32, 64, 128, 256, 512, 1024, (8, 16), (2, 4, 8)]
)
@pytest.mark.parametrize("mode", ["inclusive", "exclusive"])
@pytest.mark.parametrize("algorithm", ["raking", "raking_memoize", "warp_scans"])
def test_block_scan_with_callable(
    T, threads_per_block, items_per_thread, mode, algorithm
):
    num_threads_per_block = (
        threads_per_block
        if isinstance(threads_per_block, int)
        else reduce(mul, threads_per_block)
    )

    if algorithm == "raking_memoize" and num_threads_per_block >= 512:
        # We can hit CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES with raking_memoize in
        # certain configurations, e.g.: 1024 threads_per_block, or 512
        # threads_per_block, 3 items_per_thread, and T == uint64, etc.
        pytest.skip("raking_memoize: skipping threads_per_block >= 512")

    if mode == "inclusive":
        target_scan = cudax.block.inclusive_scan
    else:
        target_scan = cudax.block.exclusive_scan

    # I added the T type hints in an effort to fix the dtype mismatch;
    # didn't work.
    def op(a: T, b: T) -> T:
        return T(a + b)

    block_scan = target_scan(
        dtype=T,
        threads_per_block=threads_per_block,
        scan_op=op,
        items_per_thread=items_per_thread,
        algorithm=algorithm,
    )
    temp_storage_bytes = block_scan.temp_storage_bytes

    @cuda.jit(link=block_scan.files)
    def kernel(input, output):
        tid = row_major_tid()
        temp_storage = cuda.shared.array(shape=temp_storage_bytes, dtype="uint8")
        thread_data = cuda.local.array(shape=items_per_thread, dtype=T)
        for i in range(items_per_thread):
            thread_data[i] = input[tid * items_per_thread + i]
        if items_per_thread == 1:
            thread_data[0] = block_scan(temp_storage, thread_data[0])
        else:
            block_scan(temp_storage, thread_data, thread_data)
        for i in range(items_per_thread):
            output[tid * items_per_thread + i] = thread_data[i]

    dtype_np = NUMBA_TYPES_TO_NP[T]
    items_per_tile = num_threads_per_block * items_per_thread
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

    for i in range(items_per_tile):
        assert output[i] == reference[i], (i, items_per_tile, output[i], reference[i])

    sig = (T[::1], T[::1])
    sass = kernel.inspect_sass(sig)

    assert "LDL" not in sass
    assert "STL" not in sass


@pytest.mark.parametrize("mode", ["inclusive", "exclusive"])
def test_block_scan_initial_value_invariants(mode):
    if mode == "inclusive":
        target_scan = cudax.block.inclusive_scan
    else:
        target_scan = cudax.block.exclusive_scan

    # Test 1: For inclusive scans with items_per_thread == 1, initial values are
    # not supported
    if mode == "inclusive":
        with pytest.raises(
            ValueError,
            match=(
                "initial_value is not supported for inclusive "
                "scans with items_per_thread == 1"
            ),
        ):
            target_scan(
                numba.int32, 128, scan_op="+", initial_value=0, items_per_thread=1
            )

    # Test 2: For exclusive scans with items_per_thread == 1 and a block prefix
    # callback, initial values are not supported.
    if mode == "exclusive":
        prefix_op = cudax.StatefulFunction(
            BlockPrefixCallbackOp, block_prefix_callback_op_type
        )
        with pytest.raises(
            ValueError,
            match=(
                "initial_value is not supported for exclusive "
                "scans with items_per_thread == 1 and a block "
                "prefix callback operator"
            ),
        ):
            target_scan(
                numba.int32,
                128,
                scan_op="+",
                initial_value=0,
                items_per_thread=1,
                prefix_op=prefix_op,
            )

    # Test 3: For both scan types with items_per_thread > 1 and no block prefix
    # callback, initial values are required.
    with pytest.raises(
        ValueError,
        match=(
            "initial_value is required for both inclusive and "
            "exclusive scans when items_per_thread > 1 and no "
            "block prefix callback operator has been supplied"
        ),
    ):
        target_scan(numba.int32, 128, scan_op="+", items_per_thread=2)


@pytest.mark.parametrize("T", [types.uint32])
@pytest.mark.parametrize("threads_per_block", [32, 64])
@pytest.mark.parametrize("items_per_thread", [2, 3])
@pytest.mark.parametrize("initial_value", [-1, 0, 100])
@pytest.mark.parametrize("mode", ["inclusive", "exclusive"])
def test_block_scan_with_prefix_op_multi_items(
    T, threads_per_block, items_per_thread, initial_value, mode
):
    num_threads_per_block = (
        threads_per_block
        if isinstance(threads_per_block, int)
        else reduce(mul, threads_per_block)
    )

    if mode == "inclusive":
        target_scan = cudax.block.inclusive_scan
    else:
        target_scan = cudax.block.exclusive_scan

    # Define a scan operation
    def op(a, b):
        return a + b

    prefix_op = cudax.StatefulFunction(
        BlockPrefixCallbackOp, block_prefix_callback_op_type
    )

    block_scan = target_scan(
        dtype=T,
        threads_per_block=threads_per_block,
        scan_op=op,
        items_per_thread=items_per_thread,
        prefix_op=prefix_op,
    )
    temp_storage_bytes = block_scan.temp_storage_bytes

    @cuda.jit(link=block_scan.files)
    def kernel(input, output):
        tid = row_major_tid()
        temp_storage = cuda.shared.array(shape=temp_storage_bytes, dtype="uint8")
        thread_data = cuda.local.array(shape=items_per_thread, dtype=T)
        for i in range(items_per_thread):
            thread_data[i] = input[tid * items_per_thread + i]

        block_prefix_op = cuda.local.array(shape=1, dtype=block_prefix_callback_op_type)
        block_prefix_op[0] = BlockPrefixCallbackOp(initial_value)

        block_scan(temp_storage, thread_data, thread_data, block_prefix_op)

        for i in range(items_per_thread):
            output[tid * items_per_thread + i] = thread_data[i]

    dtype_np = NUMBA_TYPES_TO_NP[T]
    items_per_tile = num_threads_per_block * items_per_thread
    h_input = random_int(items_per_tile, dtype_np)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(items_per_tile, dtype=dtype_np)
    kernel[1, threads_per_block](d_input, d_output)
    cuda.synchronize()

    output = d_output.copy_to_host()

    # Calculate expected results with prefix of `initial_value`.
    if mode == "inclusive":
        # For inclusive scan with prefix, we need to add the prefix to all
        # elements.
        reference = np.zeros_like(h_input)
        running_total = initial_value
        for i in range(items_per_tile):
            running_total = running_total + h_input[i]
            reference[i] = running_total
    else:
        # For exclusive scan with prefix, the first element gets the prefix.
        reference = np.zeros_like(h_input)
        running_total = initial_value
        for i in range(items_per_tile):
            reference[i] = running_total
            running_total = running_total + h_input[i]

    for i in range(items_per_tile):
        assert (
            output[i] == reference[i]
        ), f"Mismatch at index {i}: {output[i]} != {reference[i]}"

    sig = (T[::1], T[::1])
    sass = kernel.inspect_sass(sig)

    assert "LDL" not in sass
    assert "STL" not in sass
