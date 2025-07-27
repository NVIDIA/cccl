# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import textwrap
from functools import reduce
from operator import mul

import numba
import numpy as np
import pytest
from helpers import (
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

import cuda.cccl.cooperative.experimental as coop

# coop._init_extension()

numba.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0

# ffi = cffi.FFI()

prefix_op_src = cuda.CUSource(
    textwrap.dedent("""
    extern "C"
    __device__
    struct BlockPrefixCallbackOp {
        int running_total;
        int call_count;
    };

    extern "C"
    __device__
    int
    block_prefix_callback_op_dbg(
        struct BlockPrefixCallbackOp* op,
        int block_aggregate
        )
    {
        printf("[blockIdx.x=%d/threadIdx.x=%d; tid: %d] "
               "block_prefix_callback_op(op=%p): "
               "block_aggregate=%d, running_total=%d, call_count=%d\\n",
               blockIdx.x, threadIdx.x,
               blockIdx.x * blockDim.x + threadIdx.x,
               op,
               block_aggregate,
               op->running_total,
               op->call_count);
        return 0;
    }

    extern "C"
    __device__
    int
    block_prefix_callback_op(
        int* result,
        struct BlockPrefixCallbackOp* op,
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

# Define the function signature for device declaration
prefix_op_dbg_type = types.Record.make_c_struct(
    [
        ("running_total", types.int32),
        ("call_count", types.int32),
    ]
)

args = [types.CPointer(prefix_op_dbg_type), types.int32]
sig = types.void(
    types.CPointer(prefix_op_dbg_type),
    types.int32,
)

# Declare the external CUDA function
prefix_op_dbg = cuda.declare_device(
    "block_prefix_callback_op_dbg",
    sig,
    link=prefix_op_src,
)


class BlockPrefixCallbackOp:
    """
    A sample prefix callback operator that stores and updates a running total.
    """

    def __init__(self, running_total, call_count):
        self.running_total = running_total
        self.call_count = call_count

    def __call__(self_ptr, block_aggregate):
        # op_ptr = ffi.from_buffer(self_ptr)
        # Call the external device function
        # cast → pointer‑to‑struct
        bp_ptr = cuda.cast(
            self_ptr,
            types.CPointer(prefix_op_dbg_type),
        )
        prefix_op_dbg(bp_ptr, block_aggregate)

        # old_prefix = self_ptr[0].running_total
        # call_count = self_ptr[0].call_count + 1
        # new_running_total = old_prefix + block_aggregate
        # self_ptr[0] = BlockPrefixCallbackOp(new_running_total, call_count)
        return 0
        # return old_prefix


class BlockPrefixCallbackOpType(types.Type):
    def __init__(self):
        super().__init__(name="BlockPrefixCallbackOp")


block_prefix_callback_op_type = BlockPrefixCallbackOpType()


@typeof_impl.register(BlockPrefixCallbackOp)
def typeof_block_prefix_callback_op(val, c):
    return block_prefix_callback_op_type


@type_callable(BlockPrefixCallbackOp)
def type__block_prefix_callback_op(context):
    def typer(running_total, call_count):
        return block_prefix_callback_op_type
        valid_args = isinstance(running_total, types.Integer) and isinstance(
            call_count, types.Integer
        )
        if valid_args:
            return block_prefix_callback_op_type

    return typer


@register_model(BlockPrefixCallbackOpType)
class BlockPrefixCallbackOpModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("running_total", types.int64),
            ("call_count", types.int64),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


make_attribute_wrapper(BlockPrefixCallbackOpType, "running_total", "running_total")
make_attribute_wrapper(BlockPrefixCallbackOpType, "call_count", "call_count")


# @lower_builtin(BlockPrefixCallbackOp, types.VarArg(types.Any))
@lower_builtin(BlockPrefixCallbackOp, types.Integer, types.Integer)
def impl_block_prefix_callback_op(context, builder, sig, args):
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


def test_block_load_store_scan_simple6():
    # ffi = cffi.FFI()

    # CUDA C extension source
    @cuda.jit
    def kernel1(d_in, d_out1, d_out2):
        idx = cuda.grid(1)
        if idx >= d_in.size:
            return

        # Initialize local struct
        block_prefix_op = cuda.local.array(1, dtype=block_prefix_callback_op_type)
        block_prefix_op[0]["running_total"] = d_in[idx]
        block_prefix_op[0]["call_count"] = 0

        # op_ptr = ffi.from_buffer(block_prefix_op)

        # Call the external device function
        # d_out1[idx] = block_prefix_callback_op(op_ptr, cuda.threadIdx.x)
        # d_out2[idx] = block_prefix_callback_op(op_ptr, d_out1[idx])

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


def test_block_load_store_scan_simple1():
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
            dtype=block_prefix_callback_op_type,
        )
        block_prefix_op[0] = BlockPrefixCallbackOp(0, 0)
        # block_prefix_callback_op = BlockPrefixCallbackOp(0)

        # Load with padding
        coop.block.load(
            d_in[block_offset:],
            thread_in,
            items_per_thread=items_per_thread,
            algorithm=coop.BlockLoadAlgorithm.WARP_TRANSPOSE,
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
            algorithm=coop.BlockStoreAlgorithm.DIRECT,
            num_valid_items=num_valid_items,
        )

        block_offset += items_per_block * cuda.gridDim.x

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


def test_block_load_store_num_valid_items_with_single_phase_scan():
    @cuda.jit
    def kernel(d_in, d_out, items_per_thread, num_total_items):
        threads_per_block = cuda.blockDim.x * cuda.blockDim.y * cuda.blockDim.z
        items_per_block = items_per_thread * threads_per_block

        block_offset = cuda.blockIdx.x * items_per_block
        thread_offset = cuda.threadIdx.x * items_per_thread

        thread_data = coop.local.array(items_per_thread, dtype=d_in.dtype)

        block_prefix_callback_op = cuda.local.array(
            shape=1,
            dtype=block_prefix_callback_op_type,
        )
        block_prefix_callback_op[0] = BlockPrefixCallbackOp(0)

        # block_prefix_callback_op = BlockPrefixCallbackOp(0)

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
                algorithm=coop.BlockLoadAlgorithm.WARP_TRANSPOSE,
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
                algorithm=coop.BlockStoreAlgorithm.DIRECT,
                num_valid_items=num_valid_items,
            )

            block_offset += items_per_block * cuda.gridDim.x

            cuda.syncthreads()

    dtype = np.int32
    threads_per_block = 128
    num_total_items = 1000
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


@pytest.mark.parametrize("threads_per_block", [32, 128])
@pytest.mark.parametrize("items_per_thread", [1, 4])
@pytest.mark.parametrize("mode", ["exclusive"])
def test_block_sum_prefix_op(threads_per_block, items_per_thread, mode):
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
        BlockPrefixCallbackOp,
        block_prefix_callback_op_type,
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
        algorithm=coop.BlockScanAlgorithm.RAKING,
    )
    temp_storage_bytes = block_sum.temp_storage_bytes
    if temp_storage_bytes == 0:
        print("No temp storage required for block scan.")

    @cuda.jit
    def kernel(input_arr, output_arr, call_counts):
        segment_offset = cuda.blockIdx.x * segment_size
        # temp_storage = cuda.shared.array(shape=temp_storage_bytes, dtype=numba.uint8)
        block_prefix_op = cuda.local.array(shape=1, dtype=block_prefix_callback_op_type)
        block_prefix_op[0] = BlockPrefixCallbackOp(0, 0)
        # block_prefix_op = BlockPrefixCallbackOp(0, 0)
        # block_prefix_callback_op = cuda.local.array(
        #    shape=1,
        #    dtype=block_prefix_callback_op_type,
        # )
        # block_prefix_callback_op[0] = BlockPrefixCallbackOp(0)

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
                # thread_out[0] = block_sum(temp_storage, thread_in[0], block_prefix_op)
                pass
            else:
                coop.block.scan(
                    thread_in,
                    thread_out,
                    items_per_thread,
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

    # kernel[num_segments, threads_per_block](d_input, d_output)
    kernel[1, threads_per_block](d_input, d_output, d_call_counts)
    cuda.synchronize()

    h_output = d_output.copy_to_host()
    h_call_counts = d_call_counts.copy_to_host()
    ref = np.zeros_like(h_input)
    print(f"call_counts: {h_call_counts}")

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
    print(f"sum_equal: {sum_equal}")
    np.testing.assert_array_equal(h_output, ref)

    sig = (types.int32[::1], types.int32[::1])
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
