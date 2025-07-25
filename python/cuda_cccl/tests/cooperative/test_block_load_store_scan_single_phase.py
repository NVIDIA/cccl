# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


import numba
import numpy as np
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

    # Compute the correct prefix-sum on host to compare results
    h_reference = np.zeros_like(h_input)
    if len(h_input) > 0:
        h_reference[0] = 0
        h_reference[1:] = np.cumsum(h_input[:-1])

    # h_reference2 = np.cumsum(h_input)
    # h_reference3 = np.cumsum(h_input) - h_input

    h_output = d_output.copy_to_host()
    np.testing.assert_array_equal(h_output, h_reference)
