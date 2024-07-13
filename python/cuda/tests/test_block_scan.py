# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pynvjitlink import patch
import cuda.cooperative.experimental as cudax
import numba
from helpers import random_int, NUMBA_TYPES_TO_NP
from numba.core.extending import (lower_builtin, make_attribute_wrapper,
                                  models, register_model, type_callable,
                                  typeof_impl)
from numba.core import cgutils
import numpy as np
import pytest
from numba import cuda, types
import numba

numba.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0


patch.patch_numba_linker(lto=True)


@pytest.mark.parametrize('T', [types.uint32, types.uint64])
@pytest.mark.parametrize('threads_in_block', [32, 64, 128, 256, 512, 1024])
@pytest.mark.parametrize('items_per_thread', [1, 2, 3, 4])
def test_block_exclusive_sum(T, threads_in_block, items_per_thread):
    block_exclusive_sum = cudax.block.exclusive_sum(dtype=T,
                                                    threads_in_block=threads_in_block,
                                                    items_per_thread=items_per_thread)
    temp_storage_bytes = block_exclusive_sum.temp_storage_bytes

    @cuda.jit(link=block_exclusive_sum.files)
    def kernel(input, output):
        tid = cuda.threadIdx.x
        temp_storage = cuda.shared.array(shape=temp_storage_bytes, dtype='uint8')
        thread_data = cuda.local.array(shape=items_per_thread, dtype=dtype)
        for i in range(items_per_thread):
            thread_data[i] = input[tid * items_per_thread + i]
        block_exclusive_sum(temp_storage, thread_data, thread_data)
        for i in range(items_per_thread):
            output[tid * items_per_thread + i] = thread_data[i]

    dtype = NUMBA_TYPES_TO_NP[T]
    items_per_tile = threads_in_block * items_per_thread
    h_input = random_int(items_per_tile, dtype)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(items_per_tile, dtype=dtype)
    kernel[1, threads_in_block](d_input, d_output)
    cuda.synchronize()

    output = d_output.copy_to_host()
    reference = np.cumsum(h_input) - h_input
    for i in range(items_per_tile):
        assert output[i] == reference[i]

    sig = (T[::1], T[::1])
    sass = kernel.inspect_sass(sig)

    assert 'LDL' not in sass
    assert 'STL' not in sass


class BlockPrefixCallbackOp:
    def __init__(self, running_total):
        self.running_total = running_total

    def __call__(self_ptr, block_aggregate):
        old_prefix = self_ptr[0].running_total
        self_ptr[0] = BlockPrefixCallbackOp(old_prefix + block_aggregate)
        return old_prefix


class BlockPrefixCallbackOpType(types.Type):
    def __init__(self):
        super().__init__(name='BlockPrefixCallbackOp')


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
        members = [('running_total', types.int64)]
        models.StructModel.__init__(self, dmm, fe_type, members)


make_attribute_wrapper(BlockPrefixCallbackOpType,
                       'running_total', 'running_total')


@lower_builtin(BlockPrefixCallbackOp, types.Integer)
def impl_block_prefix_callback_op(context, builder, sig, args):
    typ = sig.return_type
    [running_total] = args
    state = cgutils.create_struct_proxy(typ)(context, builder)
    state.running_total = running_total
    return state._getvalue()



@pytest.mark.parametrize('threads_in_block', [32, 64, 128, 256, 512, 1024])
@pytest.mark.parametrize('items_per_thread', [1, 2, 3, 4])
def test_block_exclusive_sum_prefix(threads_in_block, items_per_thread):
    tile_items = threads_in_block * items_per_thread
    segment_size = 2 * 1024
    num_segments = 128
    num_elements = segment_size * num_segments

    prefix_op = cudax.StatefulFunction(BlockPrefixCallbackOp,
                                      block_prefix_callback_op_type)
    block_exclusive_sum = cudax.block.exclusive_sum(dtype=numba.types.int32,
                                                    threads_in_block=threads_in_block,
                                                    items_per_thread=items_per_thread,
                                                    prefix_op=prefix_op)
    temp_storage_bytes = block_exclusive_sum.temp_storage_bytes

    @cuda.jit(link=block_exclusive_sum.files)
    def kernel(input, output):
        segment_offset = cuda.blockIdx.x * segment_size
        temp_storage = cuda.shared.array(
            shape=temp_storage_bytes, dtype='uint8')
        block_prefix_op = cuda.local.array(
            shape=1, dtype=block_prefix_callback_op_type)
        block_prefix_op[0] = BlockPrefixCallbackOp(0)
        thread_input = cuda.local.array(shape=items_per_thread, dtype='int32')
        thread_output = cuda.local.array(shape=items_per_thread, dtype='int32')

        tile_offset = 0

        while tile_offset < segment_size:
            for item in range(items_per_thread):
                item_offset = tile_offset + cuda.threadIdx.x * items_per_thread + item
                thread_input[item] = input[segment_offset +
                                           item_offset] if item_offset < segment_size else 0

            block_exclusive_sum(temp_storage, thread_input,
                                thread_output, block_prefix_op)

            for item in range(items_per_thread):
                item_offset = tile_offset + cuda.threadIdx.x * items_per_thread + item
                if item_offset < segment_size:
                    output[segment_offset + item_offset] = thread_output[item]

            tile_offset += tile_items

    h_input = np.arange(num_elements, dtype='int32')
    d_input = cuda.to_device(h_input)
    d_output = cuda.to_device(np.zeros(num_elements, dtype='int32'))
    kernel[num_segments, threads_in_block](d_input, d_output)
    cuda.synchronize()
    h_output = d_output.copy_to_host()
    h_reference = np.zeros(segment_size * num_segments, dtype='int32')
    for sid in range(num_segments):
        h_reference[sid * segment_size] = 0
        for i in range(1, segment_size):
            h_reference[sid * segment_size + i] = h_reference[sid *
                                                              segment_size + i - 1] + h_input[sid * segment_size + i - 1]

    for sid in range(num_segments):
        for i in range(segment_size):
            if h_output[sid * segment_size + i] != h_reference[sid * segment_size + i]:
                print(sid, i, h_output[sid * segment_size + i],
                      h_reference[sid * segment_size + i])

    sig = (types.int32[::1], types.int32[::1])
    sass = kernel.inspect_sass(sig)

    assert 'LDL' not in sass
    assert 'STL' not in sass
