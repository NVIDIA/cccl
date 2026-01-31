# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from functools import reduce
from operator import mul

import numba
import numpy as np
from helpers import row_major_tid
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
from cuda.coop.block import BlockDiscontinuityType, BlockExchangeType

numba.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0


@cuda.jit(device=True)
def flag_op(lhs, rhs):
    return numba.int32(1 if lhs != rhs else 0)


class PrefixState:
    def __init__(self, running_total, call_count):
        self.running_total = running_total
        self.call_count = call_count

    def __call__(self_ptr, block_aggregate):
        old_prefix = self_ptr[0].running_total
        call_count = self_ptr[0].call_count + 1
        self_ptr[0] = PrefixState(old_prefix + block_aggregate, call_count)
        return old_prefix


class PrefixStateType(types.Type):
    def __init__(self):
        super().__init__(name="PrefixState")


prefix_state_type = PrefixStateType()


@typeof_impl.register(PrefixState)
def typeof_prefix_state(val, c):
    return prefix_state_type


@type_callable(PrefixState)
def type_prefix_state(context):
    def typer(running_total, call_count):
        if isinstance(running_total, types.Integer) and isinstance(
            call_count, types.Integer
        ):
            return prefix_state_type

    return typer


@register_model(PrefixStateType)
class PrefixStateModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("running_total", types.int64),
            ("call_count", types.int64),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


make_attribute_wrapper(PrefixStateType, "running_total", "running_total")
make_attribute_wrapper(PrefixStateType, "call_count", "call_count")


@lower_builtin(PrefixState, types.Integer, types.Integer)
def impl_prefix_state(context, builder, sig, args):
    typ = sig.return_type
    running_total, call_count = args
    state = cgutils.create_struct_proxy(typ)(context, builder)
    state.running_total = running_total
    state.call_count = call_count
    return state._getvalue()


class KeyPair:
    def __init__(self, key, tie):
        self.key = key
        self.tie = tie

    def construct(this):
        this[0] = KeyPair(numba.int32(0), numba.int32(0))

    def assign(this, that):
        this[0] = KeyPair(that[0].key, that[0].tie)


class KeyPairType(types.Type):
    def __init__(self):
        super().__init__(name="KeyPair")


keypair_type = KeyPairType()
keypair_type.methods = {
    "construct": KeyPair.construct,
    "assign": KeyPair.assign,
}
KEYPAIR_METHODS = keypair_type.methods


@typeof_impl.register(KeyPair)
def typeof_keypair(val, c):
    return keypair_type


@type_callable(KeyPair)
def type_keypair(context):
    def typer(key, tie):
        if isinstance(key, types.Integer) and isinstance(tie, types.Integer):
            return keypair_type

    return typer


@register_model(KeyPairType)
class KeyPairModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [("key", types.int32), ("tie", types.int32)]
        models.StructModel.__init__(self, dmm, fe_type, members)


make_attribute_wrapper(KeyPairType, "key", "key")
make_attribute_wrapper(KeyPairType, "tie", "tie")


@lower_builtin(KeyPair, types.Integer, types.Integer)
def impl_keypair(context, builder, sig, args):
    typ = sig.return_type
    key, tie = args
    state = cgutils.create_struct_proxy(typ)(context, builder)
    state.key = key
    state.tie = tie
    return state._getvalue()


@cuda.jit(device=True)
def keypair_less(a, b):
    a_val = a[0]
    b_val = b[0]
    if a_val.key < b_val.key:
        return True
    if a_val.key > b_val.key:
        return False
    return a_val.tie < b_val.tie


def test_block_exchange_discontinuity_weird_block_dims():
    threads_per_block = (16, 4, 2)
    items_per_thread = 2
    num_threads = reduce(mul, threads_per_block)
    total_items = num_threads * items_per_thread

    striped_to_blocked = BlockExchangeType.StripedToBlocked

    @cuda.jit
    def kernel(d_in, d_out, d_flags):
        tid = row_major_tid()
        thread_data = cuda.local.array(items_per_thread, dtype=numba.int32)

        for i in range(items_per_thread):
            idx = tid + i * num_threads
            thread_data[i] = d_in[idx]

        coop.block.exchange(
            thread_data,
            items_per_thread=items_per_thread,
            block_exchange_type=striped_to_blocked,
        )

        flags = cuda.local.array(items_per_thread, dtype=numba.int32)
        coop.block.discontinuity(
            thread_data,
            flags,
            flag_op=flag_op,
            block_discontinuity_type=BlockDiscontinuityType.HEADS,
        )

        for i in range(items_per_thread):
            out_idx = tid * items_per_thread + i
            d_out[out_idx] = thread_data[i]
            d_flags[out_idx] = flags[i]

    h_input = (np.arange(total_items, dtype=np.int32) // 3).astype(np.int32)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array_like(d_input)
    d_flags = cuda.device_array(total_items, dtype=np.int32)

    kernel[1, threads_per_block](d_input, d_output, d_flags)
    cuda.synchronize()

    h_output = d_output.copy_to_host()
    h_flags = d_flags.copy_to_host()

    h_reference = h_input.copy()
    h_flags_ref = np.zeros_like(h_flags)
    h_flags_ref[0] = 1
    for idx in range(1, total_items):
        h_flags_ref[idx] = 1 if h_reference[idx] != h_reference[idx - 1] else 0

    np.testing.assert_array_equal(h_output, h_reference)
    np.testing.assert_array_equal(h_flags, h_flags_ref)


def test_block_histogram_weird_block_dims():
    threads_per_block = (32, 2, 1)
    items_per_thread = 2
    bins = 8
    num_threads = reduce(mul, threads_per_block)
    total_items = num_threads * items_per_thread

    @cuda.jit
    def kernel(d_in, d_out):
        tid = row_major_tid()
        thread_samples = cuda.local.array(items_per_thread, dtype=numba.int32)
        smem_histogram = cuda.shared.array(bins, numba.int32)

        for i in range(items_per_thread):
            thread_samples[i] = d_in[tid * items_per_thread + i]

        histo = coop.block.histogram(thread_samples, smem_histogram)
        histo.init()
        cuda.syncthreads()
        histo.composite(thread_samples)
        cuda.syncthreads()

        if tid < bins:
            d_out[tid] = smem_histogram[tid]

    h_input = (np.arange(total_items, dtype=np.int32) * 3) % bins
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(bins, dtype=np.int32)

    kernel[1, threads_per_block](d_input, d_output)
    cuda.synchronize()

    h_output = d_output.copy_to_host()
    h_reference = np.zeros(bins, dtype=np.int32)
    for value in h_input:
        h_reference[value] += 1

    np.testing.assert_array_equal(h_output, h_reference)


def test_block_scan_stateful_prefix_op_grid_stride():
    threads_per_block = 128
    items_per_thread = 1
    items_per_block = threads_per_block * items_per_thread
    num_tiles = 4
    total_items = items_per_block * num_tiles

    prefix_op = coop.StatefulFunction(
        PrefixState,
        prefix_state_type,
        name="prefix_state",
    )

    block_scan = coop.block.scan(
        dtype=numba.int32,
        threads_per_block=threads_per_block,
        items_per_thread=items_per_thread,
        mode="exclusive",
        scan_op="+",
        block_prefix_callback_op=prefix_op,
    )

    @cuda.jit
    def kernel(d_in, d_out, d_calls):
        tid = cuda.threadIdx.x
        block_prefix_op = coop.local.array(1, dtype=prefix_state_type)
        block_prefix_op[0] = PrefixState(0, 0)

        block_offset = 0
        while block_offset < d_in.size:
            thread_data = coop.local.array(items_per_thread, dtype=d_in.dtype)
            thread_data[0] = d_in[block_offset + tid]

            block_scan(
                thread_data,
                thread_data,
                block_prefix_callback_op=block_prefix_op,
            )

            d_out[block_offset + tid] = thread_data[0]
            block_offset += items_per_block
            cuda.syncthreads()

        if tid == 0:
            d_calls[0] = block_prefix_op[0].call_count

    h_input = np.random.randint(0, 4, total_items, dtype=np.int32)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array_like(d_input)
    d_calls = cuda.device_array(1, dtype=np.int64)

    kernel[1, threads_per_block](d_input, d_output, d_calls)
    cuda.synchronize()

    h_output = d_output.copy_to_host()
    h_calls = d_calls.copy_to_host()

    h_reference = np.empty_like(h_input)
    running = 0
    for idx, val in enumerate(h_input):
        h_reference[idx] = running
        running += val

    np.testing.assert_array_equal(h_output, h_reference)
    assert h_calls[0] == num_tiles


def test_block_merge_sort_custom_type():
    threads_per_block = 64
    items_per_thread = 2
    num_threads = threads_per_block
    total_items = num_threads * items_per_thread

    block_merge_sort = coop.block.merge_sort_keys(
        keypair_type,
        threads_per_block,
        items_per_thread,
        keypair_less,
        methods=keypair_type.methods,
    )

    @cuda.jit
    def kernel(d_keys, d_ties, d_out_keys, d_out_ties):
        tid = cuda.threadIdx.x
        thread_keys = coop.local.array(items_per_thread, dtype=keypair_type)
        for i in range(items_per_thread):
            idx = tid * items_per_thread + i
            thread_keys[i] = KeyPair(d_keys[idx], d_ties[idx])

        block_merge_sort(thread_keys, items_per_thread, keypair_less)

        for i in range(items_per_thread):
            idx = tid * items_per_thread + i
            d_out_keys[idx] = thread_keys[i].key
            d_out_ties[idx] = thread_keys[i].tie

    h_keys = np.random.randint(0, 8, total_items, dtype=np.int32)
    h_ties = np.random.randint(0, 16, total_items, dtype=np.int32)
    d_keys = cuda.to_device(h_keys)
    d_ties = cuda.to_device(h_ties)
    d_out_keys = cuda.device_array_like(d_keys)
    d_out_ties = cuda.device_array_like(d_ties)

    kernel[1, threads_per_block](d_keys, d_ties, d_out_keys, d_out_ties)
    cuda.synchronize()

    h_out_keys = d_out_keys.copy_to_host()
    h_out_ties = d_out_ties.copy_to_host()

    h_reference = sorted(list(zip(h_keys, h_ties)))
    ref_keys = np.array([k for k, _ in h_reference], dtype=np.int32)
    ref_ties = np.array([t for _, t in h_reference], dtype=np.int32)

    np.testing.assert_array_equal(h_out_keys, ref_keys)
    np.testing.assert_array_equal(h_out_ties, ref_ties)
