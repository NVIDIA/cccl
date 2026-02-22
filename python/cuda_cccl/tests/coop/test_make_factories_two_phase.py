# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numba
import numpy as np
import pytest
from numba import cuda

from cuda import coop
from cuda.coop._types import Invocable


def _binary_op(lhs, rhs):
    return lhs + rhs


def _compare_op(lhs, rhs):
    return lhs < rhs


def _difference_op(lhs, rhs):
    return lhs - rhs


def _flag_op(lhs, rhs):
    return lhs != rhs


@pytest.mark.parametrize(
    "factory",
    [
        lambda: coop.block.make_load(
            numba.int32, threads_per_block=64, items_per_thread=2
        ),
        lambda: coop.block.make_store(
            numba.int32, threads_per_block=64, items_per_thread=2
        ),
        lambda: coop.block.make_exchange(
            numba.int32, threads_per_block=64, items_per_thread=2
        ),
        lambda: coop.block.make_merge_sort_keys(
            numba.int32,
            threads_per_block=64,
            items_per_thread=2,
            compare_op=_compare_op,
        ),
        lambda: coop.block.make_merge_sort_pairs(
            numba.int32,
            numba.int32,
            threads_per_block=64,
            items_per_thread=2,
            compare_op=_compare_op,
        ),
        lambda: coop.block.make_radix_sort_keys(
            numba.int32,
            threads_per_block=64,
            items_per_thread=2,
        ),
        lambda: coop.block.make_radix_sort_keys_descending(
            numba.int32,
            threads_per_block=64,
            items_per_thread=2,
        ),
        lambda: coop.block.make_radix_sort_pairs(
            numba.int32,
            numba.int32,
            threads_per_block=64,
            items_per_thread=2,
        ),
        lambda: coop.block.make_radix_sort_pairs_descending(
            numba.int32,
            numba.int32,
            threads_per_block=64,
            items_per_thread=2,
        ),
        lambda: coop.block.make_radix_rank(
            numba.int32,
            threads_per_block=64,
            items_per_thread=2,
            begin_bit=0,
            end_bit=8,
        ),
        lambda: coop.block.make_reduce(
            numba.int32,
            threads_per_block=64,
            binary_op=_binary_op,
            items_per_thread=2,
        ),
        lambda: coop.block.make_sum(
            numba.int32, threads_per_block=64, items_per_thread=2
        ),
        lambda: coop.block.make_scan(
            numba.int32, threads_per_block=64, items_per_thread=2
        ),
        lambda: coop.block.make_exclusive_sum(
            numba.int32,
            threads_per_block=64,
            items_per_thread=2,
        ),
        lambda: coop.block.make_inclusive_sum(
            numba.int32,
            threads_per_block=64,
            items_per_thread=2,
        ),
        lambda: coop.block.make_exclusive_scan(
            numba.int32,
            threads_per_block=64,
            scan_op="+",
            items_per_thread=2,
        ),
        lambda: coop.block.make_inclusive_scan(
            numba.int32,
            threads_per_block=64,
            scan_op="+",
            items_per_thread=2,
        ),
        lambda: coop.block.make_adjacent_difference(
            numba.int32,
            threads_per_block=64,
            items_per_thread=1,
            difference_op=_difference_op,
        ),
        lambda: coop.block.make_discontinuity(
            numba.int32,
            threads_per_block=64,
            items_per_thread=1,
            flag_op=_flag_op,
        ),
        lambda: coop.block.make_shuffle(
            numba.int32, threads_per_block=64, items_per_thread=1
        ),
        lambda: coop.warp.make_load(
            numba.int32, items_per_thread=2, threads_in_warp=32
        ),
        lambda: coop.warp.make_store(
            numba.int32, items_per_thread=2, threads_in_warp=32
        ),
        lambda: coop.warp.make_exchange(
            numba.int32, items_per_thread=2, threads_in_warp=32
        ),
        lambda: coop.warp.make_reduce(numba.int32, _binary_op, threads_in_warp=32),
        lambda: coop.warp.make_sum(numba.int32, threads_in_warp=32),
        lambda: coop.warp.make_exclusive_sum(numba.int32, threads_in_warp=32),
        lambda: coop.warp.make_inclusive_sum(numba.int32, threads_in_warp=32),
        lambda: coop.warp.make_exclusive_scan(
            numba.int32, _binary_op, threads_in_warp=32
        ),
        lambda: coop.warp.make_inclusive_scan(
            numba.int32, _binary_op, threads_in_warp=32
        ),
        lambda: coop.warp.make_merge_sort_keys(
            numba.int32,
            items_per_thread=2,
            compare_op=_compare_op,
            threads_in_warp=32,
        ),
        lambda: coop.warp.make_merge_sort_pairs(
            numba.int32,
            numba.int32,
            items_per_thread=2,
            compare_op=_compare_op,
            threads_in_warp=32,
        ),
    ],
)
def test_make_factories_return_invocable(factory):
    instance = factory()
    assert isinstance(instance, Invocable)
    assert instance.files


def test_make_histogram_returns_stateful_instance():
    histo = coop.block.make_histogram(
        numba.uint8,
        numba.uint32,
        threads_per_block=64,
        items_per_thread=1,
    )
    assert hasattr(histo, "init")
    assert hasattr(histo, "composite")


def test_make_histogram_rejects_explicit_temp_storage():
    with pytest.raises(
        NotImplementedError,
        match="Explicit temp_storage is not yet supported for histogram.",
    ):
        coop.block.make_histogram(
            numba.uint8,
            numba.uint32,
            threads_per_block=64,
            items_per_thread=1,
            temp_storage=object(),
        )


def test_make_run_length_returns_stateful_instance():
    rle = coop.block.make_run_length(
        numba.int32,
        threads_per_block=64,
        runs_per_thread=1,
        decoded_items_per_thread=1,
    )
    assert hasattr(rle, "decode")


def test_make_block_factories_accept_dim_alias():
    load = coop.block.make_load(numba.int32, dim=64, items_per_thread=2)
    assert isinstance(load, Invocable)

    histo = coop.block.make_histogram(
        numba.uint8,
        numba.uint32,
        dim=64,
        items_per_thread=1,
    )
    assert hasattr(histo, "init")
    assert hasattr(histo, "composite")


def test_make_warp_sum_runs_in_kernel_without_explicit_link():
    warp_sum = coop.warp.make_sum(numba.int32)
    threads_in_warp = 32

    @cuda.jit
    def kernel(d_in, d_out):
        tid = cuda.threadIdx.x
        d_out[tid] = warp_sum(d_in[tid])

    h_input = np.ones(threads_in_warp, dtype=np.int32)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array_like(d_input)
    kernel[1, threads_in_warp](d_input, d_output)
    h_output = d_output.copy_to_host()

    assert h_output[0] == threads_in_warp
