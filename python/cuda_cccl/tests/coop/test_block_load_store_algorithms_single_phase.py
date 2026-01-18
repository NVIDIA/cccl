# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Single-phase BlockLoad/BlockStore algorithm coverage."""

import numba
import numpy as np
import pytest
from helpers import random_int
from numba import cuda

from cuda import coop

numba.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0


def _run_load_store_kernel(load_algo, store_algo, items_per_thread, offset=0):
    threads_per_block = 128
    total_items = threads_per_block * items_per_thread

    h_input = random_int(total_items + offset + 1, np.int32)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array_like(d_input)

    @cuda.jit
    def kernel(d_in, d_out):
        thread_data = coop.local.array(items_per_thread, dtype=d_in.dtype)
        coop.block.load(
            d_in,
            thread_data,
            items_per_thread=items_per_thread,
            algorithm=load_algo,
        )
        coop.block.store(
            d_out,
            thread_data,
            items_per_thread=items_per_thread,
            algorithm=store_algo,
        )

    kernel[1, threads_per_block](d_input[offset:], d_output[offset:])
    cuda.synchronize()

    h_output = d_output.copy_to_host()
    np.testing.assert_array_equal(
        h_output[offset : offset + total_items],
        h_input[offset : offset + total_items],
    )


@pytest.mark.parametrize(
    "load_algo",
    [
        coop.BlockLoadAlgorithm.TRANSPOSE,
        coop.BlockLoadAlgorithm.WARP_TRANSPOSE,
        coop.BlockLoadAlgorithm.WARP_TRANSPOSE_TIMESLICED,
    ],
)
def test_block_load_shared_memory_algorithms(load_algo):
    _run_load_store_kernel(load_algo, coop.BlockStoreAlgorithm.DIRECT, 4)


@pytest.mark.parametrize(
    "store_algo",
    [
        coop.BlockStoreAlgorithm.TRANSPOSE,
        coop.BlockStoreAlgorithm.WARP_TRANSPOSE,
        coop.BlockStoreAlgorithm.WARP_TRANSPOSE_TIMESLICED,
    ],
)
def test_block_store_shared_memory_algorithms(store_algo):
    _run_load_store_kernel(coop.BlockLoadAlgorithm.DIRECT, store_algo, 4)


@pytest.mark.parametrize("offset", [0, 1])
def test_block_load_vectorize_alignment(offset):
    _run_load_store_kernel(
        coop.BlockLoadAlgorithm.VECTORIZE,
        coop.BlockStoreAlgorithm.DIRECT,
        4,
        offset=offset,
    )


@pytest.mark.parametrize("offset", [0, 1])
def test_block_store_vectorize_alignment(offset):
    _run_load_store_kernel(
        coop.BlockLoadAlgorithm.DIRECT,
        coop.BlockStoreAlgorithm.VECTORIZE,
        4,
        offset=offset,
    )


def test_block_load_vectorize_odd_items_per_thread():
    _run_load_store_kernel(
        coop.BlockLoadAlgorithm.VECTORIZE,
        coop.BlockStoreAlgorithm.DIRECT,
        3,
    )


def test_block_store_vectorize_odd_items_per_thread():
    _run_load_store_kernel(
        coop.BlockLoadAlgorithm.DIRECT,
        coop.BlockStoreAlgorithm.VECTORIZE,
        3,
    )
