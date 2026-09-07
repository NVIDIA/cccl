# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

import numpy as np
import pytest

cuda = pytest.importorskip("numba_cuda_mlir.cuda")
pytest.importorskip("numba_cuda_mlir")

from numba_cuda_mlir import types

import cuda.coop.numba_mlir as coop
from cuda.coop.numba_mlir._block import BlockExchangeType
from cuda.coop.numba_mlir._warp import WarpExchangeType

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.usefixtures("numba_mlir_cuda_available"),
    pytest.mark.filterwarnings(
        "ignore::numba_cuda_mlir.numba_cuda.core.errors.NumbaPerformanceWarning"
    ),
]


def test_block_scan_thread_data_items_per_thread_mismatch_raises():
    @cuda.jit
    def kernel(d_out):
        items = coop.ThreadData(2, dtype=types.int32)
        scanned = coop.ThreadData(2, dtype=types.int32)
        coop._block.scan(
            items,
            scanned,
            dtype=types.int32,
            threads_per_block=32,
            items_per_thread=4,
        )
        d_out[cuda.threadIdx.x] = scanned[0]

    h_output = np.zeros(32, dtype=np.int32)
    with pytest.raises(
        Exception,
        match=(
            "coop single-phase 'scan' items_per_thread does not match "
            "the value inferred from coop.ThreadData"
        ),
    ):
        kernel[1, 32](h_output)


def test_warp_exchange_thread_data_items_per_thread_mismatch_raises():
    @cuda.jit
    def kernel(d_out):
        items = coop.ThreadData(4, dtype=types.int32)
        output_items = coop.ThreadData(4, dtype=types.int32)
        coop._warp.exchange(
            items,
            output_items,
            items_per_thread=5,
            warp_exchange_type=WarpExchangeType.StripedToBlocked,
            threads_in_warp=32,
        )
        d_out[cuda.threadIdx.x] = output_items[0]

    h_output = np.zeros(32, dtype=np.int32)
    with pytest.raises(
        Exception,
        match=(
            "coop single-phase 'warp_exchange' items_per_thread does not match "
            "the value inferred from coop.ThreadData"
        ),
    ):
        kernel[1, 32](h_output)


def test_thread_data_dtype_mismatch_across_primitives_raises():
    @cuda.jit
    def kernel(d_in):
        items = coop.ThreadData(2)
        output_items = coop.ThreadData(2, dtype=types.float32)
        coop._block.load(d_in, items)
        coop._block.exchange(
            items,
            output_items,
            block_exchange_type=BlockExchangeType.StripedToBlocked,
        )

    h_input = np.arange(64, dtype=np.int32)
    with pytest.raises(
        Exception,
        match="requires input/output arrays to have matching dtype",
    ):
        kernel[1, 32](h_input)


def test_temp_storage_explicit_size_too_small_raises():
    @cuda.jit
    def kernel(d_in, d_out):
        temp_storage = coop.TempStorage(1)
        items = coop.ThreadData(2, dtype=d_in.dtype)
        coop._block.load(
            d_in,
            items,
            algorithm="transpose",
            temp_storage=temp_storage,
        )
        coop._block.store(
            d_out,
            items,
            algorithm="transpose",
            temp_storage=temp_storage,
        )

    h_input = np.arange(64, dtype=np.int32)
    h_output = np.zeros_like(h_input)
    with pytest.raises(
        Exception,
        match="TempStorage size_in_bytes is smaller than required by primitive uses",
    ):
        kernel[1, 32](h_input, h_output)


def test_temp_storage_exclusive_auto_sync_true_raises():
    @cuda.jit
    def kernel(d_in, d_out):
        temp_storage = coop.TempStorage(sharing="exclusive", auto_sync=True)
        items = coop.ThreadData(2, dtype=d_in.dtype)
        coop._block.load(
            d_in,
            items,
            algorithm="transpose",
            temp_storage=temp_storage,
        )
        coop._block.store(
            d_out,
            items,
            algorithm="transpose",
            temp_storage=temp_storage,
        )

    h_input = np.arange(64, dtype=np.int32)
    h_output = np.zeros_like(h_input)
    with pytest.raises(
        Exception,
        match=("TempStorage with sharing='exclusive' does not support auto_sync=True"),
    ):
        kernel[1, 32](h_input, h_output)


def test_temp_storage_rejects_invalid_sharing():
    @cuda.jit
    def kernel(d_in, d_out):
        temp_storage = coop.TempStorage(sharing="bogus")
        items = coop.ThreadData(2, dtype=d_in.dtype)
        coop._block.load(
            d_in,
            items,
            algorithm="transpose",
            temp_storage=temp_storage,
        )
        coop._block.store(
            d_out,
            items,
            algorithm="transpose",
            temp_storage=temp_storage,
        )

    h_input = np.arange(64, dtype=np.int32)
    h_output = np.zeros_like(h_input)
    with pytest.raises(
        Exception,
        match="TempStorage sharing must be 'shared' or 'exclusive'",
    ):
        kernel[1, 32](h_input, h_output)


def test_temp_storage_getitem_and_keyword_duplicate_raises():
    @cuda.jit
    def kernel(d_in, d_out):
        temp_storage = coop.TempStorage()
        items = coop.ThreadData(2, dtype=d_in.dtype)
        coop._block.load[temp_storage](
            d_in,
            items,
            algorithm="transpose",
            temp_storage=temp_storage,
        )
        coop._block.store(d_out, items, algorithm="transpose")

    h_input = np.arange(64, dtype=np.int32)
    h_output = np.zeros_like(h_input)
    with pytest.raises(
        Exception,
        match="Duplicate coop single-phase 'load' runtime temp storage",
    ):
        kernel[1, 32](h_input, h_output)
