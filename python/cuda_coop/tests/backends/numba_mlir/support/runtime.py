# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

from __future__ import annotations

import numpy as np
import pytest

cuda = pytest.importorskip("numba_cuda_mlir.cuda")
pytest.importorskip("numba_cuda_mlir")

import numba_cuda_mlir.models as mlir_models
from numba_cuda_mlir import types
from numba_cuda_mlir._mlir import ir as mlir_ir
from numba_cuda_mlir._mlir.dialects import llvm
from numba_cuda_mlir.extending import lowering_registry
from numba_cuda_mlir.lowering_utilities import convert
from numba_cuda_mlir.models import register_model as register_mlir_model
from numba_cuda_mlir.numba_cuda.extending import (
    make_attribute_wrapper,
    type_callable,
    typeof_impl,
)
from numba_cuda_mlir.numba_cuda.extending import models as cuda_models
from numba_cuda_mlir.numba_cuda.extending import (
    register_model as register_cuda_model,
)

import cuda.coop.numba_mlir as coop

_CUDA_RECORD_MODEL = getattr(cuda_models, "Struct" + "Model")
_MLIR_RECORD_MODEL = getattr(mlir_models, "Struct" + "Model")

THREADS = 32
ITEMS_PER_THREAD = 2


class NumbaMlirRunningPrefix:
    def __call__(self_ptr, block_aggregate):
        old_prefix = self_ptr[0]
        self_ptr[0] = old_prefix + block_aggregate
        return old_prefix


NUMBA_MLIR_PREFIX_CALLBACK_OP = coop.StatefulFunction(
    NumbaMlirRunningPrefix,
    types.int32,
    name="numba_mlir_running_prefix",
)


class NumbaMlirKeyPair:
    def __init__(self, key, tie):
        self.key = key
        self.tie = tie

    def construct(this):
        this[0] = NumbaMlirKeyPair(types.int32(0), types.int32(0))

    def assign(this, that):
        this[0] = NumbaMlirKeyPair(that[0].key, that[0].tie)


class NumbaMlirKeyPairType(types.Type):
    def __init__(self):
        super().__init__(name="NumbaMlirKeyPair")

    @property
    def key(self):
        return self.__class__


numba_mlir_keypair_type = NumbaMlirKeyPairType()


@typeof_impl.register(NumbaMlirKeyPair)
def typeof_numba_mlir_keypair(val, c):
    return numba_mlir_keypair_type


@type_callable(NumbaMlirKeyPair)
def type_numba_mlir_keypair(context):
    def typer(key, tie):
        if isinstance(key, types.Integer) and isinstance(tie, types.Integer):
            return numba_mlir_keypair_type

    return typer


@register_cuda_model(NumbaMlirKeyPairType)
class NumbaMlirKeyPairCudaRecordModel(_CUDA_RECORD_MODEL):
    def __init__(self, dmm, fe_type):
        members = [("key", types.int32), ("tie", types.int32)]
        super().__init__(dmm, fe_type, members)


@register_mlir_model(NumbaMlirKeyPairType)
class NumbaMlirKeyPairMlirRecordModel(_MLIR_RECORD_MODEL):
    def __init__(self, dmm, fe_type):
        members = [("key", types.int32), ("tie", types.int32)]
        super().__init__(dmm, fe_type, members)


make_attribute_wrapper(NumbaMlirKeyPairType, "key", "key")
make_attribute_wrapper(NumbaMlirKeyPairType, "tie", "tie")


@lowering_registry.lower(NumbaMlirKeyPair, types.Integer, types.Integer)
def lower_numba_mlir_keypair(builder, target, args, kwargs):
    key = builder.load_var(args[0])
    tie = builder.load_var(args[1])
    i32 = mlir_ir.IntegerType.get_signless(32)
    key = convert(key, i32)
    tie = convert(tie, i32)
    struct_ty = builder.get_mlir_type(numba_mlir_keypair_type)
    undef = llvm.UndefOp(struct_ty)
    with_key = llvm.insertvalue(
        container=undef,
        value=key,
        position=mlir_ir.DenseI64ArrayAttr.get([0]),
    )
    result = llvm.insertvalue(
        container=with_key,
        value=tie,
        position=mlir_ir.DenseI64ArrayAttr.get([1]),
    )
    builder.store_var(target, result)


def _striped_to_blocked_reference(values, threads=THREADS, items=ITEMS_PER_THREAD):
    expected = np.empty_like(values)
    for blocked_idx in range(threads * items):
        source_thread = blocked_idx % threads
        source_item = blocked_idx // threads
        expected[blocked_idx] = values[source_thread + source_item * threads]
    return expected


def _add(a, b):
    return a + b


@cuda.jit(device=True)
def _prefix_with_block_aggregate(block_aggregate):
    return block_aggregate


@cuda.jit(device=True)
def _less(a, b):
    return a < b


@cuda.jit(device=True)
def _complex_real_greater(a, b):
    return a.real > b.real


@cuda.jit(device=True)
def _subtract(a, b):
    return a - b


@cuda.jit(device=True)
def _different(a, b):
    return a != b


def _validate_ranks(keys, ranks, begin_bit, end_bit, descending=False):
    radix_bits = end_bit - begin_bit
    mask = (1 << radix_bits) - 1
    digits = (keys >> begin_bit) & mask
    max_digit = 1 << radix_bits

    counts = np.zeros(max_digit, dtype=np.int32)
    for digit in digits:
        counts[int(digit)] += 1

    offsets = {}
    prefix = 0
    digit_iter = range(max_digit - 1, -1, -1) if descending else range(max_digit)
    for digit in digit_iter:
        offsets[digit] = prefix
        prefix += int(counts[digit])

    np.testing.assert_array_equal(
        np.sort(ranks), np.arange(len(ranks), dtype=ranks.dtype)
    )
    for idx, rank in enumerate(ranks):
        digit = int(digits[idx])
        assert offsets[digit] <= rank < offsets[digit] + int(counts[digit])


def _exclusive_digit_prefix_reference(
    keys, begin_bit, end_bit, block_threads=THREADS, descending=False
):
    radix_bits = end_bit - begin_bit
    radix_digits = 1 << radix_bits
    bins_per_thread = max(1, (radix_digits + block_threads - 1) // block_threads)
    digits = (keys >> begin_bit) & (radix_digits - 1)

    counts = np.zeros(radix_digits, dtype=np.int32)
    for digit in digits:
        counts[int(digit)] += 1

    prefix = np.zeros(radix_digits, dtype=np.int32)
    running = 0
    digit_iter = range(radix_digits - 1, -1, -1) if descending else range(radix_digits)
    for digit in digit_iter:
        prefix[digit] = running
        running += int(counts[digit])

    expected = np.full((block_threads, bins_per_thread), -1, dtype=np.int32)
    for tid in range(block_threads):
        for track in range(bins_per_thread):
            bin_idx = tid * bins_per_thread + track
            if block_threads == radix_digits or bin_idx < radix_digits:
                expected[tid, track] = prefix[bin_idx]
    return expected


def _make_topk_stress_inputs(tile_size, num_tiles, total_items):
    keys = np.empty(total_items, dtype=np.int32)
    values = np.empty(total_items, dtype=np.int32)
    for tile_idx in range(num_tiles):
        start = tile_idx * tile_size
        end = min(start + tile_size, total_items)
        local = np.arange(end - start, dtype=np.int32)
        keys[start:end] = (
            ((local * np.int32(37)) + np.int32(11)) % np.int32(tile_size)
        ) + np.int32(tile_idx * 4096)
        values[start:end] = np.int32(tile_idx * 10000) + local * np.int32(13)
    return keys, values


def _make_topk_rank_flags(tile_size, num_tiles, total_items, k):
    flags = np.zeros(num_tiles * tile_size, dtype=np.int32)
    for tile_idx in range(num_tiles):
        start = tile_idx * tile_size
        valid = min(tile_size, total_items - start)
        flags[start : start + min(k, valid)] = np.int32(1)
    return flags


def _host_pair_checksum(keys, values):
    checksum = np.int64(0)
    for key, value in zip(keys, values, strict=True):
        checksum += np.int64(int(key)) * np.int64(1315423911) + np.int64(
            int(value)
        ) * np.int64(2654435761)
    return checksum


def _assert_topk_stress_output(
    h_keys,
    h_values,
    h_keys_out,
    h_values_out,
    h_ranks_out,
    h_checksums,
    tile_size,
    num_tiles,
    total_items,
    k,
):
    for tile_idx in range(num_tiles):
        start = tile_idx * tile_size
        end = min(start + tile_size, total_items)
        valid = end - start
        runtime_k = min(k, valid)

        tile_keys = h_keys[start:end]
        tile_values = h_values[start:end]
        expected_indices = np.argsort(tile_keys)[-runtime_k:]
        expected_pairs = sorted(
            zip(tile_keys[expected_indices], tile_values[expected_indices], strict=True)
        )
        actual_pairs = sorted(
            zip(
                h_keys_out[start : start + runtime_k],
                h_values_out[start : start + runtime_k],
                strict=True,
            )
        )
        assert actual_pairs == expected_pairs

        expected_ranks = np.minimum(
            np.arange(tile_size, dtype=np.int32),
            np.int32(runtime_k),
        )
        np.testing.assert_array_equal(
            h_ranks_out[start : start + tile_size],
            expected_ranks,
        )

        expected_keys = np.asarray([key for key, _ in expected_pairs], dtype=np.int32)
        expected_values = np.asarray(
            [value for _, value in expected_pairs], dtype=np.int32
        )
        assert h_checksums[tile_idx] == _host_pair_checksum(
            expected_keys,
            expected_values,
        )
