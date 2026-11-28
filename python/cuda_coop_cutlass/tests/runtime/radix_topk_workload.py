# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Use the radix TopK block scan and pair sort in one provider bundle."""

import sys

import cutlass
import numpy as np
from cutlass import Float32, Int32, Int64, cute

import cuda.coop.cutlass as coop
from tests.runtime_support import device_array

_BINS = 256
_COLUMNS = 2048
_K = 64


@cute.kernel
def _radix_topk(
    source: cute.Pointer, values: cute.Pointer, indices: cute.Pointer
):
    tid, _, _ = cute.arch.thread_idx()
    row, _, _ = cute.arch.block_idx()
    data = cute.make_tensor(source + row * _COLUMNS, (_COLUMNS,))
    out = cute.make_tensor(values + row * _K, (_K,))
    idx = cute.make_tensor(indices + row * _K, (_K,))
    smem = cutlass.memory.SmemAllocator()
    bins = smem.allocate_tensor(Int32, cute.make_layout(_BINS))
    survivors = smem.allocate_tensor(Int32, cute.make_layout(_K))

    # The FP32 fixture contains byte-valued keys, so one radix digit suffices.
    bins[tid] = 0
    cute.arch.barrier()
    for item in cutlass.range_constexpr(_COLUMNS // _BINS):
        digit = Int32(data[item * _BINS + tid])
        cute.arch.atomic_add(bins.iterator + digit, Int32(1))
    cute.arch.barrier()

    # In ascending bin order, keep the last K slots. Within the cutoff bin,
    # keep the earliest input indices, even when more than K values tie.
    count = bins[tid]
    end = coop.inclusive_sum(coop.this_block(), count)
    slot = max(end - count - (_COLUMNS - _K), Int32(0))
    stop = end - (_COLUMNS - _K)
    if slot < stop:
        for column in range(_COLUMNS):
            if slot < stop and Int32(data[column]) == tid:
                survivors[slot] = column
                slot += 1
    cute.arch.barrier()

    keys = coop.ThreadData(items_per_thread=1)
    payload = coop.ThreadData(items_per_thread=1)
    sentinel = Int64(-(1 << 63))
    keys[0] = sentinel
    payload[0] = Float32(0)
    if tid < _K:
        column = survivors[tid]
        payload[0] = data[column]
        keys[0] = (Int64(payload[0]) << 32) | (Int64(~column) & 0xFFFFFFFF)
    keys, payload = coop.merge_sort_pairs(
        coop.this_block(),
        keys,
        payload,
        descending=True,
        valid_items=_K,
        oob_default=sentinel,
    )
    if tid < _K:
        out[tid] = payload[0]
        idx[tid] = Int64(~Int32(keys[0]))


@cute.jit
def _launch(
    source: cute.Pointer,
    values: cute.Pointer,
    indices: cute.Pointer,
    rows: Int32,
):
    _radix_topk(source, values, indices).launch(grid=rows, block=_BINS)


def run(rows):
    rng = np.random.default_rng(7)
    source = rng.integers(0, _BINS, (rows, _COLUMNS)).astype(np.float32)
    values = np.full((rows, _K), np.nan, dtype=np.float32)
    indices = np.full((rows, _K), -1, dtype=np.int64)
    with (
        device_array(source) as x,
        device_array(values) as v,
        device_array(indices) as i,
    ):
        _launch(x, v, i, Int32(rows))
    expected = np.argsort(-source, axis=1, kind="stable")[:, :_K]
    np.testing.assert_array_equal(indices, expected)
    np.testing.assert_array_equal(
        values, np.take_along_axis(source, expected, axis=1)
    )


if __name__ == "__main__":
    run(int(sys.argv[1]))
