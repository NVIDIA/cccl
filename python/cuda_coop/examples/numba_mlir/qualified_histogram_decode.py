# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Build a histogram and decode a run window with Numba-CUDA-MLIR."""

from __future__ import annotations

import numpy as np
from numba_cuda_mlir import cuda, types

import cuda.coop.numba_mlir as coop

THREADS = 32
ITEMS_PER_THREAD = 2
BINS = 17
DECODED_ITEMS_PER_THREAD = 2
DECODED_ITEMS = THREADS * DECODED_ITEMS_PER_THREAD
WINDOW_OFFSET = 3


@cuda.jit
def histogram_kernel(samples, counts):
    tid = cuda.threadIdx.x
    items = coop.ThreadData(ITEMS_PER_THREAD, dtype=types.int32)
    for item in range(ITEMS_PER_THREAD):
        items[item] = samples[tid * ITEMS_PER_THREAD + item]
    histogram = coop.histogram(
        coop.this_block(),
        items,
        bins=BINS,
        algorithm="sort",
    )
    counts[tid] = histogram[0]


@cuda.jit
def decode_kernel(run_values, run_lengths, decoded, relative, total):
    tid = cuda.threadIdx.x
    values = coop.ThreadData(1, dtype=types.int32)
    lengths = coop.ThreadData(1, dtype=types.uint32)
    values[0] = run_values[tid]
    lengths[0] = run_lengths[tid]

    # docs: start numba-qualified-run-length-decode
    relative_offsets = coop.ThreadData(
        DECODED_ITEMS_PER_THREAD,
        dtype=types.uint32,
    )
    total_decoded_size = coop.ThreadData(1, dtype=types.uint32)
    window = coop.run_length_decode(
        coop.this_block(),
        values,
        lengths,
        decoded_items_per_thread=DECODED_ITEMS_PER_THREAD,
        decoded_window_offset=WINDOW_OFFSET,
        relative_offsets=relative_offsets,
        total_decoded_size=total_decoded_size,
    )
    # docs: end numba-qualified-run-length-decode

    base = tid * DECODED_ITEMS_PER_THREAD
    for item in range(DECODED_ITEMS_PER_THREAD):
        decoded[base + item] = window[item]
        relative[base + item] = relative_offsets[item]
    total[tid] = total_decoded_size[0]


def run_example() -> tuple[np.ndarray, np.ndarray]:
    """Run both collectives and return their primary results."""

    samples = (np.arange(THREADS * ITEMS_PER_THREAD, dtype=np.int32) * 7 + 3) % BINS
    counts = np.zeros(THREADS, dtype=np.int32)
    histogram_kernel[1, THREADS](samples, counts)

    run_values = np.arange(THREADS, dtype=np.int32)
    run_lengths = np.zeros(THREADS, dtype=np.uint32)
    run_values[:4] = np.asarray([7, 8, 9, 10], dtype=np.int32)
    run_lengths[:4] = np.asarray([2, 1, 3, 4], dtype=np.uint32)
    decoded = np.zeros(DECODED_ITEMS, dtype=np.int32)
    relative = np.zeros(DECODED_ITEMS, dtype=np.uint32)
    total = np.zeros(THREADS, dtype=np.uint32)
    decode_kernel[1, THREADS](run_values, run_lengths, decoded, relative, total)
    cuda.synchronize()

    expected_counts = np.zeros(THREADS, dtype=np.int32)
    expected_counts[:BINS] = np.bincount(samples, minlength=BINS)
    np.testing.assert_array_equal(counts, expected_counts)

    stream = np.asarray([7, 7, 8, 9, 9, 9, 10, 10, 10, 10], dtype=np.int32)
    offsets = np.asarray([0, 1, 0, 0, 1, 2, 0, 1, 2, 3], dtype=np.uint32)
    valid_items = stream.size - WINDOW_OFFSET
    expected_decoded = np.zeros(DECODED_ITEMS, dtype=np.int32)
    expected_relative = np.full(DECODED_ITEMS, np.iinfo(np.uint32).max, np.uint32)
    expected_decoded[:valid_items] = stream[WINDOW_OFFSET:]
    expected_relative[:valid_items] = offsets[WINDOW_OFFSET:]
    np.testing.assert_array_equal(decoded, expected_decoded)
    np.testing.assert_array_equal(relative, expected_relative)
    np.testing.assert_array_equal(total, np.full(THREADS, stream.size, np.uint32))
    return counts, decoded


def main() -> int:
    counts, decoded = run_example()
    print({"histogram": counts, "decoded": decoded})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
