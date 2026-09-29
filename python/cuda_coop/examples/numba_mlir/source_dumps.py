# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Dump generated CUDA for a selected example kernel.

Set CUDA_COOP_SOURCE_DUMP_DIR to the output directory before running.
"""

import argparse

import numpy as np
from numba_cuda_mlir import cuda

import cuda.coop.numba_mlir as numba_coop
from cuda import coop


# docs: start dump-direct
@cuda.jit
def copy_direct(source, destination, items_per_thread):
    block = coop.this_block()
    items = coop.ThreadData(items_per_thread)
    coop.load(
        block,
        source,
        items,
        algorithm="direct",
    )
    coop.store(
        block,
        destination,
        items,
        algorithm="direct",
    )


# docs: end dump-direct


# docs: start dump-transpose
@cuda.jit
def copy_transpose(source, destination, items_per_thread):
    block = coop.this_block()
    items = coop.ThreadData(items_per_thread)
    scratch = coop.TempStorage(auto_sync=True)
    coop.load(
        block,
        source,
        items,
        algorithm="transpose",
        temp_storage=scratch,
    )
    coop.store(
        block,
        destination,
        items,
        algorithm="transpose",
        temp_storage=scratch,
    )


# docs: end dump-transpose


# docs: start dump-scan
@cuda.jit(device=True)
def maximum(left, right):
    if left > right:
        return left
    return right


@cuda.jit
def scan_maximum(source, destination):
    block = numba_coop.this_block()
    value = source[cuda.threadIdx.x]
    destination[cuda.threadIdx.x] = numba_coop.inclusive_scan(
        block,
        value,
        scan_op=maximum,
    )


# docs: end dump-scan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("example", choices=("direct", "transpose", "scan"))
    example = parser.parse_args().example
    count = 128 if example == "scan" else 256
    source = ((np.arange(count) * 17) % 113 - 51).astype(np.int32)
    destination = np.empty_like(source)
    kernel = {
        "direct": copy_direct,
        "transpose": copy_transpose,
        "scan": scan_maximum,
    }[example]
    if example == "scan":
        kernel[1, 128](source, destination)
    else:
        items_per_thread = 4
        kernel[1, count // items_per_thread](
            source, destination, items_per_thread
        )
    cuda.synchronize()
    expected = np.maximum.accumulate(source) if example == "scan" else source
    np.testing.assert_array_equal(destination, expected)
    print(f"{example}: result verified")


if __name__ == "__main__":
    main()
