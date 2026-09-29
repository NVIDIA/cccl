# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Dump generated CUDA for a selected example kernel.

Set CUDA_COOP_SOURCE_DUMP_DIR to the output directory before running.
"""

import argparse

import numpy as np
from numba_cuda_mlir import cuda

import cuda.coop.numba_mlir as numba_coop  # noqa: F401 -- activate the backend
from cuda import coop


# docs: start dump-direct
@cuda.jit
def copy_direct(source, destination):
    block = coop.this_block()
    items = coop.ThreadData(2, dtype=np.int32)
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
def copy_transpose(source, destination):
    block = coop.this_block()
    items = coop.ThreadData(2, dtype=np.int32)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("example", choices=("direct", "transpose"))
    example = parser.parse_args().example
    count = 256
    source = ((np.arange(count) * 17) % 113 - 51).astype(np.int32)
    destination = np.empty_like(source)
    kernel = {
        "direct": copy_direct,
        "transpose": copy_transpose,
    }[example]
    kernel[1, 128](source, destination)
    cuda.synchronize()
    expected = source
    np.testing.assert_array_equal(destination, expected)
    print(f"{example}: result verified")


if __name__ == "__main__":
    main()
