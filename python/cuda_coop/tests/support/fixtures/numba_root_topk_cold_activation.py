# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Fresh-process probe for automatic Numba activation through root TopK."""

from __future__ import annotations

import sys

import numpy as np

from cuda import coop

_BACKEND_MODULE = "cuda.coop.numba_mlir"
_THREADS = 32
_ITEMS_PER_THREAD = 2
_K = 7
_COMMON_PAIR_OPERATIONS = (
    "merge_sort_pairs",
    "radix_sort_pairs",
    "topk_max_pairs",
    "topk_min_pairs",
)


def main() -> None:
    assert _BACKEND_MODULE in sys.modules
    assert all(callable(getattr(coop, name)) for name in _COMMON_PAIR_OPERATIONS)

    from numba_cuda_mlir import cuda, types

    @cuda.jit
    def kernel(source, output):
        tid = cuda.threadIdx.x
        keys = coop.ThreadData(_ITEMS_PER_THREAD, dtype=types.int32)
        for item in range(_ITEMS_PER_THREAD):
            index = tid * _ITEMS_PER_THREAD + item
            keys[item] = source[index]

        selected = coop.topk_max_keys(coop.this_block(), keys, _K)
        for item in range(_ITEMS_PER_THREAD):
            index = tid * _ITEMS_PER_THREAD + item
            output[index] = selected[item]

    item_count = _THREADS * _ITEMS_PER_THREAD
    values = ((np.arange(item_count, dtype=np.int32) * 29) % item_count).astype(
        np.int32
    )
    output = np.zeros_like(values)
    kernel[1, _THREADS](values, output)
    cuda.synchronize()

    assert _BACKEND_MODULE in sys.modules
    np.testing.assert_array_equal(np.sort(output[:_K]), np.sort(values)[-_K:])


if __name__ == "__main__":
    main()
