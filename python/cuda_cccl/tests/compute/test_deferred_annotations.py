# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import cupy as cp
import numpy as np

from cuda.compute import OpKind, TransformIterator, gpu_struct, reduce_into


def test_deferred_annotations():
    # test that we can use @gpu_struct with deferred annotations
    # GH: #6421

    @gpu_struct
    class MyStruct:
        x: np.int32
        y: np.int32


def test_transform_iterator_future_annotations():
    def add_one(x: "np.int32") -> "np.int32":
        return x + np.int32(1)

    d_in = cp.arange(8, dtype=np.int32)
    d_out = cp.empty(1, dtype=np.int32)
    h_init = np.array([0], dtype=np.int32)

    transform_it = TransformIterator(d_in, add_one)
    reduce_into(transform_it, d_out, OpKind.PLUS, d_in.size, h_init)

    expected = int(cp.sum(d_in + 1).get())
    assert int(d_out.get()[0]) == expected
