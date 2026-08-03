# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import numpy as np
from _utils.device_array import DeviceArray

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

    h_in = np.arange(8, dtype=np.int32)
    d_in = DeviceArray.from_numpy(h_in)
    d_out = DeviceArray.empty(1, np.int32)
    h_init = np.array([0], dtype=np.int32)

    transform_it = TransformIterator(d_in, add_one)
    reduce_into(
        d_in=transform_it,
        d_out=d_out,
        num_items=h_in.size,
        op=OpKind.PLUS,
        h_init=h_init,
    )

    expected = int(np.sum(h_in + 1))
    assert int(d_out.copy_to_host()[0]) == expected
