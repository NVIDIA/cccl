# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import numpy as np

from cuda.compute import gpu_struct


def test_deferred_annotations():
    # test that we can use @gpu_struct with deferred annotations
    # GH: #6421

    @gpu_struct
    class MyStruct:
        x: np.int32
        y: np.int32
