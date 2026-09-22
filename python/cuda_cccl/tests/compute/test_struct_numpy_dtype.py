# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np

from cuda.compute import gpu_struct


def test_gpu_struct_ignores_numpy_field_titles():
    """Field titles are aliases and must not become additional struct fields."""
    dtype = np.dtype([(("label", "a"), np.int32), ("b", np.float32)], align=True)

    struct_type = gpu_struct(dtype)

    assert struct_type.dtype.names == ("a", "b")
    assert struct_type.dtype.fields["a"][1] == 0
    assert struct_type.dtype.fields["b"][1] == 4
    assert struct_type.dtype.itemsize == 8
