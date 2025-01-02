# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Utilities for extracting information from `__cuda_array_interface__`.
"""

from typing import Optional, Tuple
import numpy as np

from ..typing import DeviceArrayLike


def get_dtype(arr: DeviceArrayLike) -> np.dtype:
    return np.dtype(arr.__cuda_array_interface__["typestr"])


def get_stride(arr: DeviceArrayLike) -> Optional[Tuple]:
    return arr.__cuda_array_interface__["strides"]
