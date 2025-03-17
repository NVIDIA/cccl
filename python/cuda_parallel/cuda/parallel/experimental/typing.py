# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Any

from typing_extensions import (
    Protocol,
)  # TODO: typing_extensions required for Python 3.7 docs env


class DeviceArrayLike(Protocol):
    """
    Objects representing a device array, having a `.__cuda_array_interface__`
    attribute.
    """

    __cuda_array_interface__: dict


# TODO: type GpuStruct appropriately. It should be any type that has
# been decorated with `@gpu_struct`.
GpuStruct = Any
GpuStruct.__doc__ = """\
    Type of instances of classes decorated with @gpu_struct.
"""
