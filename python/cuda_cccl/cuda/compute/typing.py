# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Protocol, runtime_checkable


@runtime_checkable
class DeviceArrayLike(Protocol):
    """
    Objects representing a device array, having a `.__cuda_array_interface__`
    attribute.
    """

    __cuda_array_interface__: dict


class StreamLike(Protocol):
    """
    Objects representing a stream, having a `.__cuda_stream__` attribute.
    """

    def __cuda_stream__(self) -> tuple[int, int]: ...
