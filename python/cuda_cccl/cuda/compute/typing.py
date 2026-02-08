# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import TYPE_CHECKING, Any, Protocol

from .struct import _Struct

if TYPE_CHECKING:
    from .iterators._base import IteratorBase
else:
    IteratorBase = Any


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


GpuStruct = _Struct
