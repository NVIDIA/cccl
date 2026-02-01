# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import TYPE_CHECKING, Protocol, TypeAlias, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from .iterators._base import IteratorBase


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


@runtime_checkable
class GpuStruct(Protocol):
    """
    Type of instances of structs created with gpu_struct().
    """

    _data: np.ndarray
    __array_interface__: dict
    dtype: np.dtype


if TYPE_CHECKING:
    IteratorLike: TypeAlias = DeviceArrayLike | IteratorBase
else:
    # At runtime, just use Any to avoid circular imports
    # The actual type checking happens statically via TYPE_CHECKING
    from typing import Any

    IteratorLike: TypeAlias = Any


"""
Type alias for values that can be used as iterators in compute algorithms.

Accepts either device arrays (objects with ``__cuda_array_interface__``) or
iterator objects (instances of :class:`~cuda.compute.iterators.IteratorBase`).

This alias is used throughout the algorithms API to indicate parameters that
accept both raw device arrays and custom iterators for flexible data access patterns.
"""
