# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Callable, Protocol, TypeVar

from .iterators import IteratorBase
from .op import OpAdapter, OpKind
from .struct import _Struct


class DeviceArrayLike(Protocol):
    """Protocol for array-like objects that expose device memory via CUDA Array Interface.

    Any object implementing the ``__cuda_array_interface__`` attribute can be used
    where a :class:`DeviceArrayLike` is expected. This includes CuPy arrays, Numba
    device arrays, PyTorch CUDA tensors, and other GPU array types.

    See `CUDA Array Interface specification <https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html>`_
    for details.
    """

    __cuda_array_interface__: dict


class StreamLike(Protocol):
    """Protocol for CUDA stream objects.

    Any object implementing the ``__cuda_stream__()`` method can be used where a
    :class:`StreamLike` is expected. This includes stream objects from CuPy,
    Numba CUDA, PyTorch, and other CUDA libraries.
    """

    def __cuda_stream__(self) -> tuple[int, int]: ...


GpuStruct = TypeVar("GpuStruct", bound=_Struct)
"""
Instance of types created with :class:`cuda.compute.struct.gpu_struct`.
"""

IteratorT = TypeVar("IteratorT", bound=IteratorBase)
"""Type variable for iterator objects.

Represents any subclass of :class:`IteratorBase <cuda.compute.iterators.IteratorBase>`.
See :py:mod:`cuda.compute.iterators` for all available iterators.
"""

Operator = Callable | OpKind | OpAdapter
"""Type alias for operator objects passed to algorithm functions.

Algorithms accept the following objects as operators:

**Python Callable:**
    A Python function or lambda that will be JIT compiled by `numba.cuda`.

**OpKind Enum:**
    Pre-defined operator constants for common operations. See
    :class:`OpKind <cuda.compute.op.OpKind>` for all available operators.
"""

__all__ = ["DeviceArrayLike", "GpuStruct", "IteratorT", "Operator"]
