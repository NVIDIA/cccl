# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Lightweight device array backed by ``data_place.allocate()``.

Implements ``__cuda_array_interface__`` (CAI v3) so it can be passed
directly to ``cuda.compute`` algorithms, and provides ``copy_to_host`` /
``copy_to_device`` helpers that mirror the Numba DeviceNDArray API.
"""

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING

import numpy as np

from cuda.bindings import runtime as cudart

if TYPE_CHECKING:
    from cuda.stf._experimental._stf_bindings_impl import data_place


def _memcpy(dst: int, src: int, nbytes: int, kind: int):
    """cudaMemcpy wrapper.  *kind*: 1=H2D, 2=D2H, 3=D2D."""
    (err,) = cudart.cudaMemcpy(
        dst,
        src,
        nbytes,
        cudart.cudaMemcpyKind(kind),
    )
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"cudaMemcpy failed with error code {int(err)}")


def _finalizer(dplace, ptr: int, nbytes: int, stream_int: int):
    """Release memory back to the data place."""
    try:
        dplace.deallocate(ptr, nbytes, stream_int if stream_int else None)
    except Exception as e:
        print(f"DeviceArray: deallocation warning: {e}")


class DeviceArray:
    """1-D device array allocated through a :class:`data_place`.

    Parameters
    ----------
    size : int
        Number of elements.
    dtype : numpy dtype-like
        Element type.
    dplace : data_place
        The data place that owns the allocation.
    stream : optional
        CUDA stream for stream-ordered allocation.
    """

    __slots__ = (
        "_ptr",
        "_size",
        "_dtype",
        "_nbytes",
        "_dplace",
        "_stream_int",
        "_base",
        "_finalizer_ref",
        "__weakref__",
    )

    def __init__(self, size: int, dtype, dplace: "data_place", stream=None):
        if size < 0:
            raise ValueError("DeviceArray size must be non-negative")

        self._dtype = np.dtype(dtype)
        self._size = size
        self._nbytes = size * self._dtype.itemsize
        self._dplace = dplace
        self._stream_int = int(stream) if stream is not None else 0
        self._base = None

        if self._nbytes > 0:
            self._ptr = dplace.allocate(self._nbytes, stream)
        else:
            self._ptr = 0

        self._finalizer_ref = weakref.finalize(
            self,
            _finalizer,
            dplace,
            self._ptr,
            self._nbytes,
            self._stream_int,
        )

    @staticmethod
    def from_host(
        host_array: np.ndarray,
        dplace: "data_place",
        stream=None,
    ) -> "DeviceArray":
        """Allocate on *dplace* and copy *host_array* to the device."""
        host_array = np.ascontiguousarray(host_array)
        if host_array.ndim != 1:
            raise ValueError("DeviceArray.from_host only supports 1-D host arrays")

        arr = DeviceArray(host_array.shape[0], host_array.dtype, dplace, stream)
        if arr._nbytes > 0:
            arr.copy_to_device(host_array)
        return arr

    # -- views (slicing) ---------------------------------------------------

    @staticmethod
    def _view(
        base_or_owner: "DeviceArray",
        ptr: int,
        size: int,
        dtype: np.dtype,
        dplace: "data_place",
        stream_int: int,
    ) -> "DeviceArray":
        """Create a non-owning view into an existing DeviceArray."""
        view = object.__new__(DeviceArray)
        view._ptr = ptr
        view._size = size
        view._dtype = dtype
        view._nbytes = size * dtype.itemsize
        view._dplace = dplace
        view._stream_int = stream_int
        root = base_or_owner._base if base_or_owner._base is not None else base_or_owner
        view._base = root
        view._finalizer_ref = None
        return view

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(self._size)
            if step != 1:
                raise IndexError("DeviceArray only supports contiguous slices (step=1)")
            length = max(0, stop - start)
            new_ptr = self._ptr + start * self._dtype.itemsize
            return DeviceArray._view(
                self,
                new_ptr,
                length,
                self._dtype,
                self._dplace,
                self._stream_int,
            )
        raise TypeError(f"DeviceArray indices must be slices, not {type(key).__name__}")

    # -- CUDA Array Interface ----------------------------------------------

    @property
    def __cuda_array_interface__(self):
        return {
            "version": 3,
            "shape": (self._size,),
            "typestr": self._dtype.str,
            "data": (self._ptr, False),
            "strides": None,
        }

    # -- properties --------------------------------------------------------

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def shape(self):
        return (self._size,)

    @property
    def size(self) -> int:
        return self._size

    @property
    def nbytes(self) -> int:
        return self._nbytes

    @property
    def data_place(self) -> "data_place":
        """The :class:`data_place` backing this array."""
        return self._dplace

    # -- host <-> device transfers -----------------------------------------

    def copy_to_host(self) -> np.ndarray:
        """Synchronous device-to-host copy.  Returns a new NumPy array."""
        host = np.empty(self._size, dtype=self._dtype)
        if self._nbytes == 0:
            return host
        _memcpy(host.ctypes.data, self._ptr, self._nbytes, kind=2)
        return host

    def copy_to_device(self, host_array: np.ndarray) -> None:
        """Copy *host_array* into this device buffer (synchronous H2D).

        If *host_array* is smaller than this buffer, only the leading bytes are
        overwritten. This supports copying into sliced ``DeviceArray`` views.
        """
        host_array = np.ascontiguousarray(host_array, dtype=self._dtype)
        nbytes = host_array.nbytes
        if nbytes == 0:
            return
        if nbytes > self._nbytes:
            raise ValueError(
                f"source ({nbytes} bytes) exceeds buffer ({self._nbytes} bytes)"
            )
        _memcpy(self._ptr, host_array.ctypes.data, nbytes, kind=1)

    def __repr__(self):
        return (
            f"DeviceArray(size={self._size}, dtype={self._dtype}, "
            f"ptr=0x{self._ptr:x}, place={self._dplace.kind})"
        )
