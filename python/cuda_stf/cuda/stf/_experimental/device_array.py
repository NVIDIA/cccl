# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Lightweight device array backed by ``data_place.allocate()``.

Implements ``__cuda_array_interface__`` (CAI v3) so it can be passed
directly to ``cuda.compute`` algorithms, and provides ``copy_to_host`` /
``copy_to_device`` helpers that mirror the Numba DeviceNDArray API.

Shapes follow the public C-order contract: a :class:`DeviceArray` stores its
public shape, exposes it (with compact C-contiguous strides) through the CUDA
Array Interface, and supports contiguous ``reshape()`` views.
"""

from __future__ import annotations

import math
import weakref
from typing import TYPE_CHECKING

import numpy as np

from cuda.bindings import runtime as cudart

from ._stream_utils import get_stream_pointer

if TYPE_CHECKING:
    from cuda.stf._experimental._stf_bindings_impl import data_place


def _memcpy_sync_on_stream(dst: int, src: int, nbytes: int, kind: int, stream_int: int):
    """Stream-ordered ``cudaMemcpy`` that returns only once the copy is done.

    The copy is enqueued on *stream_int* (the allocation stream) so it is
    correctly ordered after a stream-ordered allocation, then the stream is
    synchronized to preserve the documented synchronous ``copy_to_*`` contract.
    A ``stream_int`` of 0 uses the default/null stream.

    *kind*: 1=H2D, 2=D2H, 3=D2D.
    """
    (err,) = cudart.cudaMemcpyAsync(
        dst,
        src,
        nbytes,
        cudart.cudaMemcpyKind(kind),
        stream_int,
    )
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"cudaMemcpyAsync failed with error code {int(err)}")
    (err,) = cudart.cudaStreamSynchronize(stream_int)
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"cudaStreamSynchronize failed with error code {int(err)}")


def _finalizer(dplace, ptr: int, nbytes: int, stream_int: int, stream=None):
    """Release memory back to the data place.

    *stream* is retained (unused directly) so a stream-ordered allocation's
    owning stream object stays alive until the matching deallocation runs.
    """
    try:
        dplace.deallocate(ptr, nbytes, stream_int if stream_int else None)
    except Exception as e:
        print(f"DeviceArray: deallocation warning: {e}")


def _normalize_shape(shape) -> tuple:
    """Normalize an int or a sequence of ints into a C-order shape tuple.

    Extents must be non-negative (a 0 extent yields an empty array); at most
    4 dimensions are supported (the allocation geometry limit).
    """
    if isinstance(shape, bool):
        raise TypeError("DeviceArray shape must be an int or a sequence of ints")
    if isinstance(shape, (int, np.integer)):
        shape = (int(shape),)
    shape = tuple(int(e) for e in shape)
    if not 1 <= len(shape) <= 4:
        raise ValueError(
            f"DeviceArray shape must have 1 to 4 dimensions, got {len(shape)}"
        )
    for e in shape:
        if e < 0:
            raise ValueError("DeviceArray size must be non-negative")
    return shape


class DeviceArray:
    """Device array allocated through a :class:`data_place`.

    Parameters
    ----------
    shape : int or sequence of int
        Public C-order shape (like a NumPy shape). A plain int creates a 1-D
        array.
    dtype : numpy dtype-like
        Element type.
    dplace : data_place
        The data place that owns the allocation.
    stream : optional
        CUDA stream for stream-ordered allocation.
    dims : sequence of int, optional
        Allocation geometry (C order) passed to ``data_place.allocate()``
        when it should differ from ``shape`` -- for example when a composite
        place's partition is expressed over different extents than the
        element view. Defaults to ``shape`` with ``elemsize`` equal to the
        dtype's item size, so a shaped allocation on a partitioned composite
        place works with no extra argument. ``prod(dims) * elemsize`` must
        equal ``prod(shape) * itemsize``.
    elemsize : int, optional
        Element size in bytes paired with ``dims``.
    """

    __slots__ = (
        "_ptr",
        "_shape",
        "_size",
        "_dtype",
        "_nbytes",
        "_dplace",
        "_stream",
        "_stream_int",
        "_base",
        "_finalizer_ref",
        "__weakref__",
    )

    def __init__(
        self,
        shape,
        dtype,
        dplace: "data_place",
        stream=None,
        *,
        dims=None,
        elemsize=None,
    ):
        self._shape = _normalize_shape(shape)
        self._dtype = np.dtype(dtype)
        self._size = math.prod(self._shape)
        self._nbytes = self._size * self._dtype.itemsize
        self._dplace = dplace
        # Retain the stream object (not just its raw handle): a stream-ordered
        # allocation stays valid only while the owning stream is alive, and the
        # handle is exposed through CAI so consumers can order after us.
        self._stream = stream
        self._stream_int = get_stream_pointer(stream)
        self._base = None

        if dims is None:
            if elemsize is not None:
                raise ValueError("DeviceArray: elemsize requires dims")
            dims, elemsize = self._shape, self._dtype.itemsize
        else:
            dims = tuple(int(d) for d in dims)
            elemsize = int(elemsize) if elemsize is not None else self._dtype.itemsize
            geom = elemsize
            for d in dims:
                geom *= d
            if geom != self._nbytes:
                raise ValueError(
                    f"DeviceArray: dims {dims} x elemsize {elemsize} = {geom} bytes "
                    f"!= shape {self._shape} x itemsize {self._dtype.itemsize}"
                )

        if self._nbytes > 0:
            self._ptr = dplace.allocate(dims, stream, elemsize=elemsize)
        else:
            self._ptr = 0

        self._finalizer_ref = weakref.finalize(
            self,
            _finalizer,
            dplace,
            self._ptr,
            self._nbytes,
            self._stream_int,
            self._stream,
        )

    @staticmethod
    def from_host(
        host_array: np.ndarray,
        dplace: "data_place",
        stream=None,
    ) -> "DeviceArray":
        """Allocate on *dplace* and copy *host_array* to the device.

        The host array is made C-contiguous; its shape is preserved.
        """
        host_array = np.ascontiguousarray(host_array)
        arr = DeviceArray(host_array.shape, host_array.dtype, dplace, stream)
        if arr._nbytes > 0:
            arr.copy_to_device(host_array)
        return arr

    # -- views (slicing, reshape) --------------------------------------------

    @staticmethod
    def _view(
        base_or_owner: "DeviceArray",
        ptr: int,
        shape: tuple,
        dtype: np.dtype,
        dplace: "data_place",
        stream_int: int,
        stream=None,
    ) -> "DeviceArray":
        """Create a non-owning view into an existing DeviceArray.

        The view holds the owning (root) array through ``_base`` so the
        allocation outlives every view.
        """
        view = object.__new__(DeviceArray)
        view._ptr = ptr
        view._shape = shape
        view._size = math.prod(shape)
        view._dtype = dtype
        view._nbytes = view._size * dtype.itemsize
        view._dplace = dplace
        view._stream = stream
        view._stream_int = stream_int
        root = base_or_owner._base if base_or_owner._base is not None else base_or_owner
        view._base = root
        view._finalizer_ref = None
        return view

    def reshape(self, *shape) -> "DeviceArray":
        """Return a non-owning view with a new C-order shape.

        The storage is C-contiguous, so any reshape preserving the element
        count is valid (adjacent-axis collapse, splitting, flattening). One
        dimension may be ``-1`` to be inferred, as in NumPy. The view keeps
        the owning array alive.
        """
        if len(shape) == 1 and not isinstance(shape[0], (int, np.integer)):
            shape = tuple(shape[0])
        shape = tuple(int(e) for e in shape)
        negatives = [i for i, e in enumerate(shape) if e == -1]
        if len(negatives) > 1:
            raise ValueError("DeviceArray.reshape: at most one dimension may be -1")
        if negatives:
            rest = math.prod(e for e in shape if e != -1)
            if rest == 0 or self._size % rest != 0:
                raise ValueError(
                    f"DeviceArray.reshape: cannot infer dimension for shape {shape} "
                    f"with {self._size} elements"
                )
            shape = tuple(self._size // rest if e == -1 else e for e in shape)
        shape = _normalize_shape(shape)
        if math.prod(shape) != self._size:
            raise ValueError(
                f"DeviceArray.reshape: shape {shape} has {math.prod(shape)} elements, "
                f"expected {self._size}"
            )
        return DeviceArray._view(
            self,
            self._ptr,
            shape,
            self._dtype,
            self._dplace,
            self._stream_int,
            self._stream,
        )

    def __getitem__(self, key):
        if isinstance(key, slice):
            if len(self._shape) != 1:
                raise IndexError(
                    "DeviceArray slicing is only supported on 1-D arrays; "
                    "reshape(-1) first"
                )
            start, stop, step = key.indices(self._size)
            if step != 1:
                raise IndexError("DeviceArray only supports contiguous slices (step=1)")
            length = max(0, stop - start)
            new_ptr = self._ptr + start * self._dtype.itemsize
            return DeviceArray._view(
                self,
                new_ptr,
                (length,),
                self._dtype,
                self._dplace,
                self._stream_int,
                self._stream,
            )
        raise TypeError(f"DeviceArray indices must be slices, not {type(key).__name__}")

    # -- CUDA Array Interface ----------------------------------------------

    @property
    def __cuda_array_interface__(self):
        cai = {
            "version": 3,
            "shape": self._shape,
            "typestr": self._dtype.str,
            "data": (self._ptr, False),
            # None means compact C-contiguous strides, which is exactly the
            # storage layout: shaped views never introduce gaps.
            "strides": None,
            # Advertise the allocation stream so consumers order their work
            # after our (possibly stream-ordered) allocation. CAI v3 forbids a
            # stream value of 0, so a null/default allocation stream is None.
            "stream": self._stream_int if self._stream_int else None,
        }
        if self._dtype.fields is not None:
            cai["descr"] = self._dtype.descr
        return cai

    # -- properties --------------------------------------------------------

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def shape(self) -> tuple:
        return self._shape

    @property
    def ndim(self) -> int:
        return len(self._shape)

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
        """Synchronous device-to-host copy.  Returns a new NumPy array of
        this array's shape."""
        host = np.empty(self._shape, dtype=self._dtype)
        if self._nbytes == 0:
            return host
        _memcpy_sync_on_stream(
            host.ctypes.data, self._ptr, self._nbytes, 2, self._stream_int
        )
        return host

    def copy_to_device(self, host_array: np.ndarray) -> None:
        """Copy *host_array* into this device buffer (synchronous H2D).

        The source must match this buffer's byte size exactly -- including for
        empty buffers and sliced views. A size mismatch is a programming error
        (it would otherwise silently leave part of the buffer untouched) and
        raises instead of performing a partial copy.
        """
        host_array = np.ascontiguousarray(host_array, dtype=self._dtype)
        nbytes = host_array.nbytes
        if nbytes != self._nbytes:
            raise ValueError(
                f"source ({nbytes} bytes) does not match destination buffer "
                f"({self._nbytes} bytes); sizes must match exactly"
            )
        if nbytes == 0:
            return
        _memcpy_sync_on_stream(
            self._ptr, host_array.ctypes.data, nbytes, 1, self._stream_int
        )

    def __repr__(self):
        return (
            f"DeviceArray(shape={self._shape}, dtype={self._dtype}, "
            f"ptr=0x{self._ptr:x}, place={self._dplace.kind})"
        )
