# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Lightweight device array backed by ``data_place.allocate()``.

Implements BOTH device-memory interchange protocols -- they are
complementary, and a consumer picks by construction:

* ``__cuda_array_interface__`` (CAI v3): a *description* of the memory.
  The importer retains nothing; the ``DeviceArray`` (or something holding
  it) must outlive every borrowed view. This is the borrowed / zero-copy
  path -- ``cuda.compute`` algorithms, Numba, ``torch.as_tensor``.
* ``__dlpack__`` / ``__dlpack_device__`` (DLPack): an *ownership-carrying*
  export. The capsule holds the owning array alive, and the consumer's
  deleter releases it when the imported tensor's storage dies -- e.g.
  ``torch.from_dlpack`` gives a tensor whose lifetime carries the
  allocation, with the ``DeviceArray`` finalizer remaining the single
  deallocation point.

Also provides ``copy_to_host`` / ``copy_to_device`` helpers that mirror the
Numba DeviceNDArray API.
"""

from __future__ import annotations

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
        "_stream",
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
        # Retain the stream object (not just its raw handle): a stream-ordered
        # allocation stays valid only while the owning stream is alive, and the
        # handle is exposed through CAI so consumers can order after us.
        self._stream = stream
        self._stream_int = get_stream_pointer(stream)
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
            self._stream,
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
        stream=None,
    ) -> "DeviceArray":
        """Create a non-owning view into an existing DeviceArray."""
        view = object.__new__(DeviceArray)
        view._ptr = ptr
        view._size = size
        view._dtype = dtype
        view._nbytes = size * dtype.itemsize
        view._dplace = dplace
        view._stream = stream
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
                self._stream,
            )
        raise TypeError(f"DeviceArray indices must be slices, not {type(key).__name__}")

    # -- CUDA Array Interface ----------------------------------------------

    @property
    def __cuda_array_interface__(self):
        cai = {
            "version": 3,
            "shape": (self._size,),
            "typestr": self._dtype.str,
            "data": (self._ptr, False),
            "strides": None,
            # Advertise the allocation stream so consumers order their work
            # after our (possibly stream-ordered) allocation. CAI v3 forbids a
            # stream value of 0, so a null/default allocation stream is None.
            "stream": self._stream_int if self._stream_int else None,
        }
        if self._dtype.fields is not None:
            cai["descr"] = self._dtype.descr
        return cai

    # -- DLPack --------------------------------------------------------------

    # numpy dtype kind -> dlpack.h DLDataTypeCode. bfloat16 has no numpy
    # dtype; buffers for such types are allocated through a same-size
    # storage dtype and viewed back on the consumer side.
    _DLPACK_KIND_CODE = {"i": 0, "u": 1, "f": 2, "c": 5, "b": 6}
    _DLPACK_DEVICE_CUDA = 2  # kDLCUDA

    def _device_ordinal(self) -> int:
        """The CUDA device ordinal owning this array's memory.

        The driver answers authoritatively per pointer (a composite VMM
        range spans the locality domains of ONE physical device); empty
        arrays fall back to the current device.
        """
        if self._ptr:
            try:
                from cuda.bindings import driver as _drv  # noqa: PLC0415

                err, dev = _drv.cuPointerGetAttribute(
                    _drv.CUpointer_attribute.CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
                    self._ptr,
                )
                if int(err) == 0:
                    return int(dev)
            except Exception:
                pass
        err, dev = cudart.cudaGetDevice()
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaGetDevice failed with error code {int(err)}")
        return int(dev)

    def __dlpack_device__(self):
        return (self._DLPACK_DEVICE_CUDA, self._device_ordinal())

    def _make_ready_on(self, consumer_stream) -> None:
        """DLPack stream handshake: make the data ready on the consumer's
        stream.

        Per the protocol (CUDA device type): ``-1`` requests no
        synchronization; ``None``/``1`` mean the legacy default stream;
        any other int is the consumer's stream handle. The producer inserts
        an event-wait ordering the consumer stream after the allocation
        stream -- nothing blocks on the host.
        """
        if consumer_stream == -1:
            return
        if consumer_stream is None or consumer_stream == 1:
            consumer = 0
        elif consumer_stream == 2:  # per-thread default stream
            consumer = 2
        elif isinstance(consumer_stream, int):
            if consumer_stream < 0:
                raise ValueError(f"__dlpack__: invalid stream value {consumer_stream}")
            consumer = consumer_stream
        else:
            raise TypeError("__dlpack__: stream must be an int or None")
        producer = self._stream_int
        if consumer == producer or self._nbytes == 0:
            return
        (err, event) = cudart.cudaEventCreateWithFlags(cudart.cudaEventDisableTiming)
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaEventCreateWithFlags failed ({int(err)})")
        try:
            (err,) = cudart.cudaEventRecord(event, producer)
            if err != cudart.cudaError_t.cudaSuccess:
                raise RuntimeError(f"cudaEventRecord failed ({int(err)})")
            (err,) = cudart.cudaStreamWaitEvent(consumer, event, 0)
            if err != cudart.cudaError_t.cudaSuccess:
                raise RuntimeError(f"cudaStreamWaitEvent failed ({int(err)})")
        finally:
            cudart.cudaEventDestroy(event)

    def __dlpack__(self, *, stream=None, max_version=None, dl_device=None, copy=None):
        """Export as a ``"dltensor"`` capsule (ownership-carrying).

        The capsule keeps the OWNING array (a view's root) alive; the
        consumer's deleter drops that reference when the imported tensor's
        storage dies, and the ``DeviceArray`` finalizer -- the single
        deallocation point -- then frees the memory once no other reference
        remains. Complements ``__cuda_array_interface__``, which describes
        the same memory but transfers no ownership.

        ``max_version`` is accepted and answered with an unversioned
        capsule (permitted by the spec; understood by all consumers).
        """
        if copy:
            raise BufferError(
                "DeviceArray.__dlpack__: copy=True is not supported "
                "(export is zero-copy by design)"
            )
        if dl_device is not None and tuple(dl_device) != self.__dlpack_device__():
            raise BufferError(
                f"DeviceArray.__dlpack__: cannot export to device {dl_device}; "
                f"data lives on {self.__dlpack_device__()}"
            )
        if self._dtype.fields is not None:
            raise BufferError(
                "DeviceArray.__dlpack__: structured dtypes are not "
                "representable in DLPack; use __cuda_array_interface__"
            )
        code = self._DLPACK_KIND_CODE.get(self._dtype.kind)
        if code is None:
            raise BufferError(
                f"DeviceArray.__dlpack__: dtype {self._dtype} is not "
                "representable in DLPack"
            )
        self._make_ready_on(stream)
        from cuda.stf._experimental._stf_bindings import dlpack_export  # noqa: PLC0415

        owner = self._base if self._base is not None else self
        return dlpack_export(
            owner,
            self._ptr,
            (self._size,),
            code,
            self._dtype.itemsize * 8,
            self._device_ordinal(),
        )

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
            f"DeviceArray(size={self._size}, dtype={self._dtype}, "
            f"ptr=0x{self._ptr:x}, place={self._dplace.kind})"
        )
