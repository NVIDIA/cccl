# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import math
import operator
from collections.abc import Iterable

import numpy as np
from numpy.typing import DTypeLike

from cuda.core import Buffer, Device, Stream


def get_compute_capability() -> tuple[int, int]:
    return Device().compute_capability


def _normalize_shape(shape: int | Iterable[int]) -> tuple[int, ...]:
    try:
        dimensions = (operator.index(shape),)  # type: ignore[arg-type]
    except TypeError:
        dimensions = tuple(operator.index(dimension) for dimension in shape)  # type: ignore[union-attr]

    if any(dimension < 0 for dimension in dimensions):
        raise ValueError("negative dimensions are not allowed")

    return dimensions


def _contiguous_strides(
    shape: tuple[int, ...], itemsize: int, order: str
) -> tuple[int, ...]:
    if any(dimension == 0 for dimension in shape):
        return (0,) * len(shape)

    strides = [0] * len(shape)
    stride = itemsize

    if order == "C":
        for index in range(len(shape) - 1, -1, -1):
            strides[index] = stride
            stride *= shape[index]
    else:
        for index, dimension in enumerate(shape):
            strides[index] = stride
            stride *= dimension

    return tuple(strides)


def _resolve_device_and_stream(
    device: Device | None, stream: Stream | None
) -> tuple[Device, Stream]:
    if device is None:
        device = stream.device if stream is not None else Device()

    if stream is not None and stream.device.device_id != device.device_id:
        raise ValueError("device and stream must refer to the same device")

    device.set_current()
    return device, device.default_stream if stream is None else stream


class DeviceArray:
    """A small, Buffer-backed device array for cuda-cccl tests.

    The class intentionally provides only allocation, NumPy transfers, array
    metadata, and the CUDA Array Interface. Array operations and initialization
    belong on the NumPy host arrays used by the tests.
    """

    def __init__(
        self,
        buffer: Buffer,
        device: Device,
        stream: Stream,
        shape: tuple[int, ...],
        dtype: np.dtype,
        strides: tuple[int, ...],
        order: str,
    ) -> None:
        self._buffer = buffer
        self._device = device
        self._stream = stream
        self._order = order
        self._shape = shape
        self._dtype = dtype
        self._strides = strides

    @classmethod
    def empty(
        cls,
        shape: int | Iterable[int],
        dtype: DTypeLike,
        *,
        order: str = "C",
        device: Device | None = None,
        stream: Stream | None = None,
    ) -> DeviceArray:
        """Allocate an uninitialized device array."""
        shape = _normalize_shape(shape)
        dtype = np.dtype(dtype)
        order = order.upper()
        if order not in ("C", "F"):
            raise ValueError("order must be either 'C' or 'F'")
        if dtype.itemsize == 0:
            raise ValueError("zero-sized dtypes are not supported")

        device, stream = _resolve_device_and_stream(device, stream)
        buffer = device.allocate(math.prod(shape) * dtype.itemsize, stream=stream)
        result = cls(
            buffer,
            device,
            stream,
            shape,
            dtype,
            _contiguous_strides(shape, dtype.itemsize, order),
            order,
        )

        # Device allocation is stream ordered. Synchronizing makes an empty array
        # safe to hand to a test that subsequently uses a different stream.
        stream.sync()
        return result

    @classmethod
    def from_numpy(
        cls,
        array: np.ndarray,
        *,
        device: Device | None = None,
        stream: Stream | None = None,
    ) -> DeviceArray:
        """Allocate a device array and initialize it from a NumPy array."""
        host_array = np.asarray(array)
        if host_array.dtype.itemsize == 0:
            raise ValueError("zero-sized dtypes are not supported")

        if host_array.flags.c_contiguous:
            order = "C"
        elif host_array.flags.f_contiguous:
            order = "F"
        else:
            host_array = np.ascontiguousarray(host_array)
            order = "C"

        device, stream = _resolve_device_and_stream(device, stream)
        buffer = device.allocate(host_array.nbytes, stream=stream)
        result = cls(
            buffer,
            device,
            stream,
            host_array.shape,
            host_array.dtype,
            host_array.strides,
            order,
        )
        result._copy_from_host_array(host_array, stream)
        stream.sync()
        return result

    @property
    def nbytes(self) -> int:
        return self._buffer.size

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    def __len__(self) -> int:
        if not self._shape:
            raise TypeError("len() of unsized object")
        return self._shape[0]

    @property
    def __cuda_array_interface__(self) -> dict[str, object]:
        interface: dict[str, object] = {
            "data": (0 if self.nbytes == 0 else int(self._buffer.handle), False),
            "shape": self._shape,
            "strides": None if self._is_c_contiguous() else self._strides,
            "typestr": self._dtype.str,
            "version": 3,
        }
        if self._dtype.fields is not None:
            interface["descr"] = self._dtype.descr
        return interface

    def _is_c_contiguous(self) -> bool:
        return (
            self._order == "C"
            or self.nbytes == 0
            or sum(dimension > 1 for dimension in self._shape) <= 1
        )

    @staticmethod
    def _host_buffer(array: np.ndarray) -> Buffer:
        # Buffer.from_handle does not own the host memory. `owner` ties the NumPy
        # allocation to this temporary Buffer; the caller also retains the array
        # and synchronizes the copy stream before returning.
        return Buffer.from_handle(
            ptr=int(array.ctypes.data), size=array.nbytes, owner=array
        )

    def _copy_stream(self, stream: Stream | None) -> Stream:
        if stream is None:
            # The allocation stream is not necessarily the last stream to have
            # used the array. Synchronize the device when that stream is unknown.
            self._device.sync()
            return self._stream
        if stream.device.device_id != self._device.device_id:
            raise ValueError("copy stream must belong to the array's device")
        return stream

    def _copy_from_host_array(self, array: np.ndarray, stream: Stream) -> None:
        self._buffer.copy_from(self._host_buffer(array), stream=stream)

    def copy_from_host(
        self, array: np.ndarray, *, stream: Stream | None = None
    ) -> None:
        """Replace the array's contents from a shape- and dtype-matched NumPy array."""
        host_array = np.asarray(array)
        if host_array.shape != self._shape:
            raise ValueError(
                f"source shape {host_array.shape} does not match {self._shape}"
            )
        if host_array.dtype != self._dtype:
            raise TypeError(
                f"source dtype {host_array.dtype} does not match {self._dtype}"
            )

        if self._order == "F":
            host_array = np.asfortranarray(host_array)
        else:
            host_array = np.ascontiguousarray(host_array)

        self._device.set_current()
        stream = self._copy_stream(stream)
        self._copy_from_host_array(host_array, stream)
        stream.sync()

    def copy_to_host(self, *, stream: Stream | None = None) -> np.ndarray:
        """Return an owning NumPy copy of the array."""
        self._device.set_current()
        stream = self._copy_stream(stream)

        result = np.empty(self._shape, dtype=self._dtype, order=self._order)
        self._buffer.copy_to(self._host_buffer(result), stream=stream)
        stream.sync()
        return result
