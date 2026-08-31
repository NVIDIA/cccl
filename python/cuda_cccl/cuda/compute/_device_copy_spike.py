# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Private exploratory bridge for the c.parallel.v2 DeviceCopy ABI.

This module is intentionally not imported from ``cuda.compute``. It exists to
exercise the early HostJIT-backed C API from Python while the ABI is still
settling. The production binding should move the low-level pieces into a native
extension once the descriptor layout and DLPack unboxing policy are fixed.

Current spike limits:

* CUDA Array Interface objects only, not DLPack yet.
* Contiguous inputs only.
* Source and destination are flattened to a rank-1 runtime-extent copy.
* One loaded build result for the current device, or one explicit CC.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from ._utils.protocols import (
    get_data_pointer,
    get_dtype,
    get_shape,
    is_contiguous,
    validate_and_get_stream,
)

try:
    from cuda.core import Device as CudaDevice
except ImportError:
    from cuda.core.experimental import Device as CudaDevice

from cuda.cccl import get_include_paths  # type: ignore

_AXIS_RUNTIME = 0
_LAYOUT_RIGHT = 0
_TYPE_STORAGE = 11

_DTYPE_TO_TYPE_ENUM = {
    np.dtype("int8"): 0,
    np.dtype("int16"): 1,
    np.dtype("int32"): 2,
    np.dtype("int64"): 3,
    np.dtype("uint8"): 4,
    np.dtype("uint16"): 5,
    np.dtype("uint32"): 6,
    np.dtype("uint64"): 7,
    np.dtype("float16"): 8,
    np.dtype("float32"): 9,
    np.dtype("float64"): 10,
    np.dtype("bool"): 12,
}


class _TypeInfo(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_size_t),
        ("alignment", ctypes.c_size_t),
        ("type", ctypes.c_int),
    ]


class _AxisMetadata(ctypes.Structure):
    _fields_ = [
        ("kind", ctypes.c_int),
        ("value", ctypes.c_int64),
    ]


class _ViewBuild(ctypes.Structure):
    _fields_ = [
        ("layout", ctypes.c_int),
        ("strides", ctypes.POINTER(_AxisMetadata)),
    ]


class _BuildSpec(ctypes.Structure):
    _fields_ = [
        ("value_type", _TypeInfo),
        ("rank", ctypes.c_size_t),
        ("shape", ctypes.POINTER(_AxisMetadata)),
        ("source", _ViewBuild),
        ("destination", _ViewBuild),
    ]


class _SourceView(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("byte_offset", ctypes.c_uint64),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
    ]


class _DestinationView(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("byte_offset", ctypes.c_uint64),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
    ]


class _BuildResultStruct(ctypes.Structure):
    _fields_ = [
        ("cc", ctypes.c_int),
        ("payload", ctypes.c_void_p),
        ("payload_size", ctypes.c_size_t),
        ("jit_compiler", ctypes.c_void_p),
        ("copy_fn", ctypes.c_void_p),
        ("value_type", _TypeInfo),
        ("rank", ctypes.c_size_t),
        ("source_layout", ctypes.c_int),
        ("destination_layout", ctypes.c_int),
    ]


@dataclass(frozen=True)
class _ArrayView:
    owner: Any
    data: int
    dtype: np.dtype
    num_items: int
    byte_offset: int = 0


def _type_info_from_dtype(dtype: np.dtype) -> _TypeInfo:
    return _TypeInfo(
        ctypes.c_size_t(int(dtype.itemsize)),
        ctypes.c_size_t(max(1, int(dtype.alignment))),
        ctypes.c_int(_DTYPE_TO_TYPE_ENUM.get(dtype, _TYPE_STORAGE)),
    )


def _path_bytes(value: Any) -> bytes:
    if value is None:
        return b""
    if isinstance(value, bytes):
        return value
    return os.fsencode(str(value))


def _check_cuda(status: int, where: str) -> None:
    if status != 0:
        raise RuntimeError(f"{where} failed with CUresult {status}")


def _shape_size(shape: tuple[int, ...]) -> int:
    result = 1
    for extent in shape:
        result *= int(extent)
    return result


def _cc_to_key(cc: Any) -> int:
    if isinstance(cc, (tuple, list)):
        major, minor = cc
        return int(major) * 10 + int(minor)
    if isinstance(cc, str):
        cc = cc.replace(".", "")
    return int(cc)


def _key_to_cc(key: int) -> tuple[int, int]:
    return (key // 10, key % 10)


def _normalize_single_compute_capability(
    compute_capability: Any,
) -> tuple[int, int] | None:
    if compute_capability is None:
        return None

    if isinstance(compute_capability, (int, str)):
        return _key_to_cc(_cc_to_key(compute_capability))

    if (
        isinstance(compute_capability, tuple)
        and len(compute_capability) == 2
        and all(isinstance(item, int) for item in compute_capability)
    ):
        return _key_to_cc(_cc_to_key(compute_capability))

    ccs = list(compute_capability)
    if len(ccs) != 1:
        raise ValueError("device copy spike accepts at most one compute capability")
    return _key_to_cc(_cc_to_key(ccs[0]))


def _current_compute_capability() -> tuple[int, int]:
    major, minor = CudaDevice().compute_capability
    return int(major), int(minor)


def _include_options() -> tuple[bytes, bytes, bytes, bytes]:
    thrust_path, cub_path, libcudacxx_path, cuda_include_path = (
        get_include_paths().as_tuple()
    )
    return (
        _path_bytes(f"-I{cub_path}" if cub_path is not None else None),
        _path_bytes(f"-I{thrust_path}" if thrust_path is not None else None),
        _path_bytes(f"-I{libcudacxx_path}" if libcudacxx_path is not None else None),
        _path_bytes(
            f"-I{cuda_include_path}" if cuda_include_path is not None else None
        ),
    )


def _array_view(array: Any) -> _ArrayView:
    if not is_contiguous(array):
        raise ValueError("device copy spike requires contiguous arrays")

    dtype = np.dtype(get_dtype(array))
    shape = tuple(int(extent) for extent in get_shape(array))
    num_items = _shape_size(shape)

    return _ArrayView(
        owner=array,
        data=int(get_data_pointer(array)),
        dtype=dtype,
        num_items=num_items,
    )


def _same_array_contract(source: _ArrayView, destination: _ArrayView) -> None:
    if source.dtype != destination.dtype:
        raise TypeError(
            "source and destination dtypes must match; "
            f"got {source.dtype!r} and {destination.dtype!r}"
        )

    if source.num_items != destination.num_items:
        raise ValueError(
            "source and destination sizes must match; "
            f"got {source.num_items} and {destination.num_items}"
        )


class _BuildConfig(ctypes.Structure):
    _fields_ = [
        ("extra_compile_flags", ctypes.POINTER(ctypes.c_char_p)),
        ("num_extra_compile_flags", ctypes.c_size_t),
        ("extra_include_dirs", ctypes.POINTER(ctypes.c_char_p)),
        ("num_extra_include_dirs", ctypes.c_size_t),
        ("enable_pch", ctypes.c_int),
        ("verbose", ctypes.c_int),
        ("pch_cache_dir", ctypes.c_char_p),
    ]


def _configure_library(lib: ctypes.CDLL) -> None:
    lib.cccl_device_copy_build_ex.argtypes = [
        ctypes.POINTER(_BuildResultStruct),
        _BuildSpec,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.POINTER(_BuildConfig),
    ]
    lib.cccl_device_copy_build_ex.restype = ctypes.c_int

    lib.cccl_device_copy_build.argtypes = [
        ctypes.POINTER(_BuildResultStruct),
        _BuildSpec,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
    ]
    lib.cccl_device_copy_build.restype = ctypes.c_int

    lib.cccl_device_copy.argtypes = [
        _BuildResultStruct,
        _SourceView,
        _DestinationView,
        ctypes.c_void_p,
    ]
    lib.cccl_device_copy.restype = ctypes.c_int

    lib.cccl_device_copy_cleanup.argtypes = [ctypes.POINTER(_BuildResultStruct)]
    lib.cccl_device_copy_cleanup.restype = ctypes.c_int


def _library_candidates() -> list[str | None]:
    candidates: list[str | None] = []

    env_value = os.environ.get("CCCL_C_PARALLEL_V2_LIBRARY")
    if env_value:
        candidates.append(env_value)

    try:
        from . import _bindings

        candidates.append(_bindings.__file__)
    except Exception:
        pass

    candidates.append(ctypes.util.find_library("cccl.c.parallel.v2"))
    candidates.append("libcccl.c.parallel.v2.so")
    candidates.append("cccl.c.parallel.v2.dll")
    candidates.append("libcccl.c.parallel.v2.dylib")

    return candidates


@lru_cache(maxsize=1)
def _load_library() -> ctypes.CDLL:
    errors: list[str] = []

    for candidate in _library_candidates():
        if not candidate:
            continue

        try:
            lib = ctypes.CDLL(candidate)
            _configure_library(lib)
            return lib
        except (AttributeError, OSError) as exc:
            errors.append(f"{candidate}: {exc}")

    details = "\n".join(errors) if errors else "no candidates were tried"
    raise RuntimeError(
        "could not load c.parallel.v2 DeviceCopy entry points; set "
        "CCCL_C_PARALLEL_V2_LIBRARY to libcccl.c.parallel.v2.so.\n"
        f"{details}"
    )


def _build_impl(
    type_info: _TypeInfo,
    compute_capability: tuple[int, int] | None,
) -> "_DeviceCopyBuild":
    lib = _load_library()
    build_result = _BuildResultStruct()
    shape = (_AxisMetadata * 1)(_AxisMetadata(_AXIS_RUNTIME, 0))
    cc_major, cc_minor = (
        _current_compute_capability()
        if compute_capability is None
        else compute_capability
    )

    spec = _BuildSpec(
        value_type=type_info,
        rank=1,
        shape=shape,
        source=_ViewBuild(_LAYOUT_RIGHT, None),
        destination=_ViewBuild(_LAYOUT_RIGHT, None),
    )

    cub_path, thrust_path, libcudacxx_path, cuda_include_path = _include_options()
    build_config = _BuildConfig(
        extra_compile_flags=None,
        num_extra_compile_flags=0,
        extra_include_dirs=None,
        num_extra_include_dirs=0,
        enable_pch=0,
        verbose=0,
        pch_cache_dir=None,
    )

    status = lib.cccl_device_copy_build_ex(
        ctypes.byref(build_result),
        spec,
        cc_major,
        cc_minor,
        cub_path,
        thrust_path,
        libcudacxx_path,
        cuda_include_path,
        ctypes.byref(build_config),
    )
    _check_cuda(status, "cccl_device_copy_build_ex")

    return _DeviceCopyBuild(lib, build_result)


class _DeviceCopyBuild:
    def __init__(self, lib: ctypes.CDLL, build_result: _BuildResultStruct):
        self._lib = lib
        self._build_result = build_result
        self._source_shape = (ctypes.c_int64 * 1)()
        self._destination_shape = (ctypes.c_int64 * 1)()
        self._source_owner = None
        self._destination_owner = None
        self._closed = False

    def _get_cubin(self) -> bytes:
        if not self._build_result.payload or self._build_result.payload_size == 0:
            return b""
        return ctypes.string_at(
            self._build_result.payload, self._build_result.payload_size
        )

    def close(self) -> None:
        if self._closed:
            return

        status = self._lib.cccl_device_copy_cleanup(ctypes.byref(self._build_result))
        self._closed = True
        _check_cuda(status, "cccl_device_copy_cleanup")

    def copy(
        self,
        source: _ArrayView,
        destination: _ArrayView,
        stream: int | None,
    ) -> None:
        if self._closed:
            raise RuntimeError("DeviceCopy build result is closed")

        self._source_owner = source.owner
        self._destination_owner = destination.owner
        self._source_shape[0] = source.num_items
        self._destination_shape[0] = destination.num_items

        source_view = _SourceView(
            data=source.data,
            byte_offset=source.byte_offset,
            shape=self._source_shape,
            strides=None,
        )
        destination_view = _DestinationView(
            data=destination.data,
            byte_offset=destination.byte_offset,
            shape=self._destination_shape,
            strides=None,
        )

        stream_ptr = None if stream is None else ctypes.c_void_p(int(stream))
        status = self._lib.cccl_device_copy(
            self._build_result, source_view, destination_view, stream_ptr
        )
        _check_cuda(status, "cccl_device_copy")

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class _DeviceCopy:
    def __init__(
        self,
        source: Any,
        destination: Any,
        *,
        compute_capability: Any = None,
    ):
        source_view = _array_view(source)
        destination_view = _array_view(destination)
        _same_array_contract(source_view, destination_view)

        type_info = _type_info_from_dtype(source_view.dtype)
        cc = _normalize_single_compute_capability(compute_capability)

        self._build = _build_impl(type_info, cc)
        self._dtype = source_view.dtype
        self._num_items = source_view.num_items

    def __call__(self, source: Any, destination: Any, *, stream: Any = None) -> None:
        source_view = _array_view(source)
        destination_view = _array_view(destination)
        _same_array_contract(source_view, destination_view)

        if source_view.dtype != self._dtype:
            raise TypeError(
                f"device copy was built for {self._dtype!r}, got {source_view.dtype!r}"
            )

        if source_view.num_items != self._num_items:
            raise ValueError(
                "device copy was built for "
                f"{self._num_items} items, got {source_view.num_items}"
            )

        stream_handle = validate_and_get_stream(stream)
        self._build.copy(source_view, destination_view, stream_handle)

    def close(self) -> None:
        self._build.close()

    def __enter__(self) -> "_DeviceCopy":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def _get_cubin(self) -> bytes:
        return self._build._get_cubin()


def make_device_copy(
    source: Any,
    destination: Any,
    *,
    compute_capability: Any = None,
) -> _DeviceCopy:
    """Build a private DeviceCopy object for repeated copies of like arrays."""

    return _DeviceCopy(
        source,
        destination,
        compute_capability=compute_capability,
    )


def copy_into(
    source: Any,
    destination: Any,
    *,
    stream: Any = None,
    compute_capability: Any = None,
) -> None:
    """One-shot private DeviceCopy helper for exploratory testing."""

    device_copy = make_device_copy(
        source,
        destination,
        compute_capability=compute_capability,
    )
    try:
        device_copy(source, destination, stream=stream)
    finally:
        device_copy.close()
