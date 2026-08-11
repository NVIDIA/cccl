# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from __future__ import annotations

import enum
import functools
import os
import subprocess
import tempfile
import warnings
from typing import Callable, List

try:
    from cuda.core import Device as CudaDevice
except ImportError:
    from cuda.core.experimental import Device as CudaDevice


import numpy as np

# TODO: adding a type-ignore here because `cuda` being a
# namespace package confuses mypy when `cuda.<something_else>`
# is installed, but not `cuda.cccl`. For namespace packages,
# it appears we need to actually install the sub-package
# in order for mypy to find its py.typed file. However, CI
# does type checking of `cuda.cccl` without actually installing
# it.
#
# We need to find a better solution for this.
from cuda.cccl import get_include_paths  # type: ignore

from . import types
from ._bindings import (
    CommonData,
    Iterator,
    IteratorKind,
    IteratorState,
    Op,
    OpKind,
    Pointer,
    TypeEnum,
    TypeInfo,
    Value,
    make_pointer_object,
)
from ._caching import _PerCCBuildResults
from ._utils.protocols import get_data_pointer, get_dtype, is_contiguous
from .iterators._base import IteratorBase
from .typing import DeviceArrayLike, GpuStruct

# Mapping from numpy dtype to TypeEnum for creating TypeInfo
_NUMPY_DTYPE_TO_ENUM = {
    np.dtype("int8"): TypeEnum.INT8,
    np.dtype("int16"): TypeEnum.INT16,
    np.dtype("int32"): TypeEnum.INT32,
    np.dtype("int64"): TypeEnum.INT64,
    np.dtype("uint8"): TypeEnum.UINT8,
    np.dtype("uint16"): TypeEnum.UINT16,
    np.dtype("uint32"): TypeEnum.UINT32,
    np.dtype("uint64"): TypeEnum.UINT64,
    np.dtype("float16"): TypeEnum.FLOAT16,
    np.dtype("float32"): TypeEnum.FLOAT32,
    np.dtype("float64"): TypeEnum.FLOAT64,
    np.dtype("bool"): TypeEnum.BOOLEAN,
}


@functools.lru_cache(maxsize=256)
def _type_info_from_dtype(dtype: np.dtype) -> TypeInfo:
    """
    Create a TypeInfo from a numpy dtype.
    Handles both primitive types and structured dtypes.
    """
    dtype = np.dtype(dtype)

    # Handle structured dtypes
    if dtype.type == np.void and dtype.fields is not None:
        return TypeInfo(dtype.itemsize, dtype.alignment, TypeEnum.STORAGE)

    if dtype.kind == "c":
        return TypeInfo(dtype.itemsize, dtype.alignment, TypeEnum.STORAGE)

    # Fallback for any other type
    type_enum = _NUMPY_DTYPE_TO_ENUM.get(dtype, TypeEnum.STORAGE)
    return TypeInfo(dtype.itemsize, dtype.alignment, type_enum)


def _is_well_known_op(op: OpKind) -> bool:
    return isinstance(op, OpKind) and op not in (OpKind.STATELESS, OpKind.STATEFUL)


def _device_array_to_cccl_iter(array: DeviceArrayLike) -> Iterator:
    from ._proxy import ProxyArray

    if not is_contiguous(array):
        raise ValueError("Non-contiguous arrays are not supported.")
    dtype = get_dtype(array)

    info = _type_info_from_dtype(dtype)
    state_info = _type_info_from_dtype(np.intp)
    # A ProxyArray has no GPU allocation: leave the pointer NULL for build-time
    # (ahead-of-time) compilation. The real pointer is bound at __call__ via
    # set_cccl_iterator_state().
    state = None if isinstance(array, ProxyArray) else get_data_pointer(array)
    return Iterator(
        state_info.alignment,
        IteratorKind.POINTER,
        Op(),
        Op(),
        info,
        # Note: this is slightly slower, but supports all ndarray-like objects
        # as long as they support CAI
        # TODO: switch to use gpumemoryview once it's ready
        state=state,
    )


def _none_to_cccl_iter() -> Iterator:
    # Any type could be used here, we just need to pass NULL.
    info = _type_info_from_dtype(np.uint8)
    return Iterator(info.alignment, IteratorKind.POINTER, Op(), Op(), info, state=None)


class _IteratorIO(enum.Enum):
    INPUT = 0
    OUTPUT = 1


def _to_cccl_iter(
    it: DeviceArrayLike | IteratorBase | None, io_kind: _IteratorIO
) -> Iterator:
    if it is None:
        return _none_to_cccl_iter()
    if isinstance(it, IteratorBase):
        return it.to_cccl_iter(io_kind == _IteratorIO.OUTPUT)
    return _device_array_to_cccl_iter(it)


def to_cccl_input_iter(array_or_iterator) -> Iterator:
    return _to_cccl_iter(array_or_iterator, _IteratorIO.INPUT)


def to_cccl_output_iter(array_or_iterator) -> Iterator:
    return _to_cccl_iter(array_or_iterator, _IteratorIO.OUTPUT)


def to_cccl_value_state(array_or_struct: np.ndarray | GpuStruct) -> memoryview:
    from ._proxy import _PROXY_VALUE_DATA_ERROR, ProxyValue

    if isinstance(array_or_struct, ProxyValue):
        # Reached only if a proxy leaks into an execute call — proxies describe
        # types for build, they carry no data to run with.
        raise RuntimeError(_PROXY_VALUE_DATA_ERROR)
    if isinstance(array_or_struct, np.ndarray):
        assert array_or_struct.flags.contiguous
        data = array_or_struct.data.cast("B")
        return data
    else:
        # it's a GpuStruct, use the array underlying it
        return to_cccl_value_state(array_or_struct._data)


def to_cccl_value(array_or_struct: np.ndarray | GpuStruct) -> Value:
    from ._proxy import ProxyValue

    if isinstance(array_or_struct, ProxyValue):
        # Build-time placeholder: describe the type with a correctly sized zero
        # buffer. The real value bytes are bound at __call__ via
        # set_cccl_value_state().
        info = _type_info_from_dtype(array_or_struct.dtype)
        zero_bytes = memoryview(bytearray(array_or_struct.dtype.itemsize))
        return Value(info, zero_bytes)
    if isinstance(array_or_struct, np.ndarray):
        info = _type_info_from_dtype(array_or_struct.dtype)
        return Value(info, array_or_struct.data.cast("B"))
    else:
        # it's a GpuStruct, use the array underlying it
        return to_cccl_value(array_or_struct._data)


def set_cccl_value_state(cccl_value: Value, array_or_struct: np.ndarray | GpuStruct):
    """
    Set the state of a CCCL Value object from a numpy array or GpuStruct.

    Args:
        cccl_value: The CCCL Value binding object
        array_or_struct: The numpy array or GpuStruct to get the state from
    """
    cccl_value.state = to_cccl_value_state(array_or_struct)


def get_value_type(
    d_in: DeviceArrayLike | IteratorBase | GpuStruct | np.ndarray,
):
    from ._proxy import ProxyValue
    from .struct import _Struct

    if isinstance(d_in, IteratorBase):
        return d_in.value_type

    if isinstance(d_in, ProxyValue):
        return types.from_numpy_dtype(d_in.dtype)

    if isinstance(d_in, _Struct):
        return type(d_in)._type_descriptor  # type: ignore[union-attr]

    dtype = get_dtype(d_in)

    if dtype.type == np.void:
        return types.from_numpy_dtype(dtype)

    return types.from_numpy_dtype(dtype)


def set_cccl_iterator_state(cccl_it: Iterator, input_it):
    if cccl_it.is_kind_pointer():
        ptr = get_data_pointer(input_it)
        ptr_obj = make_pointer_object(ptr, input_it)
        cccl_it.state = ptr_obj
    else:
        state_ = input_it.state
        if isinstance(state_, (IteratorState, Pointer)):
            cccl_it.state = state_
        else:
            cccl_it.state = make_pointer_object(state_, input_it)


@functools.lru_cache()
def get_includes() -> List[str]:
    def as_option(p):
        if p is None:
            return ""
        return f"-I{p}"

    paths = get_include_paths().as_tuple()
    opts = [as_option(path) for path in paths]
    return opts


def _check_compile_result(cubin: bytes):
    # check compiled code for LDL/STL instructions
    temp_cubin_file = tempfile.NamedTemporaryFile(delete=False)
    try:
        temp_cubin_file.write(cubin)
        out = subprocess.run(
            ["nvdisasm", "-gi", temp_cubin_file.name], capture_output=True
        )
        if out.returncode != 0:
            raise RuntimeError("nvdisasm failed")
        sass = out.stdout.decode("utf-8")
    except FileNotFoundError:
        sass = "nvdiasm not found, skipping SASS validation"
        warnings.warn(sass)

    assert "LDL" not in sass, "LDL instruction found in SASS"
    assert "STL" not in sass, "STL instruction found in SASS"
    return temp_cubin_file.name


# this global variable controls whether the compile result is checked
# for LDL/STL instructions. Should be set to `True` for testing only.
_check_sass: bool = False


def _common_data_for_cc(cc):
    """Build a ``CommonData`` for a given compute capability.

    ``cc`` is a ``(major, minor)`` pair. When ``None``, the current device's
    compute capability is queried (requires a live GPU).
    """
    if cc is None:
        cc_major, cc_minor = CudaDevice().compute_capability
    else:
        cc_major, cc_minor = cc
    cub_path, thrust_path, libcudacxx_path, cuda_include_path = get_includes()
    return CommonData(
        cc_major, cc_minor, cub_path, thrust_path, libcudacxx_path, cuda_include_path
    )


def call_build(build_impl_fn: Callable, *args, cc=None, **kwargs):
    """Build (compile + load) via ``build_impl_fn``, supplying compute capability and paths.

    ``cc`` is an optional ``(major, minor)`` pair; when ``None`` the current
    device's compute capability is used (the default, load-bearing behavior).
    Returns the loaded build result.
    """
    global _check_sass

    common_data = _common_data_for_cc(cc)
    result = build_impl_fn(
        *args,
        common_data,
        **kwargs,
    )

    if _check_sass:
        cubin = result._get_cubin()
        temp_cubin_file_name = _check_compile_result(cubin)
        os.unlink(temp_cubin_file_name)

    return result


def call_compile(build_impl_cls: Callable, *args, cc, **kwargs):
    """Compile only (no load) for an explicit compute capability ``cc``.

    ``build_impl_cls`` is a ``Device<Algo>BuildResult`` type exposing a
    ``compile(...)`` staticmethod. Unlike :func:`call_build`, this never touches
    the CUDA driver — it can run on a machine with no GPU. The returned build
    result is *not* loaded; call ``.load()`` (once, on a matching device) before
    executing. ``cc`` is a ``(major, minor)`` pair and is required.
    """
    common_data = _common_data_for_cc(cc)
    # build_impl_cls is a Device<Algo>BuildResult class exposing a compile()
    # staticmethod; it's typed Callable here, so silence the attr check.
    return build_impl_cls.compile(*args, common_data, **kwargs)  # type: ignore[attr-defined]


def build_for_ccs(build_impl_cls: Callable, *args, compute_capability=None, **kwargs):
    """Build the ``{cc_key: build_result}`` map for an algorithm.

    With ``compute_capability=None`` (the default), this performs a fused
    build+load for the current device and returns a single-entry map whose
    result is already loaded. Otherwise it compiles (without loading) for each
    requested compute capability and returns ``{cc_key: build_result}``, with
    each result loaded lazily on first use by ``resolve_build_result``.
    """
    ccs = normalize_compute_capabilities(compute_capability)
    if ccs is None:
        # Fused build+load for the current device. Query its cc once (clear error
        # if no device) and pass it through, so call_build doesn't re-query.
        device_id, cc_key = current_device_info()
        build_result = call_build(build_impl_cls, *args, cc=key_to_cc(cc_key), **kwargs)
        # The fused build already loaded the kernels; mark it so the lazy
        # load() in resolve_build_result() is a no-op (a second C load would leak /
        # re-register the library).
        build_result._loaded = True
        return _PerCCBuildResults({cc_key: build_result}, loaded_device_id=device_id)
    return _PerCCBuildResults(
        {
            cc_to_key(cc): call_compile(build_impl_cls, *args, cc=cc, **kwargs)
            for cc in ccs
        }
    )


def cc_to_key(cc) -> int:
    """Normalize a compute capability to the integer key ``major * 10 + minor``.

    Accepts an int (``90``, ``75``), a ``(major, minor)`` pair, or a string
    like ``"90"`` / ``"9.0"``.
    """
    if isinstance(cc, (tuple, list)):
        major, minor = cc
        return int(major) * 10 + int(minor)
    if isinstance(cc, str):
        cc = cc.replace(".", "")
        return int(cc)
    return int(cc)


def key_to_cc(key: int):
    """Inverse of :func:`cc_to_key`: integer key -> ``(major, minor)`` pair."""
    return (key // 10, key % 10)


def normalize_compute_capabilities(compute_capability):
    """Normalize the ``compute_capability=`` argument of ``make_<algo>``.

    Returns a sorted list of unique ``(major, minor)`` pairs, or ``None`` to
    mean "use the current device" (the default build path). Accepts a single
    cc (int / pair / str) or a list thereof.
    """
    if compute_capability is None:
        return None
    if isinstance(compute_capability, (int, str)):
        ccs = [compute_capability]
    elif (
        isinstance(compute_capability, tuple)
        and len(compute_capability) == 2
        and all(isinstance(x, int) for x in compute_capability)
    ):
        # a single (major, minor) pair
        ccs = [compute_capability]
    else:
        ccs = list(compute_capability)
    keys = sorted({cc_to_key(cc) for cc in ccs})
    if not keys:
        raise ValueError("compute_capability list is empty")
    return [key_to_cc(k) for k in keys]


def current_device_info() -> tuple[int, int]:
    """Return the current device ordinal and packed compute-capability key.

    Raises a clear, actionable error if no CUDA device is available: building
    without a GPU has no device to infer the target arch from, so the caller
    must pass an explicit ``compute_capability=``.
    """
    try:
        device = CudaDevice()
        cc = device.compute_capability
    except Exception as e:
        raise RuntimeError(
            "No compute_capability was given and no CUDA device is available to target."
        ) from e
    return device.device_id, cc_to_key(tuple(cc))


def current_device_cc_key() -> int:
    """The current device's compute capability as a ``major * 10 + minor`` key."""
    return current_device_info()[1]


def current_device_id() -> int:
    """The current CUDA device ordinal, without a compute-capability query.

    The compute-capability query roughly doubles the cost of
    ``current_device_info()``, and callers that only key per-device state
    (see ``resolve_build_result``) run on every algorithm invocation.
    """
    try:
        return CudaDevice().device_id
    except Exception as e:
        raise RuntimeError("No CUDA device is available to execute on.") from e


def resolve_build_result(build_results: dict, bound_result=None):
    """Load the build result for the current device.

    ``bound_result`` is a default-build wrapper's construction-time binding
    (see cache_build_results): already the loaded result for the wrapper's
    device, returned without any device query. Deserialized wrappers have no
    binding and resolve per call.
    """
    if bound_result is not None:
        return bound_result

    # Wrappers always hold a _PerCCBuildResults (build_for_ccs and
    # deserialization both produce one); the per-device ownership/clone
    # protocol in resolve() relies on it, so fail loudly on anything else
    # rather than fall back to an unprotected load.
    assert isinstance(build_results, _PerCCBuildResults)

    if len(build_results) == 1:
        # A singular _PerCCBuildResults is used as-is whatever the current device's
        # compute capability is (single-target blobs were already cc-checked at
        # deserialization), so only the device ordinal is needed to key the
        # per-device loaded state. This path runs on every call of AOT and
        # deserialized wrappers; skip the costlier compute-capability query.
        (build_result_cc,) = build_results
        device_id = current_device_id()
    else:
        device_id, device_cc_key = current_device_info()
        build_result_cc = device_cc_key
        if build_result_cc not in build_results:
            available = ", ".join(
                f"{maj}.{minor}"
                for maj, minor in (key_to_cc(k) for k in sorted(build_results))
            )
            major, minor = key_to_cc(build_result_cc)
            raise RuntimeError(
                f"This algorithm was compiled for compute capabilities [{available}], "
                f"but the current device has compute capability {major}.{minor}. "
                f"Rebuild with compute_capability including {major}{minor}."
            )

    return build_results.resolve(build_result_cc, device_id)
