import ctypes
from typing import Callable

import numba
import numpy as np
from numba import cuda
from numba.cuda.cudadrv import enums

from .. import _cccl as cccl
from .._bindings import get_bindings, get_paths
from .._caching import CachableFunction, cache_with_key
from .._utils import protocols
from ..iterators._iterators import IteratorBase
from ..typing import DeviceArrayLike, GpuStruct


class _SegmentedReduce:
    def __del__(self):
        if self.build_result is None:
            return
        bindings = get_bindings()
        bindings.cccl_device_segmented_reduce_cleanup(ctypes.byref(self.build_result))

    def __init__(
        self,
        d_in: DeviceArrayLike | IteratorBase,
        d_out: DeviceArrayLike,
        start_offsets_in: DeviceArrayLike | IteratorBase,
        end_offsets_in: DeviceArrayLike | IteratorBase,
        op: Callable,
        h_init: np.ndarray | GpuStruct,
    ):
        self.build_result = None
        self.d_in_cccl = cccl.to_cccl_iter(d_in)
        self.d_out_cccl = cccl.to_cccl_iter(d_out)
        self.start_offsets_in_cccl = cccl.to_cccl_iter(start_offsets_in)
        self.end_offsets_in_cccl = cccl.to_cccl_iter(end_offsets_in)
        self.h_init_cccl = cccl.to_cccl_value(h_init)
        cc_major, cc_minor = cuda.get_current_device().compute_capability
        cub_path, thrust_path, libcudacxx_path, cuda_include_path = get_paths()
        if isinstance(h_init, np.ndarray):
            value_type = numba.from_dtype(h_init.dtype)
        else:
            value_type = numba.typeof(h_init)
        sig = (value_type, value_type)
        self.op_wrapper = cccl.to_cccl_op(op, sig)
        self.build_result = cccl.DeviceSegmentedReduceBuildResult()
        self.bindings = get_bindings()
        error = self.bindings.cccl_device_segmented_reduce_build(
            ctypes.byref(self.build_result),
            self.d_in_cccl,
            self.d_out_cccl,
            self.start_offsets_in_cccl,
            self.end_offsets_in_cccl,
            self.op_wrapper,
            self.h_init_cccl,
            cc_major,
            cc_minor,
            ctypes.c_char_p(cub_path),
            ctypes.c_char_p(thrust_path),
            ctypes.c_char_p(libcudacxx_path),
            ctypes.c_char_p(cuda_include_path),
        )
        if error != enums.CUDA_SUCCESS:
            raise ValueError("Error building reduce")

    @staticmethod
    def _set_iterator(_in_cccl, _in):
        if _in_cccl.type.value == cccl.IteratorKind.POINTER:
            _in_cccl.state = protocols.get_data_pointer(_in)
        else:
            _in_cccl.state = _in.state

    def __call__(
        self,
        temp_storage,
        d_in,
        d_out,
        num_segments: int,
        start_offsets_in,
        end_offsets_in,
        h_init,
        stream=None,
    ):
        _SegmentedReduce._set_iterator(self.d_in_cccl, d_in)
        _SegmentedReduce._set_iterator(self.d_out_cccl, d_out)
        _SegmentedReduce._set_iterator(self.start_offsets_in_cccl, start_offsets_in)
        _SegmentedReduce._set_iterator(self.end_offsets_in_cccl, end_offsets_in)
        self.h_init_cccl.state = h_init.__array_interface__["data"][0]

        stream_handle = protocols.validate_and_get_stream(stream)

        if temp_storage is None:
            temp_storage_bytes = ctypes.c_size_t()
            d_temp_storage = None
        else:
            temp_storage_bytes = ctypes.c_size_t(temp_storage.nbytes)
            d_temp_storage = protocols.get_data_pointer(temp_storage)

        error = self.bindings.cccl_device_segmented_reduce(
            self.build_result,
            ctypes.c_void_p(d_temp_storage),
            ctypes.byref(temp_storage_bytes),
            self.d_in_cccl,
            self.d_out_cccl,
            ctypes.c_ulonglong(num_segments),
            self.start_offsets_in_cccl,
            self.end_offsets_in_cccl,
            self.op_wrapper,
            self.h_init_cccl,
            ctypes.c_void_p(stream_handle),
        )

        if error != enums.CUDA_SUCCESS:
            raise ValueError("Error reducing")

        return temp_storage_bytes.value


def _to_key(d_in: DeviceArrayLike | IteratorBase):
    "Return key for an input array-like argument or an iterator"
    d_in_key = (
        d_in.kind if isinstance(d_in, IteratorBase) else protocols.get_dtype(d_in)
    )
    return d_in_key


def make_cache_key(
    d_in: DeviceArrayLike | IteratorBase,
    d_out: DeviceArrayLike,
    start_offsets_in: DeviceArrayLike | IteratorBase,
    end_offsets_in: DeviceArrayLike | IteratorBase,
    op: Callable,
    h_init: np.ndarray,
):
    d_in_key = _to_key(d_in)
    d_out_key = protocols.get_dtype(d_out)
    start_offsets_in_key = _to_key(start_offsets_in)
    end_offsets_in_key = _to_key(end_offsets_in)
    op_key = CachableFunction(op)
    h_init_key = h_init.dtype
    return (
        d_in_key,
        d_out_key,
        start_offsets_in_key,
        end_offsets_in_key,
        op_key,
        h_init_key,
    )


@cache_with_key(make_cache_key)
def segmented_reduce(
    d_in: DeviceArrayLike | IteratorBase,
    d_out: DeviceArrayLike,
    start_offsets_in: DeviceArrayLike | IteratorBase,
    end_offsets_in: DeviceArrayLike | IteratorBase,
    op: Callable,
    h_init: np.ndarray,
):
    """Computes a device-wide segmented reduction using the specified binary ``op`` and initial value ``init``.

    Example:
        Below, ``segmented_reduce`` is used to compute the minimum value of a sequence of integers.

        .. literalinclude:: ../../python/cuda_parallel/tests/test_segmented_reduce_api.py
            :language: python
            :dedent:
            :start-after: example-begin reduce-min
            :end-before: example-end reduce-min

    Args:
        d_in: Device array or iterator containing the input sequence of data items
        d_out: Device array that will store the result of the reduction
        start_offsets_in: Device array or iterator containing offsets to start of segments
        end_offsets_in: Device array or iterator containing offsets to end of segments
        op: Callable representing the binary operator to apply
        init: Numpy array storing initial value of the reduction

    Returns:
        A callable object that can be used to perform the reduction
    """
    return _SegmentedReduce(d_in, d_out, start_offsets_in, end_offsets_in, op, h_init)
