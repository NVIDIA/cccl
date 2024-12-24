# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import ctypes
import numba
import numpy as np
from numba import cuda
from numba.cuda.cudadrv import enums
from typing import Callable

from .. import _cccl as cccl
from .._bindings import get_paths, get_bindings


class _Op:
    def __init__(self, dtype: np.dtype, op: Callable):
        value_type = numba.from_dtype(dtype)
        self.ltoir, _ = cuda.compile(
            op, sig=value_type(value_type, value_type), output="ltoir"
        )
        self.name = op.__name__.encode("utf-8")

    def handle(self) -> cccl.Op:
        return cccl.Op(
            cccl.OpKind.STATELESS,
            self.name,
            ctypes.c_char_p(self.ltoir),
            len(self.ltoir),
            1,
            1,
            None,
        )


def _dtype_validation(dt1, dt2):
    if dt1 != dt2:
        raise TypeError(f"dtype mismatch: __init__={dt1}, __call__={dt2}")


class _Reduce:
    # TODO: constructor shouldn't require concrete `d_in`, `d_out`:
    def __init__(self, d_in, d_out, op: Callable, h_init: np.ndarray):
        d_in_cccl = cccl.to_cccl_iter(d_in)
        self._ctor_d_in_cccl_type_enum_name = cccl.type_enum_as_name(
            d_in_cccl.value_type.type.value
        )
        self._ctor_d_out_dtype = d_out.dtype
        self._ctor_init_dtype = h_init.dtype
        cc_major, cc_minor = cuda.get_current_device().compute_capability
        cub_path, thrust_path, libcudacxx_path, cuda_include_path = get_paths()
        bindings = get_bindings()
        self.op_wrapper = _Op(h_init.dtype, op)
        d_out_cccl = cccl.to_cccl_iter(d_out)
        self.build_result = cccl.DeviceReduceBuildResult()

        # TODO Figure out caching
        error = bindings.cccl_device_reduce_build(
            ctypes.byref(self.build_result),
            d_in_cccl,
            d_out_cccl,
            self.op_wrapper.handle(),
            cccl.host_array_to_value(h_init),
            cc_major,
            cc_minor,
            ctypes.c_char_p(cub_path),
            ctypes.c_char_p(thrust_path),
            ctypes.c_char_p(libcudacxx_path),
            ctypes.c_char_p(cuda_include_path),
        )
        if error != enums.CUDA_SUCCESS:
            raise ValueError("Error building reduce")

    def __call__(self, temp_storage, d_in, d_out, num_items: int, h_init: np.ndarray):
        d_in_cccl = cccl.to_cccl_iter(d_in)
        if d_in_cccl.type.value == cccl.IteratorKind.ITERATOR:
            assert num_items is not None
        else:
            assert d_in_cccl.type.value == cccl.IteratorKind.POINTER
            if num_items is None:
                num_items = d_in.size
            else:
                assert num_items == d_in.size
        _dtype_validation(
            self._ctor_d_in_cccl_type_enum_name,
            cccl.type_enum_as_name(d_in_cccl.value_type.type.value),
        )
        _dtype_validation(self._ctor_d_out_dtype, d_out.dtype)
        _dtype_validation(self._ctor_init_dtype, h_init.dtype)
        bindings = get_bindings()
        if temp_storage is None:
            temp_storage_bytes = ctypes.c_size_t()
            d_temp_storage = None
        else:
            temp_storage_bytes = ctypes.c_size_t(temp_storage.nbytes)
            # Note: this is slightly slower, but supports all ndarray-like objects as long as they support CAI
            # TODO: switch to use gpumemoryview once it's ready
            d_temp_storage = temp_storage.__cuda_array_interface__["data"][0]
        d_out_cccl = cccl.to_cccl_iter(d_out)
        error = bindings.cccl_device_reduce(
            self.build_result,
            d_temp_storage,
            ctypes.byref(temp_storage_bytes),
            d_in_cccl,
            d_out_cccl,
            ctypes.c_ulonglong(num_items),
            self.op_wrapper.handle(),
            cccl.host_array_to_value(h_init),
            None,
        )
        if error != enums.CUDA_SUCCESS:
            raise ValueError("Error reducing")

        return temp_storage_bytes.value

    def __del__(self):
        bindings = get_bindings()
        bindings.cccl_device_reduce_cleanup(ctypes.byref(self.build_result))


# TODO Figure out `sum` without operator and initial value
# TODO Accept stream
def reduce_into(d_in, d_out, op: Callable, h_init: np.ndarray):
    """Computes a device-wide reduction using the specified binary ``op`` functor and initial value ``init``.

    Example:
        The code snippet below illustrates a user-defined min-reduction of a
        device vector of ``int`` data elements.

        .. literalinclude:: ../../python/cuda_parallel/tests/test_reduce_api.py
            :language: python
            :dedent:
            :start-after: example-begin imports
            :end-before: example-end imports

        Below is the code snippet that demonstrates the usage of the ``reduce_into`` API:

        .. literalinclude:: ../../python/cuda_parallel/tests/test_reduce_api.py
            :language: python
            :dedent:
            :start-after: example-begin reduce-min
            :end-before: example-end reduce-min

    Args:
        d_in: CUDA device array storing the input sequence of data items
        d_out: CUDA device array storing the output aggregate
        op: Binary reduction
        init: Numpy array storing initial value of the reduction

    Returns:
        A callable object that can be used to perform the reduction
    """
    return _Reduce(d_in, d_out, op, h_init)
