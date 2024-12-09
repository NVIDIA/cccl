# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import importlib
import ctypes
import shutil
import numba
import os

from numba import cuda, types
from numba.cuda.cudadrv import enums


# MUST match `cccl_type_enum` in c/include/cccl/c/types.h
class _TypeEnum(ctypes.c_int):
    INT8 = 0
    INT16 = 1
    INT32 = 2
    INT64 = 3
    UINT8 = 4
    UINT16 = 5
    UINT32 = 6
    UINT64 = 7
    FLOAT32 = 8
    FLOAT64 = 9
    STORAGE = 10


def _cccl_type_enum_as_name(enum_value):
    assert isinstance(enum_value, int)
    return (
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float32",
        "float64",
        "STORAGE",
    )[enum_value]


# MUST match `cccl_op_kind_t` in c/include/cccl/c/types.h
class _CCCLOpKindEnum(ctypes.c_int):
    STATELESS = 0
    STATEFUL = 1


# MUST match `cccl_iterator_kind_t` in c/include/cccl/c/types.h
class _CCCLIteratorKindEnum(ctypes.c_int):
    POINTER = 0
    ITERATOR = 1


def _type_to_enum(numba_type):
    mapping = {
        types.int8: _TypeEnum.INT8,
        types.int16: _TypeEnum.INT16,
        types.int32: _TypeEnum.INT32,
        types.int64: _TypeEnum.INT64,
        types.uint8: _TypeEnum.UINT8,
        types.uint16: _TypeEnum.UINT16,
        types.uint32: _TypeEnum.UINT32,
        types.uint64: _TypeEnum.UINT64,
        types.float32: _TypeEnum.FLOAT32,
        types.float64: _TypeEnum.FLOAT64,
    }
    if numba_type in mapping:
        return mapping[numba_type]
    return _TypeEnum.STORAGE


# MUST match `cccl_type_info` in c/include/cccl/c/types.h
class _TypeInfo(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_int),
        ("alignment", ctypes.c_int),
        ("type", _TypeEnum),
    ]


# MUST match `cccl_op_t` in c/include/cccl/c/types.h
class _CCCLOp(ctypes.Structure):
    _fields_ = [
        ("type", _CCCLOpKindEnum),
        ("name", ctypes.c_char_p),
        ("ltoir", ctypes.c_char_p),
        ("ltoir_size", ctypes.c_int),
        ("size", ctypes.c_int),
        ("alignment", ctypes.c_int),
        ("state", ctypes.c_void_p),
    ]


# MUST match `cccl_string_view` in c/include/cccl/c/types.h
class _CCCLStringView(ctypes.Structure):
    _fields_ = [("begin", ctypes.c_char_p), ("size", ctypes.c_int)]


# MUST match `cccl_string_views` in c/include/cccl/c/types.h
class _CCCLStringViews(ctypes.Structure):
    _fields_ = [("views", ctypes.POINTER(_CCCLStringView)), ("size", ctypes.c_int)]


# MUST match `cccl_iterator_t` in c/include/cccl/c/types.h
class _CCCLIterator(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_int),
        ("alignment", ctypes.c_int),
        ("type", _CCCLIteratorKindEnum),
        ("advance", _CCCLOp),
        ("dereference", _CCCLOp),
        ("value_type", _TypeInfo),
        ("state", ctypes.c_void_p),
        ("ltoirs", ctypes.POINTER(_CCCLStringViews)),
    ]


# MUST match `cccl_value_t` in c/include/cccl/c/types.h
class _CCCLValue(ctypes.Structure):
    _fields_ = [("type", _TypeInfo), ("state", ctypes.c_void_p)]


def _type_to_info_from_numba_type(numba_type):
    context = cuda.descriptor.cuda_target.target_context
    size = context.get_value_type(numba_type).get_abi_size(context.target_data)
    alignment = context.get_value_type(numba_type).get_abi_alignment(
        context.target_data
    )
    return _TypeInfo(size, alignment, _type_to_enum(numba_type))


def _type_to_info(numpy_type):
    numba_type = numba.from_dtype(numpy_type)
    return _type_to_info_from_numba_type(numba_type)


def _device_array_to_pointer(array):
    dtype = array.dtype
    info = _type_to_info(dtype)
    # Note: this is slightly slower, but supports all ndarray-like objects as long as they support CAI
    # TODO: switch to use gpumemoryview once it's ready
    return _CCCLIterator(
        1,
        1,
        _CCCLIteratorKindEnum.POINTER,
        _CCCLOp(),
        _CCCLOp(),
        info,
        array.__cuda_array_interface__["data"][0],
        None,
    )


def _host_array_to_value(array):
    dtype = array.dtype
    info = _type_to_info(dtype)
    return _CCCLValue(info, array.ctypes.data)


class _Op:
    def __init__(self, dtype, op):
        value_type = numba.from_dtype(dtype)
        self.ltoir, _ = cuda.compile(
            op, sig=value_type(value_type, value_type), output="ltoir"
        )
        self.name = op.__name__.encode("utf-8")

    def handle(self):
        return _CCCLOp(
            _CCCLOpKindEnum.STATELESS,
            self.name,
            ctypes.c_char_p(self.ltoir),
            len(self.ltoir),
            1,
            1,
            None,
        )


def _extract_ctypes_ltoirs(numba_cuda_compile_results):
    view_lst = [
        _CCCLStringView(ltoir, len(ltoir)) for ltoir, _ in numba_cuda_compile_results
    ]
    view_arr = (_CCCLStringView * len(view_lst))(*view_lst)
    return ctypes.pointer(_CCCLStringViews(view_arr, len(view_arr)))


def _facade_iter_as_cccl_iter(d_in):
    def prefix_name(name):
        return (d_in.prefix + "_" + name).encode("utf-8")

    # type name ltoi ltoir_size size alignment state
    adv = _CCCLOp(
        _CCCLOpKindEnum.STATELESS, prefix_name("advance"), None, 0, 1, 1, None
    )
    drf = _CCCLOp(
        _CCCLOpKindEnum.STATELESS, prefix_name("dereference"), None, 0, 1, 1, None
    )
    info = _type_to_info_from_numba_type(d_in.ntype)
    ltoirs = _extract_ctypes_ltoirs(d_in.ltoirs)
    # size alignment type advance dereference value_type state ltoirs
    return _CCCLIterator(
        d_in.size,
        d_in.alignment,
        _CCCLIteratorKindEnum.ITERATOR,
        adv,
        drf,
        info,
        d_in.state_c_void_p,
        ltoirs,
    )


def _d_in_as_cccl_iter(d_in):
    if hasattr(d_in, "ntype"):
        return _facade_iter_as_cccl_iter(d_in)
    assert hasattr(d_in, "dtype")
    return _device_array_to_pointer(d_in)


def _get_cuda_path():
    cuda_path = os.environ.get("CUDA_PATH", "")
    if os.path.exists(cuda_path):
        return cuda_path

    nvcc_path = shutil.which("nvcc")
    if nvcc_path is not None:
        return os.path.dirname(os.path.dirname(nvcc_path))

    default_path = "/usr/local/cuda"
    if os.path.exists(default_path):
        return default_path

    return None


_bindings = None
_paths = None


def _get_bindings():
    global _bindings
    if _bindings is None:
        include_path = importlib.resources.files("cuda.parallel.experimental").joinpath(
            "cccl"
        )
        cccl_c_path = os.path.join(include_path, "libcccl.c.parallel.so")
        _bindings = ctypes.CDLL(cccl_c_path)
        _bindings.cccl_device_reduce.restype = ctypes.c_int
        _bindings.cccl_device_reduce.argtypes = [
            _CCCLDeviceReduceBuildResult,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_ulonglong),
            _CCCLIterator,
            _CCCLIterator,
            ctypes.c_ulonglong,
            _CCCLOp,
            _CCCLValue,
            ctypes.c_void_p,
        ]
        _bindings.cccl_device_reduce_cleanup.restype = ctypes.c_int
    return _bindings


def _get_paths():
    global _paths
    if _paths is None:
        # Using `.parent` for compatibility with pip install --editable:
        include_path = importlib.resources.files("cuda.parallel").parent.joinpath(
            "_include"
        )
        include_path_str = str(include_path)
        include_option = "-I" + include_path_str
        cub_path = include_option.encode("utf-8")
        thrust_path = cub_path
        libcudacxx_path_str = str(os.path.join(include_path, "libcudacxx"))
        libcudacxx_option = "-I" + libcudacxx_path_str
        libcudacxx_path = libcudacxx_option.encode("utf-8")
        cuda_include_str = os.path.join(_get_cuda_path(), "include")
        cuda_include_option = "-I" + cuda_include_str
        cuda_include_path = cuda_include_option.encode("utf-8")
        _paths = cub_path, thrust_path, libcudacxx_path, cuda_include_path
    return _paths


# MUST match `cccl_device_reduce_build_result_t` in c/include/cccl/c/reduce.h
class _CCCLDeviceReduceBuildResult(ctypes.Structure):
    _fields_ = [
        ("cc", ctypes.c_int),
        ("cubin", ctypes.c_void_p),
        ("cubin_size", ctypes.c_size_t),
        ("library", ctypes.c_void_p),
        ("accumulator_size", ctypes.c_ulonglong),
        ("single_tile_kernel", ctypes.c_void_p),
        ("single_tile_second_kernel", ctypes.c_void_p),
        ("reduction_kernel", ctypes.c_void_p),
    ]


def _dtype_validation(dt1, dt2):
    if dt1 != dt2:
        raise TypeError(f"dtype mismatch: __init__={dt1}, __call__={dt2}")


class _Reduce:
    def __init__(self, d_in, d_out, op, init):
        d_in_cccl = _d_in_as_cccl_iter(d_in)
        self._ctor_d_in_cccl_type_enum_name = _cccl_type_enum_as_name(
            d_in_cccl.value_type.type.value
        )
        self._ctor_d_out_dtype = d_out.dtype
        self._ctor_init_dtype = init.dtype
        cc_major, cc_minor = cuda.get_current_device().compute_capability
        cub_path, thrust_path, libcudacxx_path, cuda_include_path = _get_paths()
        bindings = _get_bindings()
        accum_t = init.dtype
        self.op_wrapper = _Op(accum_t, op)
        d_out_ptr = _device_array_to_pointer(d_out)
        self.build_result = _CCCLDeviceReduceBuildResult()

        # TODO Figure out caching
        error = bindings.cccl_device_reduce_build(
            ctypes.byref(self.build_result),
            d_in_cccl,
            d_out_ptr,
            self.op_wrapper.handle(),
            _host_array_to_value(init),
            cc_major,
            cc_minor,
            ctypes.c_char_p(cub_path),
            ctypes.c_char_p(thrust_path),
            ctypes.c_char_p(libcudacxx_path),
            ctypes.c_char_p(cuda_include_path),
        )
        if error != enums.CUDA_SUCCESS:
            raise ValueError("Error building reduce")

    def __call__(self, temp_storage, d_in, d_out, num_items, init):
        d_in_cccl = _d_in_as_cccl_iter(d_in)
        if d_in_cccl.type.value == _CCCLIteratorKindEnum.ITERATOR:
            assert num_items is not None
        else:
            assert d_in_cccl.type.value == _CCCLIteratorKindEnum.POINTER
            if num_items is None:
                num_items = d_in.size
            else:
                assert num_items == d_in.size
        _dtype_validation(
            self._ctor_d_in_cccl_type_enum_name,
            _cccl_type_enum_as_name(d_in_cccl.value_type.type.value),
        )
        _dtype_validation(self._ctor_d_out_dtype, d_out.dtype)
        _dtype_validation(self._ctor_init_dtype, init.dtype)
        bindings = _get_bindings()
        if temp_storage is None:
            temp_storage_bytes = ctypes.c_size_t()
            d_temp_storage = None
        else:
            temp_storage_bytes = ctypes.c_size_t(temp_storage.nbytes)
            # Note: this is slightly slower, but supports all ndarray-like objects as long as they support CAI
            # TODO: switch to use gpumemoryview once it's ready
            d_temp_storage = temp_storage.__cuda_array_interface__["data"][0]
        d_out_ptr = _device_array_to_pointer(d_out)
        error = bindings.cccl_device_reduce(
            self.build_result,
            d_temp_storage,
            ctypes.byref(temp_storage_bytes),
            d_in_cccl,
            d_out_ptr,
            ctypes.c_ulonglong(num_items),
            self.op_wrapper.handle(),
            _host_array_to_value(init),
            None,
        )
        if error != enums.CUDA_SUCCESS:
            raise ValueError("Error reducing")

        return temp_storage_bytes.value

    def __del__(self):
        bindings = _get_bindings()
        bindings.cccl_device_reduce_cleanup(ctypes.byref(self.build_result))


# TODO Figure out iterators
# TODO Figure out `sum` without operator and initial value
# TODO Accept stream
def reduce_into(d_in, d_out, op, init):
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
    return _Reduce(d_in, d_out, op, init)
