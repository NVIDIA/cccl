# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# cython: language_level=3
# cython: freethreading_compatible=True

from cpython.mem cimport PyMem_Free, PyMem_Malloc
from cpython.object cimport PyObject
from cpython.tuple cimport PyTuple_New, PyTuple_SET_ITEM
from cpython.long cimport PyLong_FromLongLong
from cpython.ref cimport Py_INCREF
from libc.stdint cimport int64_t, uint64_t, uintptr_t

import operator


cdef extern from "cccl/c/device_copy.h":
    ctypedef enum cccl_device_copy_axis_metadata_kind_t:
        CCCL_DEVICE_COPY_AXIS_RUNTIME
        CCCL_DEVICE_COPY_AXIS_STATIC

    ctypedef enum cccl_device_copy_layout_kind_t:
        CCCL_DEVICE_COPY_LAYOUT_RIGHT
        CCCL_DEVICE_COPY_LAYOUT_LEFT
        CCCL_DEVICE_COPY_LAYOUT_STRIDE
        CCCL_DEVICE_COPY_LAYOUT_STRIDE_RELAXED

    cdef struct cccl_device_copy_axis_metadata_t:
        cccl_device_copy_axis_metadata_kind_t kind
        int64_t value

    cdef struct cccl_device_copy_source_view_t:
        const void* data
        uint64_t byte_offset
        const int64_t* shape
        const int64_t* strides

    cdef struct cccl_device_copy_destination_view_t:
        void* data
        uint64_t byte_offset
        const int64_t* shape
        const int64_t* strides


cdef object _as_index(object value, str name):
    try:
        return operator.index(value)
    except TypeError:
        raise TypeError(f"{name} must be an integer, got {type(value)}") from None


cdef uint64_t _as_uint64(object value, str name) except *:
    cdef object index = _as_index(value, name)
    if index < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    if index > 0xFFFFFFFFFFFFFFFF:
        raise OverflowError(f"{name} does not fit in uint64_t")
    return <uint64_t>index


cdef uintptr_t _as_uintptr(object value, str name) except *:
    cdef object index = _as_index(value, name)
    if index < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return <uintptr_t>index


cdef size_t _validate_rank(object values, str name) except *:
    cdef Py_ssize_t rank
    try:
        rank = len(values)
    except TypeError:
        raise TypeError(f"{name} must be a sequence") from None
    if rank == 0:
        return 1
    return <size_t>rank


cdef void* _alloc_array(size_t count, size_t item_size) except NULL:
    cdef void* result
    if item_size != 0 and count > (<size_t>-1) // item_size:
        raise MemoryError()
    result = PyMem_Malloc(count * item_size)
    if result == NULL:
        raise MemoryError()
    return result


cdef int64_t* _copy_int64_sequence(
    object values,
    size_t rank,
    str name,
    bint require_non_negative,
) except NULL:
    cdef int64_t* result = <int64_t*>_alloc_array(rank, sizeof(int64_t))
    cdef Py_ssize_t i
    cdef int64_t item

    try:
        for i in range(<Py_ssize_t>rank):
            item = <int64_t>_as_index(values[i], f"{name}[{i}]")
            if require_non_negative and item < 0:
                raise ValueError(f"{name}[{i}] must be non-negative, got {item}")
            result[i] = item
    except Exception:
        PyMem_Free(result)
        raise

    return result


cdef tuple _int64_tuple(const int64_t* values, size_t count):
    cdef object item
    cdef size_t i
    cdef tuple result = PyTuple_New(count)

    for i in range(count):
        item = PyLong_FromLongLong(<long long>values[i])
        # PyTuple_SET_ITEM steals a reference. item is a managed Cython
        # object local, so give the tuple its own reference first.
        Py_INCREF(item)
        PyTuple_SET_ITEM(result, <Py_ssize_t>i, <object>item)
    return result


cdef class _RuntimeAxisMetadata:
    cdef size_t _rank
    cdef cccl_device_copy_axis_metadata_t* _metadata

    def __cinit__(self):
        self._rank = 0
        self._metadata = NULL

    def __init__(self, object rank):
        cdef Py_ssize_t rank_value = <Py_ssize_t>_as_index(rank, "rank")
        cdef Py_ssize_t i

        if rank_value < 0:
            raise ValueError(f"rank must be non-negative, got {rank_value}")
        if rank_value == 0:
            rank_value = 1

        self._rank = <size_t>rank_value
        self._metadata = <cccl_device_copy_axis_metadata_t*>_alloc_array(
            self._rank, sizeof(cccl_device_copy_axis_metadata_t)
        )

        for i in range(rank_value):
            self._metadata[i].kind = CCCL_DEVICE_COPY_AXIS_RUNTIME
            self._metadata[i].value = 0

    def __dealloc__(self):
        if self._metadata != NULL:
            PyMem_Free(self._metadata)
            self._metadata = NULL

    cdef const cccl_device_copy_axis_metadata_t* _ptr(self) noexcept:
        return self._metadata

    @property
    def rank(self):
        return self._rank

    def _metadata_pointer(self):
        return <uintptr_t>self._metadata

    def _as_tuple(self):
        cdef tuple result = PyTuple_New(<Py_ssize_t>self._rank)
        cdef object item
        cdef size_t i

        for i in range(self._rank):
            item = (<int>self._metadata[i].kind, self._metadata[i].value)
            # PyTuple_SET_ITEM steals a reference; see _int64_tuple for details.
            Py_INCREF(item)
            PyTuple_SET_ITEM(result, <Py_ssize_t>i, item)
        return result


cdef class _PreparedDeviceCopyView:
    cdef object _owner
    cdef size_t _rank
    cdef uintptr_t _data_ptr
    cdef uint64_t _byte_offset
    cdef int64_t* _shape
    cdef int64_t* _strides

    def __cinit__(self):
        self._owner = None
        self._rank = 0
        self._data_ptr = 0
        self._byte_offset = 0
        self._shape = NULL
        self._strides = NULL

    def __init__(
        self,
        object owner,
        object data_ptr,
        object byte_offset,
        object shape,
        object strides=None,
    ):
        self._rank = _validate_rank(shape, "shape")
        if len(shape) == 0:
            if strides is not None and len(strides) != 0:
                raise ValueError("0-rank scalar strides must be empty when provided")
            shape = (1,)
            strides = (1,)
        elif strides is not None and len(strides) != self._rank:
            raise ValueError(
                f"strides rank must match shape rank {self._rank}, got {len(strides)}"
            )

        self._owner = owner
        self._data_ptr = _as_uintptr(data_ptr, "data_ptr")
        self._byte_offset = _as_uint64(byte_offset, "byte_offset")
        self._shape = _copy_int64_sequence(shape, self._rank, "shape", True)
        if strides is not None:
            self._strides = _copy_int64_sequence(strides, self._rank, "strides", False)

    def __dealloc__(self):
        if self._shape != NULL:
            PyMem_Free(self._shape)
            self._shape = NULL
        if self._strides != NULL:
            PyMem_Free(self._strides)
            self._strides = NULL

    cdef cccl_device_copy_source_view_t _as_source_view(self) noexcept:
        cdef cccl_device_copy_source_view_t view
        view.data = <const void*>self._data_ptr
        view.byte_offset = self._byte_offset
        view.shape = self._shape
        view.strides = self._strides
        return view

    cdef cccl_device_copy_destination_view_t _as_destination_view(self) noexcept:
        cdef cccl_device_copy_destination_view_t view
        view.data = <void*>self._data_ptr
        view.byte_offset = self._byte_offset
        view.shape = self._shape
        view.strides = self._strides
        return view

    @property
    def data_ptr(self):
        return self._data_ptr

    @property
    def byte_offset(self):
        return self._byte_offset

    @property
    def rank(self):
        return self._rank

    @property
    def shape(self):
        return _int64_tuple(self._shape, self._rank)

    @property
    def strides(self):
        if self._strides == NULL:
            return None
        return _int64_tuple(self._strides, self._rank)

    def _descriptor_pointers(self):
        return {
            "data": self._data_ptr,
            "shape": <uintptr_t>self._shape,
            "strides": <uintptr_t>self._strides,
        }

    def __repr__(self):
        return (
            f"{type(self).__name__}("
            f"data_ptr=0x{self._data_ptr:x}, "
            f"byte_offset={self._byte_offset}, "
            f"shape={self.shape!r}, "
            f"strides={self.strides!r})"
        )


def _make_runtime_axis_metadata(object rank):
    return _RuntimeAxisMetadata(rank)


def _prepare_runtime_strided_view(
    object owner,
    object data_ptr,
    object byte_offset,
    object shape,
    object strides=None,
):
    return _PreparedDeviceCopyView(owner, data_ptr, byte_offset, shape, strides)
