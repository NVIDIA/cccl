# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# cython: language_level=3
# cython: freethreading_compatible=True

from cpython.mem cimport PyMem_Free, PyMem_Malloc
from cpython.object cimport PyObject
from cpython.pycapsule cimport PyCapsule_GetPointer, PyCapsule_IsValid, PyCapsule_SetName
from cpython.tuple cimport PyTuple_New, PyTuple_SET_ITEM
from cpython.long cimport PyLong_FromLongLong
from cpython.ref cimport Py_INCREF
from libc.stdint cimport int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t, uintptr_t

import operator


# Keep these local Cython declarations in sync with DLPack v1.3 dlpack.h.
cdef extern from *:
    """
    #include <dlpack/dlpack.h>
    #if DLPACK_MAJOR_VERSION != 1 || DLPACK_MINOR_VERSION < 3
    #error "cuda.compute device copy requires DLPack v1.3 or a newer compatible 1.x dlpack.h"
    #endif
    """

cdef extern from "dlpack/dlpack.h":
    cdef enum:
        DLPACK_MAJOR_VERSION
        DLPACK_MINOR_VERSION

    ctypedef enum DLDeviceType:
        kDLCUDA
        kDLCUDAManaged

cdef const char* DLPACK_CAPSULE_NAME = "dltensor"
cdef const char* USED_DLPACK_CAPSULE_NAME = "used_dltensor"
cdef const char* DLPACK_VERSIONED_CAPSULE_NAME = "dltensor_versioned"
cdef const char* USED_DLPACK_VERSIONED_CAPSULE_NAME = "used_dltensor_versioned"


cdef extern from "dlpack/dlpack.h":
    ctypedef struct DLPackVersion:
        uint32_t major
        uint32_t minor


cdef extern from "dlpack/dlpack.h":
    ctypedef struct DLDevice:
        DLDeviceType device_type
        int32_t device_id


cdef extern from "dlpack/dlpack.h":
    ctypedef struct DLDataType:
        uint8_t code
        uint8_t bits
        uint16_t lanes


cdef extern from "dlpack/dlpack.h":
    ctypedef struct DLTensor:
        void* data
        DLDevice device
        int32_t ndim
        DLDataType dtype
        int64_t* shape
        int64_t* strides
        uint64_t byte_offset


cdef extern from "dlpack/dlpack.h":
    ctypedef struct DLManagedTensor:
        DLTensor dl_tensor
        void* manager_ctx
        void (*deleter)(DLManagedTensor* self)


cdef extern from "dlpack/dlpack.h":
    ctypedef struct DLManagedTensorVersioned:
        DLPackVersion version
        void* manager_ctx
        void (*deleter)(DLManagedTensorVersioned* self)
        uint64_t flags
        DLTensor dl_tensor


cdef extern from "dlpack/dlpack.h":
    ctypedef void (*DLPackSetError)(
        void* error_ctx,
        const char* kind,
        const char* message,
    )

cdef extern from "dlpack/dlpack.h":
    ctypedef int (*DLPackManagedTensorAllocator)(
        DLTensor* prototype,
        DLManagedTensorVersioned** out,
        void* error_ctx,
        DLPackSetError set_error,
    )

cdef extern from "dlpack/dlpack.h":
    ctypedef int (*DLPackManagedTensorFromPyObjectNoSync)(
        void* py_object,
        DLManagedTensorVersioned** out,
    )


cdef extern from "dlpack/dlpack.h":
    ctypedef int (*DLPackDLTensorFromPyObjectNoSync)(
        void* py_object,
        DLTensor* out,
    )


cdef extern from "dlpack/dlpack.h":
    ctypedef int (*DLPackManagedTensorToPyObjectNoSync)(
        DLManagedTensorVersioned* tensor,
        void** out_py_object,
    )

cdef extern from "dlpack/dlpack.h":
    ctypedef int (*DLPackCurrentWorkStream)(
        DLDeviceType device_type,
        int32_t device_id,
        void** out_current_stream,
    )


cdef extern from "dlpack/dlpack.h":
    ctypedef struct DLPackExchangeAPIHeader:
        DLPackVersion version
        DLPackExchangeAPIHeader* prev_api


cdef extern from "dlpack/dlpack.h":
    ctypedef struct DLPackExchangeAPI:
        DLPackExchangeAPIHeader header
        DLPackManagedTensorAllocator managed_tensor_allocator
        DLPackManagedTensorFromPyObjectNoSync managed_tensor_from_py_object_no_sync
        DLPackManagedTensorToPyObjectNoSync managed_tensor_to_py_object_no_sync
        DLPackDLTensorFromPyObjectNoSync dltensor_from_py_object_no_sync
        DLPackCurrentWorkStream current_work_stream


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


cdef class _DLPackManagedTensorOwner:
    cdef object _producer
    cdef DLManagedTensor* _legacy
    cdef DLManagedTensorVersioned* _versioned
    cdef void (*_legacy_deleter)(DLManagedTensor* self)
    cdef void (*_versioned_deleter)(DLManagedTensorVersioned* self)

    def __cinit__(self):
        self._legacy = NULL
        self._versioned = NULL
        self._legacy_deleter = NULL
        self._versioned_deleter = NULL

    def __dealloc__(self):
        if self._versioned != NULL:
            if self._versioned_deleter != NULL:
                self._versioned_deleter(self._versioned)
            self._versioned = NULL
        if self._legacy != NULL:
            if self._legacy_deleter != NULL:
                self._legacy_deleter(self._legacy)
            self._legacy = NULL

    cdef DLTensor* _tensor(self) noexcept:
        if self._versioned != NULL:
            return &self._versioned.dl_tensor
        if self._legacy != NULL:
            return &self._legacy.dl_tensor
        return NULL

    @property
    def device_type(self):
        cdef DLTensor* tensor = self._tensor()
        if tensor == NULL:
            return None
        return tensor.device.device_type

    @property
    def device_id(self):
        cdef DLTensor* tensor = self._tensor()
        if tensor == NULL:
            return None
        return tensor.device.device_id

    @property
    def dtype(self):
        cdef DLTensor* tensor = self._tensor()
        if tensor == NULL:
            return None
        return (tensor.dtype.code, tensor.dtype.bits, tensor.dtype.lanes)

    @property
    def flags(self):
        if self._versioned == NULL:
            return 0
        return self._versioned.flags

    def _tensor_pointer(self):
        return <uintptr_t>self._tensor()


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


cdef const char* DLPACK_EXCHANGE_API_CAPSULE_NAME = "dlpack_exchange_api"


def _layout_stride_relaxed():
    return <int>CCCL_DEVICE_COPY_LAYOUT_STRIDE_RELAXED


cdef:
    # The minimum guard avoids undefined behavior when taking abs(INT64_MIN).
    int64_t _COPY_PLAN_INT64_MIN = -0x7FFFFFFFFFFFFFFF - 1
    int64_t _COPY_PLAN_INT64_MAX = 0x7FFFFFFFFFFFFFFF


cdef int64_t _copy_plan_extent(object value) except? -1:
    cdef int64_t result = <int64_t>value
    if result < 0:
        raise ValueError("copy plan shape extents must be non-negative")
    return result


cdef int64_t _copy_plan_checked_i64(object value, str message) except? -1:
    try:
        return <int64_t>value
    except OverflowError:
        raise OverflowError(message)


cdef int64_t _copy_plan_checked_add(int64_t lhs, int64_t rhs, str message) except? -1:
    cdef object lhs_obj = lhs
    cdef object rhs_obj = rhs
    return _copy_plan_checked_i64(lhs_obj + rhs_obj, message)


cdef int64_t _copy_plan_checked_mul(int64_t lhs, int64_t rhs, str message) except? -1:
    cdef object lhs_obj = lhs
    cdef object rhs_obj = rhs
    return _copy_plan_checked_i64(lhs_obj * rhs_obj, message)


cdef int64_t _copy_plan_axis_delta(int64_t extent, int64_t stride, str message) except? -1:
    if extent == 0:
        return 0
    return _copy_plan_checked_mul(extent - 1, stride, message)


cdef int64_t _copy_plan_abs_stride(int64_t stride) except? -1:
    if stride == _COPY_PLAN_INT64_MIN:
        raise OverflowError("copy plan stride magnitude is too large")
    if stride < 0:
        return -stride
    return stride


cdef bint _copy_plan_abs_stride_noexcept(int64_t stride, int64_t* result) noexcept:
    if stride == _COPY_PLAN_INT64_MIN:
        return False
    if stride < 0:
        result[0] = -stride
    else:
        result[0] = stride
    return True


cdef bint _copy_plan_checked_nonnegative_add_noexcept(
    int64_t lhs,
    int64_t rhs,
    int64_t* result,
) noexcept:
    if lhs < 0 or rhs < 0:
        return False
    if lhs > _COPY_PLAN_INT64_MAX - rhs:
        return False
    result[0] = lhs + rhs
    return True


cdef bint _copy_plan_checked_nonnegative_mul_noexcept(
    int64_t lhs,
    int64_t rhs,
    int64_t* result,
) noexcept:
    if lhs < 0 or rhs < 0:
        return False
    if rhs != 0 and lhs > _COPY_PLAN_INT64_MAX // rhs:
        return False
    result[0] = lhs * rhs
    return True


cdef bint _copy_plan_is_unique_mapping(
    size_t rank,
    const int64_t* shape,
    const int64_t* strides,
    int64_t* axis_scratch,
) noexcept:
    cdef size_t axis_count = 0
    cdef size_t i
    cdef size_t j
    cdef size_t axis
    cdef size_t previous_axis
    cdef int64_t axis_stride
    cdef int64_t previous_stride
    cdef int64_t extent
    cdef int64_t covered = 1
    cdef int64_t delta

    if axis_scratch == NULL:
        return False

    for i in range(rank):
        extent = shape[i]
        if extent < 0:
            return False
        if extent <= 1:
            continue

        if not _copy_plan_abs_stride_noexcept(strides[i], &axis_stride):
            return False
        if axis_stride == 0:
            return False

        if i > <size_t>_COPY_PLAN_INT64_MAX:
            return False
        axis_scratch[axis_count] = <int64_t>i
        axis_count += 1

    for i in range(1, axis_count):
        axis = <size_t>axis_scratch[i]
        if not _copy_plan_abs_stride_noexcept(strides[axis], &axis_stride):
            return False
        j = i
        while j > 0:
            previous_axis = <size_t>axis_scratch[j - 1]
            if not _copy_plan_abs_stride_noexcept(strides[previous_axis], &previous_stride):
                return False
            if previous_stride < axis_stride:
                break
            if previous_stride == axis_stride and shape[previous_axis] <= shape[axis]:
                break
            axis_scratch[j] = axis_scratch[j - 1]
            j -= 1
        axis_scratch[j] = <int64_t>axis

    for i in range(axis_count):
        axis = <size_t>axis_scratch[i]
        if not _copy_plan_abs_stride_noexcept(strides[axis], &axis_stride):
            return False
        if axis_stride < covered:
            return False

        if not _copy_plan_checked_nonnegative_mul_noexcept(
            shape[axis] - 1,
            axis_stride,
            &delta,
        ):
            return False
        if not _copy_plan_checked_nonnegative_add_noexcept(covered, delta, &covered):
            return False

    return True


cdef void _copy_plan_require_unique_mapping(
    size_t rank,
    const int64_t* shape,
    const int64_t* strides,
    int64_t* axis_scratch,
    str label,
) except *:
    if not _copy_plan_is_unique_mapping(rank, shape, strides, axis_scratch):
        raise ValueError(label + " mapping is not unique or uniqueness could not be proven")


cdef int64_t* _copy_plan_alloc_int64(size_t count) except NULL:
    cdef int64_t* result
    if count == 0:
        count = 1
    result = <int64_t*>PyMem_Malloc(count * sizeof(int64_t))
    if result == NULL:
        raise MemoryError()
    return result


cdef class _DeviceCopyPlan:
    cdef size_t _original_rank
    cdef size_t _rank
    cdef int64_t _elements
    cdef int64_t _source_element_offset
    cdef int64_t _destination_element_offset
    cdef bint _empty
    cdef bint _contiguous_1d
    cdef int64_t* _original_shape
    cdef int64_t* _original_source_strides
    cdef int64_t* _original_destination_strides
    cdef int64_t* _axis_order
    cdef int64_t* _axis_scratch
    cdef int64_t* _shape
    cdef int64_t* _source_strides
    cdef int64_t* _destination_strides

    def __cinit__(self):
        self._original_rank = 0
        self._rank = 0
        self._elements = 0
        self._source_element_offset = 0
        self._destination_element_offset = 0
        self._empty = False
        self._contiguous_1d = False
        self._original_shape = NULL
        self._original_source_strides = NULL
        self._original_destination_strides = NULL
        self._axis_order = NULL
        self._axis_scratch = NULL
        self._shape = NULL
        self._source_strides = NULL
        self._destination_strides = NULL

    def __dealloc__(self):
        if self._original_shape != NULL:
            PyMem_Free(self._original_shape)
        if self._original_source_strides != NULL:
            PyMem_Free(self._original_source_strides)
        if self._original_destination_strides != NULL:
            PyMem_Free(self._original_destination_strides)
        if self._axis_order != NULL:
            PyMem_Free(self._axis_order)
        if self._axis_scratch != NULL:
            PyMem_Free(self._axis_scratch)
        if self._shape != NULL:
            PyMem_Free(self._shape)
        if self._source_strides != NULL:
            PyMem_Free(self._source_strides)
        if self._destination_strides != NULL:
            PyMem_Free(self._destination_strides)

    def __init__(
        self,
        object shape,
        object source_strides,
        object destination_strides,
        object source_element_offset=0,
        object destination_element_offset=0,
    ):
        cdef Py_ssize_t shape_rank = len(shape)
        if len(source_strides) != shape_rank:
            raise ValueError("source strides rank must match shape rank")
        if len(destination_strides) != shape_rank:
            raise ValueError("destination strides rank must match shape rank")

        self._original_rank = <size_t>shape_rank
        self._source_element_offset = <int64_t>source_element_offset
        self._destination_element_offset = <int64_t>destination_element_offset
        self._allocate(<size_t>shape_rank)
        self._load_original(shape, source_strides, destination_strides)
        self._elements = self._compute_elements()
        self._empty = self._elements == 0

        self._validate_destination_unique()
        self._validate_element_interval(
            self._source_element_offset,
            self._original_source_strides,
            "source array view refers to elements before its allocation base",
        )
        self._validate_element_interval(
            self._destination_element_offset,
            self._original_destination_strides,
            "destination array view refers to elements before its allocation base",
        )
        self._build_normalized()

    cdef void _allocate(self, size_t rank) except *:
        self._original_shape = _copy_plan_alloc_int64(rank)
        self._original_source_strides = _copy_plan_alloc_int64(rank)
        self._original_destination_strides = _copy_plan_alloc_int64(rank)
        self._axis_order = _copy_plan_alloc_int64(rank)
        self._axis_scratch = _copy_plan_alloc_int64(rank)
        self._shape = _copy_plan_alloc_int64(rank)
        self._source_strides = _copy_plan_alloc_int64(rank)
        self._destination_strides = _copy_plan_alloc_int64(rank)

    cdef void _load_original(self, object shape, object source_strides, object destination_strides) except *:
        cdef size_t i
        for i in range(self._original_rank):
            self._original_shape[i] = _copy_plan_extent(shape[i])
            self._original_source_strides[i] = <int64_t>source_strides[i]
            self._original_destination_strides[i] = <int64_t>destination_strides[i]

    cdef int64_t _compute_elements(self) except? -1:
        cdef int64_t elements = 1
        cdef size_t i
        cdef int64_t extent
        for i in range(self._original_rank):
            extent = self._original_shape[i]
            if extent == 0:
                return 0
            elements = _copy_plan_checked_mul(elements, extent, "copy plan element count is too large")
        return elements

    cdef void _validate_destination_unique(self) except *:
        _copy_plan_require_unique_mapping(
            self._original_rank,
            self._original_shape,
            self._original_destination_strides,
            self._axis_scratch,
            "destination",
        )

    cdef bint _is_unique_mapping(self, size_t rank, const int64_t* shape, const int64_t* strides) noexcept:
        return _copy_plan_is_unique_mapping(rank, shape, strides, self._axis_scratch)

    cdef void _validate_element_interval(
        self,
        int64_t element_offset,
        const int64_t* strides,
        str message,
    ) except *:
        cdef int64_t minimum = element_offset
        cdef int64_t delta
        cdef size_t i
        if self._empty:
            return
        for i in range(self._original_rank):
            delta = _copy_plan_axis_delta(
                self._original_shape[i],
                strides[i],
                "array offset span is too large",
            )
            if delta < 0:
                minimum = _copy_plan_checked_add(minimum, delta, "array offset span is too large")
        if minimum < 0:
            raise ValueError(message)

    cdef void _build_normalized(self) except *:
        cdef size_t i
        cdef size_t write = 0
        cdef int64_t extent
        cdef int64_t source_stride
        cdef int64_t destination_stride

        if self._empty:
            self._rank = 0
            self._contiguous_1d = True
            return

        for i in range(self._original_rank):
            extent = self._original_shape[i]
            if extent == 1:
                continue

            source_stride = self._original_source_strides[i]
            destination_stride = self._original_destination_strides[i]
            if destination_stride < 0:
                self._source_element_offset = _copy_plan_checked_add(
                    self._source_element_offset,
                    _copy_plan_axis_delta(extent, source_stride, "normalized source offset is too large"),
                    "normalized source offset is too large",
                )
                self._destination_element_offset = _copy_plan_checked_add(
                    self._destination_element_offset,
                    _copy_plan_axis_delta(extent, destination_stride, "normalized destination offset is too large"),
                    "normalized destination offset is too large",
                )
                source_stride = _copy_plan_checked_mul(source_stride, -1, "normalized source stride magnitude is too large")
                destination_stride = _copy_plan_checked_mul(
                    destination_stride,
                    -1,
                    "normalized destination stride magnitude is too large",
                )

            self._axis_order[write] = <int64_t>i
            self._shape[write] = extent
            self._source_strides[write] = source_stride
            self._destination_strides[write] = destination_stride
            write += 1

        self._rank = write
        self._sort_axes()
        self._collapse_axes()
        self._contiguous_1d = self._rank <= 1 and (
            self._rank == 0
            or (self._source_strides[0] == 1 and self._destination_strides[0] == 1)
        )

    cdef int64_t _axis_sort_key_destination(self, size_t axis) except? -1:
        return _copy_plan_abs_stride(self._destination_strides[axis])

    cdef int64_t _axis_sort_key_source(self, size_t axis) except? -1:
        return _copy_plan_abs_stride(self._source_strides[axis])

    cdef bint _axis_should_move_left(self, size_t lhs, size_t rhs) except *:
        cdef int64_t lhs_destination = self._axis_sort_key_destination(lhs)
        cdef int64_t rhs_destination = self._axis_sort_key_destination(rhs)
        if lhs_destination != rhs_destination:
            return lhs_destination < rhs_destination

        cdef int64_t lhs_source = self._axis_sort_key_source(lhs)
        cdef int64_t rhs_source = self._axis_sort_key_source(rhs)
        if lhs_source != rhs_source:
            return lhs_source < rhs_source

        return self._axis_order[lhs] > self._axis_order[rhs]

    cdef void _swap_axes(self, size_t lhs, size_t rhs) noexcept:
        cdef int64_t temporary
        temporary = self._axis_order[lhs]
        self._axis_order[lhs] = self._axis_order[rhs]
        self._axis_order[rhs] = temporary

        temporary = self._shape[lhs]
        self._shape[lhs] = self._shape[rhs]
        self._shape[rhs] = temporary

        temporary = self._source_strides[lhs]
        self._source_strides[lhs] = self._source_strides[rhs]
        self._source_strides[rhs] = temporary

        temporary = self._destination_strides[lhs]
        self._destination_strides[lhs] = self._destination_strides[rhs]
        self._destination_strides[rhs] = temporary

    cdef void _sort_axes(self) except *:
        cdef size_t i
        cdef size_t j
        for i in range(1, self._rank):
            j = i
            while j > 0 and self._axis_should_move_left(j - 1, j):
                self._swap_axes(j - 1, j)
                j -= 1

    cdef bint _can_collapse(self, size_t outer, size_t inner) except *:
        if self._shape[outer] == 0 or self._shape[inner] == 0:
            return True
        return (
            self._source_strides[outer]
            == _copy_plan_checked_mul(
                self._source_strides[inner],
                self._shape[inner],
                "source stride span is too large",
            )
            and self._destination_strides[outer]
            == _copy_plan_checked_mul(
                self._destination_strides[inner],
                self._shape[inner],
                "destination stride span is too large",
            )
        )

    cdef void _collapse_axes(self) except *:
        cdef size_t read
        cdef size_t write = 0

        for read in range(self._rank):
            if write != 0 and self._can_collapse(write - 1, read):
                self._shape[write - 1] = _copy_plan_checked_mul(
                    self._shape[write - 1],
                    self._shape[read],
                    "collapsed extent is too large",
                )
                if self._axis_order[read] < self._axis_order[write - 1]:
                    self._axis_order[write - 1] = self._axis_order[read]
                self._source_strides[write - 1] = self._source_strides[read]
                self._destination_strides[write - 1] = self._destination_strides[read]
                continue

            if write != read:
                self._axis_order[write] = self._axis_order[read]
                self._shape[write] = self._shape[read]
                self._source_strides[write] = self._source_strides[read]
                self._destination_strides[write] = self._destination_strides[read]
            write += 1

        self._rank = write

    cdef size_t _native_rank(self) noexcept:
        return self._rank

    cdef const int64_t* _native_shape(self) noexcept:
        return self._shape

    cdef const int64_t* _native_source_strides(self) noexcept:
        return self._source_strides

    cdef const int64_t* _native_destination_strides(self) noexcept:
        return self._destination_strides

    cdef int64_t _native_source_element_offset(self) noexcept:
        return self._source_element_offset

    cdef int64_t _native_destination_element_offset(self) noexcept:
        return self._destination_element_offset

    @property
    def original_rank(self):
        return self._original_rank

    @property
    def rank(self):
        return self._rank

    @property
    def elements(self):
        return self._elements

    @property
    def source_element_offset(self):
        return self._source_element_offset

    @property
    def destination_element_offset(self):
        return self._destination_element_offset

    @property
    def empty(self):
        return bool(self._empty)

    @property
    def contiguous_1d(self):
        return bool(self._contiguous_1d)

    @property
    def original_shape(self):
        return _int64_tuple(self._original_shape, self._original_rank)

    @property
    def original_source_strides(self):
        return _int64_tuple(self._original_source_strides, self._original_rank)

    @property
    def original_destination_strides(self):
        return _int64_tuple(self._original_destination_strides, self._original_rank)

    @property
    def axis_order(self):
        return _int64_tuple(self._axis_order, self._rank)

    @property
    def shape(self):
        return _int64_tuple(self._shape, self._rank)

    @property
    def source_strides(self):
        return _int64_tuple(self._source_strides, self._rank)

    @property
    def destination_strides(self):
        return _int64_tuple(self._destination_strides, self._rank)

    @property
    def original_source_unique(self):
        return bool(self._is_unique_mapping(
            self._original_rank,
            self._original_shape,
            self._original_source_strides,
        ))

    @property
    def source_unique(self):
        return bool(self._is_unique_mapping(
            self._rank,
            self._shape,
            self._source_strides,
        ))

    @property
    def destination_unique(self):
        return bool(self._is_unique_mapping(
            self._rank,
            self._shape,
            self._destination_strides,
        ))


def _make_device_copy_plan(
    object shape,
    object source_strides,
    object destination_strides,
    object source_element_offset=0,
    object destination_element_offset=0,
):
    return _DeviceCopyPlan(
        shape,
        source_strides,
        destination_strides,
        source_element_offset,
        destination_element_offset,
    )


cdef DLPackExchangeAPI* _dlpack_exchange_api(object obj) except? NULL:
    cdef object capsule
    cdef DLPackExchangeAPI* api
    cdef DLPackExchangeAPIHeader* header

    try:
        capsule = getattr(type(obj), "__dlpack_c_exchange_api__")
    except AttributeError:
        return NULL

    api = <DLPackExchangeAPI*>PyCapsule_GetPointer(capsule, DLPACK_EXCHANGE_API_CAPSULE_NAME)
    if api == NULL:
        raise BufferError("DLPack C exchange API capsule is invalid")

    header = &api.header
    while header != NULL:
        if header.version.major == <uint32_t>DLPACK_MAJOR_VERSION:
            return <DLPackExchangeAPI*>header
        header = header.prev_api
    raise BufferError("DLPack C exchange API has no compatible major version")


cdef object _dlpack_capsule_from_object(object obj, object stream):
    cdef object dlpack

    if (
        PyCapsule_IsValid(obj, DLPACK_VERSIONED_CAPSULE_NAME)
        or PyCapsule_IsValid(obj, DLPACK_CAPSULE_NAME)
    ):
        return obj

    try:
        dlpack = obj.__dlpack__
    except AttributeError:
        raise TypeError("object does not support DLPack") from None

    try:
        if stream is None:
            return dlpack(max_version=(DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION))
        return dlpack(stream=stream, max_version=(DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION))
    except TypeError:
        if stream is None:
            return dlpack()
        return dlpack(stream=stream)


cdef _DLPackManagedTensorOwner _consume_versioned_dlpack_capsule(object capsule):
    cdef DLManagedTensorVersioned* tensor
    cdef _DLPackManagedTensorOwner owner = _DLPackManagedTensorOwner()

    if not PyCapsule_IsValid(capsule, DLPACK_VERSIONED_CAPSULE_NAME):
        raise BufferError("object is not a valid versioned DLPack capsule")

    tensor = <DLManagedTensorVersioned*>PyCapsule_GetPointer(capsule, DLPACK_VERSIONED_CAPSULE_NAME)
    PyCapsule_SetName(capsule, USED_DLPACK_VERSIONED_CAPSULE_NAME)

    owner._producer = capsule
    owner._versioned = tensor
    owner._versioned_deleter = tensor.deleter
    if tensor.version.major != <uint32_t>DLPACK_MAJOR_VERSION:
        raise BufferError(f"unsupported DLPack major version {tensor.version.major}")
    return owner


cdef _DLPackManagedTensorOwner _consume_legacy_dlpack_capsule(object capsule):
    cdef DLManagedTensor* tensor
    cdef _DLPackManagedTensorOwner owner = _DLPackManagedTensorOwner()

    if not PyCapsule_IsValid(capsule, DLPACK_CAPSULE_NAME):
        raise BufferError("object is not a valid DLPack capsule")

    tensor = <DLManagedTensor*>PyCapsule_GetPointer(capsule, DLPACK_CAPSULE_NAME)
    PyCapsule_SetName(capsule, USED_DLPACK_CAPSULE_NAME)

    owner._producer = capsule
    owner._legacy = tensor
    owner._legacy_deleter = tensor.deleter
    return owner


cdef _DLPackManagedTensorOwner _consume_dlpack_capsule(object capsule):
    if PyCapsule_IsValid(capsule, DLPACK_VERSIONED_CAPSULE_NAME):
        return _consume_versioned_dlpack_capsule(capsule)
    if PyCapsule_IsValid(capsule, DLPACK_CAPSULE_NAME):
        return _consume_legacy_dlpack_capsule(capsule)
    raise BufferError("object is not a valid DLPack capsule")


cdef void _validate_dlpack_tensor(DLTensor* tensor) except *:
    cdef int32_t axis
    cdef int64_t extent
    cdef bint is_empty = False

    if tensor == NULL:
        raise BufferError("DLPack tensor pointer must not be null")
    if tensor.device.device_type != kDLCUDA and tensor.device.device_type != kDLCUDAManaged:
        raise BufferError(f"expected a CUDA DLPack tensor, got device type {tensor.device.device_type}")
    if tensor.ndim < 0:
        raise BufferError("DLPack tensor rank must be non-negative")
    if tensor.ndim != 0 and tensor.shape == NULL:
        raise BufferError("DLPack tensor shape must not be null for non-scalar tensors")
    if tensor.ndim == 0:
        if tensor.data == NULL:
            raise BufferError("non-empty DLPack tensor must have a data pointer")
        return

    for axis in range(tensor.ndim):
        extent = tensor.shape[axis]
        if extent < 0:
            raise BufferError("DLPack tensor shape entries must be non-negative")
        if extent == 0:
            is_empty = True

    if not is_empty and tensor.data == NULL:
        raise BufferError("non-empty DLPack tensor must have a data pointer")


cdef tuple _dlpack_shape_tuple(DLTensor* tensor):
    if tensor.ndim == 0:
        return ()
    return _int64_tuple(tensor.shape, <size_t>tensor.ndim)


cdef tuple _dlpack_compact_c_strides_tuple(DLTensor* tensor):
    cdef size_t rank = <size_t>tensor.ndim
    cdef int64_t* strides = <int64_t*>_alloc_array(rank, sizeof(int64_t))
    cdef uint64_t running = 1
    cdef size_t axis = rank
    cdef tuple result

    try:
        while axis > 0:
            axis -= 1
            if running > <uint64_t>0x7FFFFFFFFFFFFFFF:
                raise BufferError("DLPack tensor strides are too large")
            strides[axis] = <int64_t>running
            if tensor.shape[axis] != 0 and running > (<uint64_t>-1) // <uint64_t>tensor.shape[axis]:
                raise BufferError("DLPack tensor strides are too large")
            running *= <uint64_t>tensor.shape[axis]

        result = _int64_tuple(strides, rank)
    finally:
        PyMem_Free(strides)

    return result


cdef tuple _dlpack_strides_tuple(DLTensor* tensor):
    if tensor.ndim == 0:
        return ()
    if tensor.strides == NULL:
        return _dlpack_compact_c_strides_tuple(tensor)
    return _int64_tuple(tensor.strides, <size_t>tensor.ndim)


cdef _PreparedDeviceCopyView _prepare_view_from_dlpack_tensor(object owner, DLTensor* tensor):
    cdef tuple shape
    cdef tuple strides

    _validate_dlpack_tensor(tensor)
    shape = _dlpack_shape_tuple(tensor)
    strides = _dlpack_strides_tuple(tensor)
    return _PreparedDeviceCopyView(
        owner,
        <uintptr_t>tensor.data,
        <uint64_t>tensor.byte_offset,
        shape,
        strides,
    )


cdef _PreparedDeviceCopyView _prepare_view_from_dlpack_owner(_DLPackManagedTensorOwner owner):
    return _prepare_view_from_dlpack_tensor(owner, owner._tensor())


cdef _PreparedDeviceCopyView _prepare_dlpack_view_from_c_exchange(
    object obj,
    DLPackExchangeAPI* api,
):
    cdef DLTensor tensor
    cdef DLManagedTensorVersioned* managed_tensor
    cdef _DLPackManagedTensorOwner owner

    if api.dltensor_from_py_object_no_sync != NULL:
        if api.dltensor_from_py_object_no_sync(<void*><PyObject*>obj, &tensor) != 0:
            raise BufferError("DLPack C exchange dltensor_from_py_object_no_sync failed")
        return _prepare_view_from_dlpack_tensor(obj, &tensor)

    if api.managed_tensor_from_py_object_no_sync != NULL:
        managed_tensor = NULL
        if api.managed_tensor_from_py_object_no_sync(<void*><PyObject*>obj, &managed_tensor) != 0:
            raise BufferError("DLPack C exchange managed_tensor_from_py_object_no_sync failed")
        owner = _DLPackManagedTensorOwner()
        owner._producer = obj
        owner._versioned = managed_tensor
        owner._versioned_deleter = managed_tensor.deleter
        if managed_tensor.version.major != <uint32_t>DLPACK_MAJOR_VERSION:
            raise BufferError(f"unsupported DLPack major version {managed_tensor.version.major}")
        return _prepare_view_from_dlpack_owner(owner)

    raise BufferError("DLPack C exchange API does not provide tensor export")


def _prepare_dlpack_view(object obj, object stream=None):
    cdef DLPackExchangeAPI* api
    cdef object capsule
    cdef _DLPackManagedTensorOwner owner

    api = _dlpack_exchange_api(obj)
    if api != NULL:
        return _prepare_dlpack_view_from_c_exchange(obj, api)

    capsule = _dlpack_capsule_from_object(obj, stream)
    owner = _consume_dlpack_capsule(capsule)
    return _prepare_view_from_dlpack_owner(owner)


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
