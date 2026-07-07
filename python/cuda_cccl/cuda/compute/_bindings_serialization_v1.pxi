# Auto-generated serialization extern declarations and implementations for v1 backend.
# Included at the end of _bindings_impl.pyx by CMake-selected include.

cdef extern from "cccl/c/serialization.h":
    cdef void cccl_serialization_buffer_free(void*) nogil

cdef extern from "cccl/c/serialization_diagnostics.h":
    cdef const char* cccl_serialization_last_error() nogil
    cdef CUresult cccl_serialization_validate_blob(const void*, size_t) nogil


cdef _serialization_check_loadable(const void* buf, size_t size):
    # Validate the blob header magic and, for CUBIN
    # payloads, that the target compute-capability major matches this device,
    # BEFORE the opaque cuLibraryLoadData failure. On mismatch, raise with the
    # C layer's descriptive message instead of a bare CUDA error code.
    cdef CUresult st
    with nogil:
        st = cccl_serialization_validate_blob(buf, size)
    if st != 0:
        raise RuntimeError((<bytes> cccl_serialization_last_error()).decode("utf-8", "replace"))

cdef extern from "cccl/c/reduce.h":
    cdef CUresult cccl_device_reduce_load(
        cccl_device_reduce_build_result_t*
    ) nogil
    cdef CUresult cccl_device_reduce_serialize(
        const cccl_device_reduce_build_result_t*,
        void**,
        size_t*
    ) nogil
    cdef CUresult cccl_device_reduce_deserialize(
        cccl_device_reduce_build_result_t*,
        const void*,
        size_t
    ) nogil

cdef extern from "cccl/c/scan.h":
    cdef CUresult cccl_device_scan_load(
        cccl_device_scan_build_result_t*
    ) nogil
    cdef CUresult cccl_device_scan_serialize(
        const cccl_device_scan_build_result_t*,
        void**,
        size_t*
    ) nogil
    cdef CUresult cccl_device_scan_deserialize(
        cccl_device_scan_build_result_t*,
        const void*,
        size_t
    ) nogil

cdef extern from "cccl/c/segmented_reduce.h":
    cdef CUresult cccl_device_segmented_reduce_load(
        cccl_device_segmented_reduce_build_result_t*
    ) nogil
    cdef CUresult cccl_device_segmented_reduce_serialize(
        const cccl_device_segmented_reduce_build_result_t*,
        void**,
        size_t*
    ) nogil
    cdef CUresult cccl_device_segmented_reduce_deserialize(
        cccl_device_segmented_reduce_build_result_t*,
        const void*,
        size_t
    ) nogil

cdef extern from "cccl/c/merge_sort.h":
    cdef CUresult cccl_device_merge_sort_load(
        cccl_device_merge_sort_build_result_t*
    ) nogil
    cdef CUresult cccl_device_merge_sort_serialize(
        const cccl_device_merge_sort_build_result_t*,
        void**,
        size_t*
    ) nogil
    cdef CUresult cccl_device_merge_sort_deserialize(
        cccl_device_merge_sort_build_result_t*,
        const void*,
        size_t
    ) nogil

cdef extern from "cccl/c/unique_by_key.h":
    cdef CUresult cccl_device_unique_by_key_load(
        cccl_device_unique_by_key_build_result_t*
    ) nogil
    cdef CUresult cccl_device_unique_by_key_serialize(
        const cccl_device_unique_by_key_build_result_t*,
        void**,
        size_t*
    ) nogil
    cdef CUresult cccl_device_unique_by_key_deserialize(
        cccl_device_unique_by_key_build_result_t*,
        const void*,
        size_t
    ) nogil

cdef extern from "cccl/c/radix_sort.h":
    cdef CUresult cccl_device_radix_sort_load(
        cccl_device_radix_sort_build_result_t*
    ) nogil
    cdef CUresult cccl_device_radix_sort_serialize(
        const cccl_device_radix_sort_build_result_t*,
        void**,
        size_t*
    ) nogil
    cdef CUresult cccl_device_radix_sort_deserialize(
        cccl_device_radix_sort_build_result_t*,
        const void*,
        size_t
    ) nogil

cdef extern from "cccl/c/transform.h":
    cdef CUresult cccl_device_transform_load(
        cccl_device_transform_build_result_t*
    ) nogil
    cdef CUresult cccl_device_transform_serialize(
        const cccl_device_transform_build_result_t*,
        void**,
        size_t*
    ) nogil
    cdef CUresult cccl_device_transform_deserialize(
        cccl_device_transform_build_result_t*,
        const void*,
        size_t
    ) nogil

cdef extern from "cccl/c/histogram.h":
    cdef CUresult cccl_device_histogram_load(
        cccl_device_histogram_build_result_t*
    ) nogil
    cdef CUresult cccl_device_histogram_serialize(
        const cccl_device_histogram_build_result_t*,
        void**,
        size_t*
    ) nogil
    cdef CUresult cccl_device_histogram_deserialize(
        cccl_device_histogram_build_result_t*,
        const void*,
        size_t
    ) nogil

cdef extern from "cccl/c/binary_search.h":
    cdef CUresult cccl_device_binary_search_load(
        cccl_device_binary_search_build_result_t*
    ) nogil
    cdef CUresult cccl_device_binary_search_serialize(
        const cccl_device_binary_search_build_result_t*,
        void**,
        size_t*
    ) nogil
    cdef CUresult cccl_device_binary_search_deserialize(
        cccl_device_binary_search_build_result_t*,
        const void*,
        size_t
    ) nogil

cdef extern from "cccl/c/three_way_partition.h":
    cdef CUresult cccl_device_three_way_partition_load(
        cccl_device_three_way_partition_build_result_t*
    ) nogil
    cdef CUresult cccl_device_three_way_partition_serialize(
        const cccl_device_three_way_partition_build_result_t*,
        void**,
        size_t*
    ) nogil
    cdef CUresult cccl_device_three_way_partition_deserialize(
        cccl_device_three_way_partition_build_result_t*,
        const void*,
        size_t
    ) nogil

cdef extern from "cccl/c/segmented_sort.h":
    cdef CUresult cccl_device_segmented_sort_load(
        cccl_device_segmented_sort_build_result_t*
    ) nogil
    cdef CUresult cccl_device_segmented_sort_serialize(
        const cccl_device_segmented_sort_build_result_t*,
        void**,
        size_t*
    ) nogil
    cdef CUresult cccl_device_segmented_sort_deserialize(
        cccl_device_segmented_sort_build_result_t*,
        const void*,
        size_t
    ) nogil

def _serialization_reduce_serialize(DeviceReduceBuildResult self):
    cdef CUresult status = -1
    cdef void* buf_ptr = NULL
    cdef size_t buf_size = 0
    with nogil:
        status = cccl_device_reduce_serialize(&self.build_data, &buf_ptr, &buf_size)
    if status != 0:
        if buf_ptr != NULL:
            cccl_serialization_buffer_free(buf_ptr)
        raise RuntimeError(
            f"Failed serializing reduce, error code: {status}"
        )
    try:
        return PyBytes_FromStringAndSize(<const char*>buf_ptr, buf_size)
    finally:
        cccl_serialization_buffer_free(buf_ptr)

def _serialization_reduce_deserialize(blob):
    # Copy to immutable bytes: the C deserialize reads the buffer under
    # `nogil`, so a writable bytearray/memoryview could be mutated mid-read.
    blob = bytes(blob)
    cdef DeviceReduceBuildResult self = DeviceReduceBuildResult.__new__(
        DeviceReduceBuildResult)
    cdef CUresult status = -1
    cdef const unsigned char[::1] view = blob
    cdef size_t buf_size = view.shape[0]
    if buf_size == 0:
        raise RuntimeError("Cannot deserialize empty blob")
    cdef const void* buf_ptr = <const void*>&view[0]
    _serialization_check_loadable(buf_ptr, buf_size)
    with nogil:
        status = cccl_device_reduce_deserialize(
            &self.build_data, buf_ptr, buf_size)
    if status != 0:
        raise RuntimeError(
            f"Failed deserializing reduce, error code: {status}"
        )
    with nogil:
        status = cccl_device_reduce_load(&self.build_data)
    if status != 0:
        raise RuntimeError(
            f"Failed loading reduce after deserialize, error code: {status}"
        )
    return self

def _serialization_scan_serialize(DeviceScanBuildResult self):
    cdef CUresult status = -1
    cdef void* buf_ptr = NULL
    cdef size_t buf_size = 0
    with nogil:
        status = cccl_device_scan_serialize(&self.build_data, &buf_ptr, &buf_size)
    if status != 0:
        if buf_ptr != NULL:
            cccl_serialization_buffer_free(buf_ptr)
        raise RuntimeError(f"Failed serializing scan, error code: {status}")
    try:
        return PyBytes_FromStringAndSize(<const char*>buf_ptr, buf_size)
    finally:
        cccl_serialization_buffer_free(buf_ptr)

def _serialization_scan_deserialize(blob):
    # Copy to immutable bytes: the C deserialize reads the buffer under
    # `nogil`, so a writable bytearray/memoryview could be mutated mid-read.
    blob = bytes(blob)
    cdef DeviceScanBuildResult self = DeviceScanBuildResult.__new__(
        DeviceScanBuildResult)
    cdef CUresult status = -1
    cdef const unsigned char[::1] view = blob
    cdef size_t buf_size = view.shape[0]
    if buf_size == 0:
        raise RuntimeError("Cannot deserialize empty blob")
    cdef const void* buf_ptr = <const void*>&view[0]
    _serialization_check_loadable(buf_ptr, buf_size)
    with nogil:
        status = cccl_device_scan_deserialize(
            &self.build_data, buf_ptr, buf_size)
    if status != 0:
        raise RuntimeError(f"Failed deserializing scan, error code: {status}")
    with nogil:
        status = cccl_device_scan_load(&self.build_data)
    if status != 0:
        raise RuntimeError(f"Failed loading scan after deserialize, error code: {status}")
    return self

def _serialization_segmented_reduce_serialize(DeviceSegmentedReduceBuildResult self):
    cdef CUresult status = -1
    cdef void* buf_ptr = NULL
    cdef size_t buf_size = 0
    with nogil:
        status = cccl_device_segmented_reduce_serialize(&self.build_data, &buf_ptr, &buf_size)
    if status != 0:
        if buf_ptr != NULL:
            cccl_serialization_buffer_free(buf_ptr)
        raise RuntimeError(f"Failed serializing segmented_reduce, error code: {status}")
    try:
        return PyBytes_FromStringAndSize(<const char*>buf_ptr, buf_size)
    finally:
        cccl_serialization_buffer_free(buf_ptr)

def _serialization_segmented_reduce_deserialize(blob):
    # Copy to immutable bytes: the C deserialize reads the buffer under
    # `nogil`, so a writable bytearray/memoryview could be mutated mid-read.
    blob = bytes(blob)
    cdef DeviceSegmentedReduceBuildResult self = DeviceSegmentedReduceBuildResult.__new__(
        DeviceSegmentedReduceBuildResult)
    cdef CUresult status = -1
    cdef const unsigned char[::1] view = blob
    cdef size_t buf_size = view.shape[0]
    if buf_size == 0:
        raise RuntimeError("Cannot deserialize empty blob")
    cdef const void* buf_ptr = <const void*>&view[0]
    _serialization_check_loadable(buf_ptr, buf_size)
    with nogil:
        status = cccl_device_segmented_reduce_deserialize(
            &self.build_data, buf_ptr, buf_size)
    if status != 0:
        raise RuntimeError(f"Failed deserializing segmented_reduce, error code: {status}")
    with nogil:
        status = cccl_device_segmented_reduce_load(&self.build_data)
    if status != 0:
        raise RuntimeError(f"Failed loading segmented_reduce after deserialize, error code: {status}")
    return self

def _serialization_merge_sort_serialize(DeviceMergeSortBuildResult self):
    cdef CUresult status = -1
    cdef void* buf_ptr = NULL
    cdef size_t buf_size = 0
    with nogil:
        status = cccl_device_merge_sort_serialize(&self.build_data, &buf_ptr, &buf_size)
    if status != 0:
        if buf_ptr != NULL:
            cccl_serialization_buffer_free(buf_ptr)
        raise RuntimeError(f"Failed serializing merge_sort, error code: {status}")
    try:
        return PyBytes_FromStringAndSize(<const char*>buf_ptr, buf_size)
    finally:
        cccl_serialization_buffer_free(buf_ptr)

def _serialization_merge_sort_deserialize(blob):
    # Copy to immutable bytes: the C deserialize reads the buffer under
    # `nogil`, so a writable bytearray/memoryview could be mutated mid-read.
    blob = bytes(blob)
    cdef DeviceMergeSortBuildResult self = DeviceMergeSortBuildResult.__new__(
        DeviceMergeSortBuildResult)
    cdef CUresult status = -1
    cdef const unsigned char[::1] view = blob
    cdef size_t buf_size = view.shape[0]
    if buf_size == 0:
        raise RuntimeError("Cannot deserialize empty blob")
    cdef const void* buf_ptr = <const void*>&view[0]
    _serialization_check_loadable(buf_ptr, buf_size)
    with nogil:
        status = cccl_device_merge_sort_deserialize(
            &self.build_data, buf_ptr, buf_size)
    if status != 0:
        raise RuntimeError(f"Failed deserializing merge_sort, error code: {status}")
    with nogil:
        status = cccl_device_merge_sort_load(&self.build_data)
    if status != 0:
        raise RuntimeError(f"Failed loading merge_sort after deserialize, error code: {status}")
    return self

def _serialization_unique_by_key_serialize(DeviceUniqueByKeyBuildResult self):
    cdef CUresult status = -1
    cdef void* buf_ptr = NULL
    cdef size_t buf_size = 0
    with nogil:
        status = cccl_device_unique_by_key_serialize(&self.build_data, &buf_ptr, &buf_size)
    if status != 0:
        if buf_ptr != NULL:
            cccl_serialization_buffer_free(buf_ptr)
        raise RuntimeError(f"Failed serializing unique_by_key, error code: {status}")
    try:
        return PyBytes_FromStringAndSize(<const char*>buf_ptr, buf_size)
    finally:
        cccl_serialization_buffer_free(buf_ptr)

def _serialization_unique_by_key_deserialize(blob):
    # Copy to immutable bytes: the C deserialize reads the buffer under
    # `nogil`, so a writable bytearray/memoryview could be mutated mid-read.
    blob = bytes(blob)
    cdef DeviceUniqueByKeyBuildResult self = DeviceUniqueByKeyBuildResult.__new__(
        DeviceUniqueByKeyBuildResult)
    cdef CUresult status = -1
    cdef const unsigned char[::1] view = blob
    cdef size_t buf_size = view.shape[0]
    if buf_size == 0:
        raise RuntimeError("Cannot deserialize empty blob")
    cdef const void* buf_ptr = <const void*>&view[0]
    _serialization_check_loadable(buf_ptr, buf_size)
    with nogil:
        status = cccl_device_unique_by_key_deserialize(
            &self.build_data, buf_ptr, buf_size)
    if status != 0:
        raise RuntimeError(f"Failed deserializing unique_by_key, error code: {status}")
    with nogil:
        status = cccl_device_unique_by_key_load(&self.build_data)
    if status != 0:
        raise RuntimeError(f"Failed loading unique_by_key after deserialize, error code: {status}")
    return self

def _serialization_radix_sort_serialize(DeviceRadixSortBuildResult self):
    cdef CUresult status = -1
    cdef void* buf_ptr = NULL
    cdef size_t buf_size = 0
    with nogil:
        status = cccl_device_radix_sort_serialize(&self.build_data, &buf_ptr, &buf_size)
    if status != 0:
        if buf_ptr != NULL:
            cccl_serialization_buffer_free(buf_ptr)
        raise RuntimeError(f"Failed serializing radix_sort, error code: {status}")
    try:
        return PyBytes_FromStringAndSize(<const char*>buf_ptr, buf_size)
    finally:
        cccl_serialization_buffer_free(buf_ptr)

def _serialization_radix_sort_deserialize(blob):
    # Copy to immutable bytes: the C deserialize reads the buffer under
    # `nogil`, so a writable bytearray/memoryview could be mutated mid-read.
    blob = bytes(blob)
    cdef DeviceRadixSortBuildResult self = DeviceRadixSortBuildResult.__new__(
        DeviceRadixSortBuildResult)
    cdef CUresult status = -1
    cdef const unsigned char[::1] view = blob
    cdef size_t buf_size = view.shape[0]
    if buf_size == 0:
        raise RuntimeError("Cannot deserialize empty blob")
    cdef const void* buf_ptr = <const void*>&view[0]
    _serialization_check_loadable(buf_ptr, buf_size)
    with nogil:
        status = cccl_device_radix_sort_deserialize(
            &self.build_data, buf_ptr, buf_size)
    if status != 0:
        raise RuntimeError(f"Failed deserializing radix_sort, error code: {status}")
    with nogil:
        status = cccl_device_radix_sort_load(&self.build_data)
    if status != 0:
        raise RuntimeError(f"Failed loading radix_sort after deserialize, error code: {status}")
    return self

def _serialization_unary_transform_serialize(DeviceUnaryTransform self):
    cdef CUresult status = -1
    cdef void* buf_ptr = NULL
    cdef size_t buf_size = 0
    with nogil:
        status = cccl_device_transform_serialize(&self.build_data, &buf_ptr, &buf_size)
    if status != 0:
        if buf_ptr != NULL:
            cccl_serialization_buffer_free(buf_ptr)
        raise RuntimeError(f"Failed serializing unary_transform, error code: {status}")
    try:
        return PyBytes_FromStringAndSize(<const char*>buf_ptr, buf_size)
    finally:
        cccl_serialization_buffer_free(buf_ptr)

def _serialization_unary_transform_deserialize(blob):
    # Copy to immutable bytes: the C deserialize reads the buffer under
    # `nogil`, so a writable bytearray/memoryview could be mutated mid-read.
    blob = bytes(blob)
    cdef DeviceUnaryTransform self = DeviceUnaryTransform.__new__(DeviceUnaryTransform)
    cdef CUresult status = -1
    cdef const unsigned char[::1] view = blob
    cdef size_t buf_size = view.shape[0]
    if buf_size == 0:
        raise RuntimeError("Cannot deserialize empty blob")
    cdef const void* buf_ptr = <const void*>&view[0]
    _serialization_check_loadable(buf_ptr, buf_size)
    with nogil:
        status = cccl_device_transform_deserialize(
            &self.build_data, buf_ptr, buf_size)
    if status != 0:
        raise RuntimeError(f"Failed deserializing unary_transform, error code: {status}")
    with nogil:
        status = cccl_device_transform_load(&self.build_data)
    if status != 0:
        raise RuntimeError(f"Failed loading unary_transform after deserialize, error code: {status}")
    return self

def _serialization_binary_transform_serialize(DeviceBinaryTransform self):
    cdef CUresult status = -1
    cdef void* buf_ptr = NULL
    cdef size_t buf_size = 0
    with nogil:
        status = cccl_device_transform_serialize(&self.build_data, &buf_ptr, &buf_size)
    if status != 0:
        if buf_ptr != NULL:
            cccl_serialization_buffer_free(buf_ptr)
        raise RuntimeError(f"Failed serializing binary_transform, error code: {status}")
    try:
        return PyBytes_FromStringAndSize(<const char*>buf_ptr, buf_size)
    finally:
        cccl_serialization_buffer_free(buf_ptr)

def _serialization_binary_transform_deserialize(blob):
    # Copy to immutable bytes: the C deserialize reads the buffer under
    # `nogil`, so a writable bytearray/memoryview could be mutated mid-read.
    blob = bytes(blob)
    cdef DeviceBinaryTransform self = DeviceBinaryTransform.__new__(DeviceBinaryTransform)
    cdef CUresult status = -1
    cdef const unsigned char[::1] view = blob
    cdef size_t buf_size = view.shape[0]
    if buf_size == 0:
        raise RuntimeError("Cannot deserialize empty blob")
    cdef const void* buf_ptr = <const void*>&view[0]
    _serialization_check_loadable(buf_ptr, buf_size)
    with nogil:
        status = cccl_device_transform_deserialize(
            &self.build_data, buf_ptr, buf_size)
    if status != 0:
        raise RuntimeError(f"Failed deserializing binary_transform, error code: {status}")
    with nogil:
        status = cccl_device_transform_load(&self.build_data)
    if status != 0:
        raise RuntimeError(f"Failed loading binary_transform after deserialize, error code: {status}")
    return self

def _serialization_histogram_serialize(DeviceHistogramBuildResult self):
    cdef CUresult status = -1
    cdef void* buf_ptr = NULL
    cdef size_t buf_size = 0
    with nogil:
        status = cccl_device_histogram_serialize(&self.build_data, &buf_ptr, &buf_size)
    if status != 0:
        if buf_ptr != NULL:
            cccl_serialization_buffer_free(buf_ptr)
        raise RuntimeError(f"Failed serializing histogram, error code: {status}")
    try:
        return PyBytes_FromStringAndSize(<const char*>buf_ptr, buf_size)
    finally:
        cccl_serialization_buffer_free(buf_ptr)

def _serialization_histogram_deserialize(blob):
    # Copy to immutable bytes: the C deserialize reads the buffer under
    # `nogil`, so a writable bytearray/memoryview could be mutated mid-read.
    blob = bytes(blob)
    cdef DeviceHistogramBuildResult self = DeviceHistogramBuildResult.__new__(
        DeviceHistogramBuildResult)
    cdef CUresult status = -1
    cdef const unsigned char[::1] view = blob
    cdef size_t buf_size = view.shape[0]
    if buf_size == 0:
        raise RuntimeError("Cannot deserialize empty blob")
    cdef const void* buf_ptr = <const void*>&view[0]
    _serialization_check_loadable(buf_ptr, buf_size)
    with nogil:
        status = cccl_device_histogram_deserialize(
            &self.build_data, buf_ptr, buf_size)
    if status != 0:
        raise RuntimeError(f"Failed deserializing histogram, error code: {status}")
    with nogil:
        status = cccl_device_histogram_load(&self.build_data)
    if status != 0:
        raise RuntimeError(f"Failed loading histogram after deserialize, error code: {status}")
    return self

def _serialization_binary_search_serialize(DeviceBinarySearchBuildResult self):
    cdef CUresult status = -1
    cdef void* buf_ptr = NULL
    cdef size_t buf_size = 0
    with nogil:
        status = cccl_device_binary_search_serialize(&self.build_data, &buf_ptr, &buf_size)
    if status != 0:
        if buf_ptr != NULL:
            cccl_serialization_buffer_free(buf_ptr)
        raise RuntimeError(f"Failed serializing binary_search, error code: {status}")
    try:
        return PyBytes_FromStringAndSize(<const char*>buf_ptr, buf_size)
    finally:
        cccl_serialization_buffer_free(buf_ptr)

def _serialization_binary_search_deserialize(blob):
    # Copy to immutable bytes: the C deserialize reads the buffer under
    # `nogil`, so a writable bytearray/memoryview could be mutated mid-read.
    blob = bytes(blob)
    cdef DeviceBinarySearchBuildResult self = DeviceBinarySearchBuildResult.__new__(
        DeviceBinarySearchBuildResult)
    cdef CUresult status = -1
    cdef const unsigned char[::1] view = blob
    cdef size_t buf_size = view.shape[0]
    if buf_size == 0:
        raise RuntimeError("Cannot deserialize empty blob")
    cdef const void* buf_ptr = <const void*>&view[0]
    _serialization_check_loadable(buf_ptr, buf_size)
    with nogil:
        status = cccl_device_binary_search_deserialize(
            &self.build_data, buf_ptr, buf_size)
    if status != 0:
        raise RuntimeError(f"Failed deserializing binary_search, error code: {status}")
    with nogil:
        status = cccl_device_binary_search_load(&self.build_data)
    if status != 0:
        raise RuntimeError(f"Failed loading binary_search after deserialize, error code: {status}")
    return self

def _serialization_three_way_partition_serialize(DeviceThreeWayPartitionBuildResult self):
    cdef CUresult status = -1
    cdef void* buf_ptr = NULL
    cdef size_t buf_size = 0
    with nogil:
        status = cccl_device_three_way_partition_serialize(&self.build_data, &buf_ptr, &buf_size)
    if status != 0:
        if buf_ptr != NULL:
            cccl_serialization_buffer_free(buf_ptr)
        raise RuntimeError(f"Failed serializing three_way_partition, error code: {status}")
    try:
        return PyBytes_FromStringAndSize(<const char*>buf_ptr, buf_size)
    finally:
        cccl_serialization_buffer_free(buf_ptr)

def _serialization_three_way_partition_deserialize(blob):
    # Copy to immutable bytes: the C deserialize reads the buffer under
    # `nogil`, so a writable bytearray/memoryview could be mutated mid-read.
    blob = bytes(blob)
    cdef DeviceThreeWayPartitionBuildResult self = DeviceThreeWayPartitionBuildResult.__new__(
        DeviceThreeWayPartitionBuildResult)
    cdef CUresult status = -1
    cdef const unsigned char[::1] view = blob
    cdef size_t buf_size = view.shape[0]
    if buf_size == 0:
        raise RuntimeError("Cannot deserialize empty blob")
    cdef const void* buf_ptr = <const void*>&view[0]
    _serialization_check_loadable(buf_ptr, buf_size)
    with nogil:
        status = cccl_device_three_way_partition_deserialize(
            &self.build_data, buf_ptr, buf_size)
    if status != 0:
        raise RuntimeError(f"Failed deserializing three_way_partition, error code: {status}")
    with nogil:
        status = cccl_device_three_way_partition_load(&self.build_data)
    if status != 0:
        raise RuntimeError(f"Failed loading three_way_partition after deserialize, error code: {status}")
    return self

def _serialization_segmented_sort_serialize(DeviceSegmentedSortBuildResult self):
    cdef CUresult status = -1
    cdef void* buf_ptr = NULL
    cdef size_t buf_size = 0
    with nogil:
        status = cccl_device_segmented_sort_serialize(&self.build_data, &buf_ptr, &buf_size)
    if status != 0:
        if buf_ptr != NULL:
            cccl_serialization_buffer_free(buf_ptr)
        raise RuntimeError(f"Failed serializing segmented_sort, error code: {status}")
    try:
        return PyBytes_FromStringAndSize(<const char*>buf_ptr, buf_size)
    finally:
        cccl_serialization_buffer_free(buf_ptr)

def _serialization_segmented_sort_deserialize(blob):
    # Copy to immutable bytes: the C deserialize reads the buffer under
    # `nogil`, so a writable bytearray/memoryview could be mutated mid-read.
    blob = bytes(blob)
    cdef DeviceSegmentedSortBuildResult self = DeviceSegmentedSortBuildResult.__new__(
        DeviceSegmentedSortBuildResult)
    cdef CUresult status = -1
    cdef const unsigned char[::1] view = blob
    cdef size_t buf_size = view.shape[0]
    if buf_size == 0:
        raise RuntimeError("Cannot deserialize empty blob")
    cdef const void* buf_ptr = <const void*>&view[0]
    _serialization_check_loadable(buf_ptr, buf_size)
    with nogil:
        status = cccl_device_segmented_sort_deserialize(
            &self.build_data, buf_ptr, buf_size)
    if status != 0:
        raise RuntimeError(f"Failed deserializing segmented_sort, error code: {status}")
    with nogil:
        status = cccl_device_segmented_sort_load(&self.build_data)
    if status != 0:
        raise RuntimeError(f"Failed loading segmented_sort after deserialize, error code: {status}")
    return self
