# v2 (HostJIT) backend — serialize/deserialize/compile/load.
# Included at the end of _bindings_impl.pyx.
#
# v2's blob format/portability story differs from v1's:
# the payload is a full compiled shared library (.so/.dll), not a driver-loadable
# device blob, and cross-machine loading needs to locate *this* machine's CUDA
# Toolkit (to pre-resolve libcudart before dlopen) — hence cccl_device_<algo>_load
# takes an extra ctk_path argument v1's equivalent doesn't have. We always pass
# NULL (auto-detect) from here: auto-detection finds the CTK on whichever machine
# is actually calling _load(), which is exactly correct regardless of which
# machine originally compiled/serialized the blob.

cdef extern from "cccl/c/serialization.h":
    cdef void cccl_serialization_v2_buffer_free(void*) nogil

cdef extern from "cccl/c/reduce.h":
    cdef CUresult cccl_device_reduce_compile(
        cccl_device_reduce_build_result_t*,
        cccl_iterator_t,
        cccl_iterator_t,
        cccl_op_t,
        cccl_value_t,
        cccl_determinism_t,
        int, int, const char*, const char*, const char*, const char*,
        void*
    ) nogil
    cdef CUresult cccl_device_reduce_load(
        cccl_device_reduce_build_result_t*,
        const char*
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

def _reduce_serialize(DeviceReduceBuildResult self):
    cdef CUresult status = -1
    cdef void* buf_ptr = NULL
    cdef size_t buf_size = 0
    with nogil:
        status = cccl_device_reduce_serialize(&self.build_data, &buf_ptr, &buf_size)
    if status != 0:
        if buf_ptr != NULL:
            cccl_serialization_v2_buffer_free(buf_ptr)
        raise RuntimeError(f"Failed serializing reduce, error code: {status}")
    try:
        return PyBytes_FromStringAndSize(<const char*>buf_ptr, buf_size)
    finally:
        cccl_serialization_v2_buffer_free(buf_ptr)

def _reduce_deserialize(blob, load=True, check_cc=True):
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
    # NOTE: unlike v1, v2 has no cccl_serialization_validate_blob equivalent
    # yet (no separate pre-flight header check) -- deserialize() itself
    # validates magic/algo/os-arch and raises a descriptive RuntimeError.
    # check_cc is accepted for API-shape parity with v1 but is a no-op here:
    # v2 deserialize doesn't check cc against a live device; a cc mismatch
    # surfaces later, at call time.
    with nogil:
        status = cccl_device_reduce_deserialize(
            &self.build_data, buf_ptr, buf_size)
    if status != 0:
        raise RuntimeError(f"Failed deserializing reduce, error code: {status}")
    if load:
        with nogil:
            status = cccl_device_reduce_load(&self.build_data, NULL)
        if status != 0:
            raise RuntimeError(f"Failed loading reduce after deserialize, error code: {status}")
        self._loaded = True
    return self

def _reduce_compile(
    Iterator d_in,
    Iterator d_out,
    Op op,
    Value h_init,
    cccl_determinism_t determinism,
    CommonData common_data,
):
    cdef DeviceReduceBuildResult self = DeviceReduceBuildResult.__new__(DeviceReduceBuildResult)
    cdef CUresult status = -1
    cdef int cc_major = common_data.get_cc_major()
    cdef int cc_minor = common_data.get_cc_minor()
    cdef const char *cub_path = common_data.cub_path_get_c_str()
    cdef const char *thrust_path = common_data.thrust_path_get_c_str()
    cdef const char *libcudacxx_path = common_data.libcudacxx_path_get_c_str()
    cdef const char *ctk_path = common_data.ctk_path_get_c_str()
    with nogil:
        status = cccl_device_reduce_compile(
            &self.build_data,
            d_in.iter_data,
            d_out.iter_data,
            op.op_data,
            h_init.value_data,
            determinism,
            cc_major,
            cc_minor,
            cub_path,
            thrust_path,
            libcudacxx_path,
            ctk_path,
            NULL,
        )
    if status != 0:
        raise RuntimeError(f"Failed compiling reduce, error code: {status}")
    return self

def _reduce_load(DeviceReduceBuildResult self):
    cdef CUresult status = -1
    with nogil:
        status = cccl_device_reduce_load(&self.build_data, NULL)
    if status != 0:
        raise RuntimeError(f"Failed loading reduce, error code: {status}")

cdef extern from "cccl/c/scan.h":
    cdef CUresult cccl_device_scan_compile(
        cccl_device_scan_build_result_t*,
        cccl_iterator_t,
        cccl_iterator_t,
        cccl_op_t,
        cccl_type_info,
        bint,
        cccl_init_kind_t,
        int, int, const char*, const char*, const char*, const char*,
        void*
    ) nogil
    cdef CUresult cccl_device_scan_load(
        cccl_device_scan_build_result_t*,
        const char*
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

def _scan_serialize(DeviceScanBuildResult self):
    cdef CUresult status = -1
    cdef void* buf_ptr = NULL
    cdef size_t buf_size = 0
    with nogil:
        status = cccl_device_scan_serialize(&self.build_data, &buf_ptr, &buf_size)
    if status != 0:
        if buf_ptr != NULL:
            cccl_serialization_v2_buffer_free(buf_ptr)
        raise RuntimeError(f"Failed serializing scan, error code: {status}")
    try:
        return PyBytes_FromStringAndSize(<const char*>buf_ptr, buf_size)
    finally:
        cccl_serialization_v2_buffer_free(buf_ptr)

def _scan_deserialize(blob, load=True, check_cc=True):
    blob = bytes(blob)
    cdef DeviceScanBuildResult self = DeviceScanBuildResult.__new__(
        DeviceScanBuildResult)
    cdef CUresult status = -1
    cdef const unsigned char[::1] view = blob
    cdef size_t buf_size = view.shape[0]
    if buf_size == 0:
        raise RuntimeError("Cannot deserialize empty blob")
    cdef const void* buf_ptr = <const void*>&view[0]
    with nogil:
        status = cccl_device_scan_deserialize(
            &self.build_data, buf_ptr, buf_size)
    if status != 0:
        raise RuntimeError(f"Failed deserializing scan, error code: {status}")
    if load:
        with nogil:
            status = cccl_device_scan_load(&self.build_data, NULL)
        if status != 0:
            raise RuntimeError(f"Failed loading scan after deserialize, error code: {status}")
        self._loaded = True
    return self

def _scan_compile(
    Iterator d_in,
    Iterator d_out,
    Op op,
    TypeInfo init_type,
    bint force_inclusive,
    cccl_init_kind_t init_kind,
    CommonData common_data,
):
    cdef DeviceScanBuildResult self = DeviceScanBuildResult.__new__(DeviceScanBuildResult)
    cdef CUresult status = -1
    cdef int cc_major = common_data.get_cc_major()
    cdef int cc_minor = common_data.get_cc_minor()
    cdef const char *cub_path = common_data.cub_path_get_c_str()
    cdef const char *thrust_path = common_data.thrust_path_get_c_str()
    cdef const char *libcudacxx_path = common_data.libcudacxx_path_get_c_str()
    cdef const char *ctk_path = common_data.ctk_path_get_c_str()
    with nogil:
        status = cccl_device_scan_compile(
            &self.build_data,
            d_in.iter_data,
            d_out.iter_data,
            op.op_data,
            init_type.type_info,
            force_inclusive,
            init_kind,
            cc_major,
            cc_minor,
            cub_path,
            thrust_path,
            libcudacxx_path,
            ctk_path,
            NULL,
        )
    if status != 0:
        raise RuntimeError(f"Failed compiling scan, error code: {status}")
    return self

def _scan_load(DeviceScanBuildResult self):
    cdef CUresult status = -1
    with nogil:
        status = cccl_device_scan_load(&self.build_data, NULL)
    if status != 0:
        raise RuntimeError(f"Failed loading scan, error code: {status}")

cdef extern from "cccl/c/segmented_reduce.h":
    cdef CUresult cccl_device_segmented_reduce_compile(
        cccl_device_segmented_reduce_build_result_t*,
        cccl_iterator_t,
        cccl_iterator_t,
        cccl_iterator_t,
        cccl_iterator_t,
        cccl_op_t,
        cccl_value_t,
        int, int, const char*, const char*, const char*, const char*,
        void*
    ) nogil
    cdef CUresult cccl_device_segmented_reduce_load(
        cccl_device_segmented_reduce_build_result_t*,
        const char*
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

def _segmented_reduce_serialize(DeviceSegmentedReduceBuildResult self):
    cdef CUresult status = -1
    cdef void* buf_ptr = NULL
    cdef size_t buf_size = 0
    with nogil:
        status = cccl_device_segmented_reduce_serialize(&self.build_data, &buf_ptr, &buf_size)
    if status != 0:
        if buf_ptr != NULL:
            cccl_serialization_v2_buffer_free(buf_ptr)
        raise RuntimeError(f"Failed serializing segmented_reduce, error code: {status}")
    try:
        return PyBytes_FromStringAndSize(<const char*>buf_ptr, buf_size)
    finally:
        cccl_serialization_v2_buffer_free(buf_ptr)

def _segmented_reduce_deserialize(blob, load=True, check_cc=True):
    blob = bytes(blob)
    cdef DeviceSegmentedReduceBuildResult self = DeviceSegmentedReduceBuildResult.__new__(
        DeviceSegmentedReduceBuildResult)
    cdef CUresult status = -1
    cdef const unsigned char[::1] view = blob
    cdef size_t buf_size = view.shape[0]
    if buf_size == 0:
        raise RuntimeError("Cannot deserialize empty blob")
    cdef const void* buf_ptr = <const void*>&view[0]
    with nogil:
        status = cccl_device_segmented_reduce_deserialize(
            &self.build_data, buf_ptr, buf_size)
    if status != 0:
        raise RuntimeError(f"Failed deserializing segmented_reduce, error code: {status}")
    if load:
        with nogil:
            status = cccl_device_segmented_reduce_load(&self.build_data, NULL)
        if status != 0:
            raise RuntimeError(f"Failed loading segmented_reduce after deserialize, error code: {status}")
        self._loaded = True
    return self

def _segmented_reduce_compile(
    Iterator d_in,
    Iterator d_out,
    Iterator start_offsets,
    Iterator end_offsets,
    Op op,
    Value h_init,
    CommonData common_data,
):
    cdef DeviceSegmentedReduceBuildResult self = DeviceSegmentedReduceBuildResult.__new__(DeviceSegmentedReduceBuildResult)
    cdef CUresult status = -1
    cdef int cc_major = common_data.get_cc_major()
    cdef int cc_minor = common_data.get_cc_minor()
    cdef const char *cub_path = common_data.cub_path_get_c_str()
    cdef const char *thrust_path = common_data.thrust_path_get_c_str()
    cdef const char *libcudacxx_path = common_data.libcudacxx_path_get_c_str()
    cdef const char *ctk_path = common_data.ctk_path_get_c_str()
    with nogil:
        status = cccl_device_segmented_reduce_compile(
            &self.build_data,
            d_in.iter_data,
            d_out.iter_data,
            start_offsets.iter_data,
            end_offsets.iter_data,
            op.op_data,
            h_init.value_data,
            cc_major,
            cc_minor,
            cub_path,
            thrust_path,
            libcudacxx_path,
            ctk_path,
            NULL,
        )
    if status != 0:
        raise RuntimeError(f"Failed compiling segmented_reduce, error code: {status}")
    return self

def _segmented_reduce_load(DeviceSegmentedReduceBuildResult self):
    cdef CUresult status = -1
    with nogil:
        status = cccl_device_segmented_reduce_load(&self.build_data, NULL)
    if status != 0:
        raise RuntimeError(f"Failed loading segmented_reduce, error code: {status}")

cdef extern from "cccl/c/unique_by_key.h":
    cdef CUresult cccl_device_unique_by_key_compile(
        cccl_device_unique_by_key_build_result_t*,
        cccl_iterator_t,
        cccl_iterator_t,
        cccl_iterator_t,
        cccl_iterator_t,
        cccl_iterator_t,
        cccl_op_t,
        int, int, const char*, const char*, const char*, const char*,
        void*
    ) nogil
    cdef CUresult cccl_device_unique_by_key_load(
        cccl_device_unique_by_key_build_result_t*,
        const char*
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

def _unique_by_key_serialize(DeviceUniqueByKeyBuildResult self):
    cdef CUresult status = -1
    cdef void* buf_ptr = NULL
    cdef size_t buf_size = 0
    with nogil:
        status = cccl_device_unique_by_key_serialize(&self.build_data, &buf_ptr, &buf_size)
    if status != 0:
        if buf_ptr != NULL:
            cccl_serialization_v2_buffer_free(buf_ptr)
        raise RuntimeError(f"Failed serializing unique_by_key, error code: {status}")
    try:
        return PyBytes_FromStringAndSize(<const char*>buf_ptr, buf_size)
    finally:
        cccl_serialization_v2_buffer_free(buf_ptr)

def _unique_by_key_deserialize(blob, load=True, check_cc=True):
    blob = bytes(blob)
    cdef DeviceUniqueByKeyBuildResult self = DeviceUniqueByKeyBuildResult.__new__(
        DeviceUniqueByKeyBuildResult)
    cdef CUresult status = -1
    cdef const unsigned char[::1] view = blob
    cdef size_t buf_size = view.shape[0]
    if buf_size == 0:
        raise RuntimeError("Cannot deserialize empty blob")
    cdef const void* buf_ptr = <const void*>&view[0]
    with nogil:
        status = cccl_device_unique_by_key_deserialize(
            &self.build_data, buf_ptr, buf_size)
    if status != 0:
        raise RuntimeError(f"Failed deserializing unique_by_key, error code: {status}")
    if load:
        with nogil:
            status = cccl_device_unique_by_key_load(&self.build_data, NULL)
        if status != 0:
            raise RuntimeError(f"Failed loading unique_by_key after deserialize, error code: {status}")
        self._loaded = True
    return self

def _unique_by_key_compile(
    Iterator d_keys_in,
    Iterator d_values_in,
    Iterator d_keys_out,
    Iterator d_values_out,
    Iterator d_num_selected_out,
    Op comparison_op,
    CommonData common_data,
):
    cdef DeviceUniqueByKeyBuildResult self = DeviceUniqueByKeyBuildResult.__new__(DeviceUniqueByKeyBuildResult)
    cdef CUresult status = -1
    cdef int cc_major = common_data.get_cc_major()
    cdef int cc_minor = common_data.get_cc_minor()
    cdef const char *cub_path = common_data.cub_path_get_c_str()
    cdef const char *thrust_path = common_data.thrust_path_get_c_str()
    cdef const char *libcudacxx_path = common_data.libcudacxx_path_get_c_str()
    cdef const char *ctk_path = common_data.ctk_path_get_c_str()
    with nogil:
        status = cccl_device_unique_by_key_compile(
            &self.build_data,
            d_keys_in.iter_data,
            d_values_in.iter_data,
            d_keys_out.iter_data,
            d_values_out.iter_data,
            d_num_selected_out.iter_data,
            comparison_op.op_data,
            cc_major,
            cc_minor,
            cub_path,
            thrust_path,
            libcudacxx_path,
            ctk_path,
            NULL,
        )
    if status != 0:
        raise RuntimeError(f"Failed compiling unique_by_key, error code: {status}")
    return self

def _unique_by_key_load(DeviceUniqueByKeyBuildResult self):
    cdef CUresult status = -1
    with nogil:
        status = cccl_device_unique_by_key_load(&self.build_data, NULL)
    if status != 0:
        raise RuntimeError(f"Failed loading unique_by_key, error code: {status}")

cdef extern from "cccl/c/merge_sort.h":
    cdef CUresult cccl_device_merge_sort_compile(
        cccl_device_merge_sort_build_result_t*,
        cccl_iterator_t,
        cccl_iterator_t,
        cccl_iterator_t,
        cccl_iterator_t,
        cccl_op_t,
        int, int, const char*, const char*, const char*, const char*,
        void*
    ) nogil
    cdef CUresult cccl_device_merge_sort_load(
        cccl_device_merge_sort_build_result_t*,
        const char*
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

def _merge_sort_serialize(DeviceMergeSortBuildResult self):
    cdef CUresult status = -1
    cdef void* buf_ptr = NULL
    cdef size_t buf_size = 0
    with nogil:
        status = cccl_device_merge_sort_serialize(&self.build_data, &buf_ptr, &buf_size)
    if status != 0:
        if buf_ptr != NULL:
            cccl_serialization_v2_buffer_free(buf_ptr)
        raise RuntimeError(f"Failed serializing merge_sort, error code: {status}")
    try:
        return PyBytes_FromStringAndSize(<const char*>buf_ptr, buf_size)
    finally:
        cccl_serialization_v2_buffer_free(buf_ptr)

def _merge_sort_deserialize(blob, load=True, check_cc=True):
    blob = bytes(blob)
    cdef DeviceMergeSortBuildResult self = DeviceMergeSortBuildResult.__new__(
        DeviceMergeSortBuildResult)
    cdef CUresult status = -1
    cdef const unsigned char[::1] view = blob
    cdef size_t buf_size = view.shape[0]
    if buf_size == 0:
        raise RuntimeError("Cannot deserialize empty blob")
    cdef const void* buf_ptr = <const void*>&view[0]
    with nogil:
        status = cccl_device_merge_sort_deserialize(
            &self.build_data, buf_ptr, buf_size)
    if status != 0:
        raise RuntimeError(f"Failed deserializing merge_sort, error code: {status}")
    if load:
        with nogil:
            status = cccl_device_merge_sort_load(&self.build_data, NULL)
        if status != 0:
            raise RuntimeError(f"Failed loading merge_sort after deserialize, error code: {status}")
        self._loaded = True
    return self

def _merge_sort_compile(
    Iterator d_in_keys,
    Iterator d_in_items,
    Iterator d_out_keys,
    Iterator d_out_items,
    Op op,
    CommonData common_data,
):
    cdef DeviceMergeSortBuildResult self = DeviceMergeSortBuildResult.__new__(DeviceMergeSortBuildResult)
    cdef CUresult status = -1
    cdef int cc_major = common_data.get_cc_major()
    cdef int cc_minor = common_data.get_cc_minor()
    cdef const char *cub_path = common_data.cub_path_get_c_str()
    cdef const char *thrust_path = common_data.thrust_path_get_c_str()
    cdef const char *libcudacxx_path = common_data.libcudacxx_path_get_c_str()
    cdef const char *ctk_path = common_data.ctk_path_get_c_str()
    with nogil:
        status = cccl_device_merge_sort_compile(
            &self.build_data,
            d_in_keys.iter_data,
            d_in_items.iter_data,
            d_out_keys.iter_data,
            d_out_items.iter_data,
            op.op_data,
            cc_major,
            cc_minor,
            cub_path,
            thrust_path,
            libcudacxx_path,
            ctk_path,
            NULL,
        )
    if status != 0:
        raise RuntimeError(f"Failed compiling merge_sort, error code: {status}")
    return self

def _merge_sort_load(DeviceMergeSortBuildResult self):
    cdef CUresult status = -1
    with nogil:
        status = cccl_device_merge_sort_load(&self.build_data, NULL)
    if status != 0:
        raise RuntimeError(f"Failed loading merge_sort, error code: {status}")

cdef extern from "cccl/c/radix_sort.h":
    cdef CUresult cccl_device_radix_sort_compile(
        cccl_device_radix_sort_build_result_t*,
        cccl_sort_order_t,
        cccl_iterator_t,
        cccl_iterator_t,
        cccl_op_t,
        const char*,
        int, int, const char*, const char*, const char*, const char*,
        void*
    ) nogil
    cdef CUresult cccl_device_radix_sort_load(
        cccl_device_radix_sort_build_result_t*,
        const char*
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

def _radix_sort_serialize(DeviceRadixSortBuildResult self):
    cdef CUresult status = -1
    cdef void* buf_ptr = NULL
    cdef size_t buf_size = 0
    with nogil:
        status = cccl_device_radix_sort_serialize(&self.build_data, &buf_ptr, &buf_size)
    if status != 0:
        if buf_ptr != NULL:
            cccl_serialization_v2_buffer_free(buf_ptr)
        raise RuntimeError(f"Failed serializing radix_sort, error code: {status}")
    try:
        return PyBytes_FromStringAndSize(<const char*>buf_ptr, buf_size)
    finally:
        cccl_serialization_v2_buffer_free(buf_ptr)

def _radix_sort_deserialize(blob, load=True, check_cc=True):
    blob = bytes(blob)
    cdef DeviceRadixSortBuildResult self = DeviceRadixSortBuildResult.__new__(
        DeviceRadixSortBuildResult)
    cdef CUresult status = -1
    cdef const unsigned char[::1] view = blob
    cdef size_t buf_size = view.shape[0]
    if buf_size == 0:
        raise RuntimeError("Cannot deserialize empty blob")
    cdef const void* buf_ptr = <const void*>&view[0]
    with nogil:
        status = cccl_device_radix_sort_deserialize(
            &self.build_data, buf_ptr, buf_size)
    if status != 0:
        raise RuntimeError(f"Failed deserializing radix_sort, error code: {status}")
    if load:
        with nogil:
            status = cccl_device_radix_sort_load(&self.build_data, NULL)
        if status != 0:
            raise RuntimeError(f"Failed loading radix_sort after deserialize, error code: {status}")
        self._loaded = True
    return self

def _radix_sort_compile(
    cccl_sort_order_t order,
    Iterator d_keys_in,
    Iterator d_values_in,
    Op decomposer_op,
    const char* decomposer_return_type,
    CommonData common_data,
):
    cdef DeviceRadixSortBuildResult self = DeviceRadixSortBuildResult.__new__(DeviceRadixSortBuildResult)
    cdef CUresult status = -1
    cdef int cc_major = common_data.get_cc_major()
    cdef int cc_minor = common_data.get_cc_minor()
    cdef const char *cub_path = common_data.cub_path_get_c_str()
    cdef const char *thrust_path = common_data.thrust_path_get_c_str()
    cdef const char *libcudacxx_path = common_data.libcudacxx_path_get_c_str()
    cdef const char *ctk_path = common_data.ctk_path_get_c_str()
    with nogil:
        status = cccl_device_radix_sort_compile(
            &self.build_data,
            order,
            d_keys_in.iter_data,
            d_values_in.iter_data,
            decomposer_op.op_data,
            decomposer_return_type,
            cc_major,
            cc_minor,
            cub_path,
            thrust_path,
            libcudacxx_path,
            ctk_path,
            NULL,
        )
    if status != 0:
        raise RuntimeError(f"Failed compiling radix_sort, error code: {status}")
    return self

def _radix_sort_load(DeviceRadixSortBuildResult self):
    cdef CUresult status = -1
    with nogil:
        status = cccl_device_radix_sort_load(&self.build_data, NULL)
    if status != 0:
        raise RuntimeError(f"Failed loading radix_sort, error code: {status}")

cdef extern from "cccl/c/transform.h":
    cdef CUresult cccl_device_unary_transform_compile(
        cccl_device_transform_build_result_t*,
        cccl_iterator_t,
        cccl_iterator_t,
        cccl_op_t,
        int, int, const char*, const char*, const char*, const char*,
        void*
    ) nogil
    cdef CUresult cccl_device_unary_transform_load(
        cccl_device_transform_build_result_t*,
        const char*
    ) nogil
    cdef CUresult cccl_device_unary_transform_serialize(
        const cccl_device_transform_build_result_t*,
        void**,
        size_t*
    ) nogil
    cdef CUresult cccl_device_unary_transform_deserialize(
        cccl_device_transform_build_result_t*,
        const void*,
        size_t
    ) nogil

def _unary_transform_serialize(DeviceUnaryTransform self):
    cdef CUresult status = -1
    cdef void* buf_ptr = NULL
    cdef size_t buf_size = 0
    with nogil:
        status = cccl_device_unary_transform_serialize(&self.build_data, &buf_ptr, &buf_size)
    if status != 0:
        if buf_ptr != NULL:
            cccl_serialization_v2_buffer_free(buf_ptr)
        raise RuntimeError(f"Failed serializing unary_transform, error code: {status}")
    try:
        return PyBytes_FromStringAndSize(<const char*>buf_ptr, buf_size)
    finally:
        cccl_serialization_v2_buffer_free(buf_ptr)

def _unary_transform_deserialize(blob, load=True, check_cc=True):
    blob = bytes(blob)
    cdef DeviceUnaryTransform self = DeviceUnaryTransform.__new__(
        DeviceUnaryTransform)
    cdef CUresult status = -1
    cdef const unsigned char[::1] view = blob
    cdef size_t buf_size = view.shape[0]
    if buf_size == 0:
        raise RuntimeError("Cannot deserialize empty blob")
    cdef const void* buf_ptr = <const void*>&view[0]
    with nogil:
        status = cccl_device_unary_transform_deserialize(
            &self.build_data, buf_ptr, buf_size)
    if status != 0:
        raise RuntimeError(f"Failed deserializing unary_transform, error code: {status}")
    if load:
        with nogil:
            status = cccl_device_unary_transform_load(&self.build_data, NULL)
        if status != 0:
            raise RuntimeError(f"Failed loading unary_transform after deserialize, error code: {status}")
        self._loaded = True
    return self

def _unary_transform_compile(
    Iterator d_in,
    Iterator d_out,
    Op op,
    CommonData common_data,
):
    cdef DeviceUnaryTransform self = DeviceUnaryTransform.__new__(DeviceUnaryTransform)
    cdef CUresult status = -1
    cdef int cc_major = common_data.get_cc_major()
    cdef int cc_minor = common_data.get_cc_minor()
    cdef const char *cub_path = common_data.cub_path_get_c_str()
    cdef const char *thrust_path = common_data.thrust_path_get_c_str()
    cdef const char *libcudacxx_path = common_data.libcudacxx_path_get_c_str()
    cdef const char *ctk_path = common_data.ctk_path_get_c_str()
    with nogil:
        status = cccl_device_unary_transform_compile(
            &self.build_data,
            d_in.iter_data,
            d_out.iter_data,
            op.op_data,
            cc_major,
            cc_minor,
            cub_path,
            thrust_path,
            libcudacxx_path,
            ctk_path,
            NULL,
        )
    if status != 0:
        raise RuntimeError(f"Failed compiling unary_transform, error code: {status}")
    return self

def _unary_transform_load(DeviceUnaryTransform self):
    cdef CUresult status = -1
    with nogil:
        status = cccl_device_unary_transform_load(&self.build_data, NULL)
    if status != 0:
        raise RuntimeError(f"Failed loading unary_transform, error code: {status}")

cdef extern from "cccl/c/transform.h":
    cdef CUresult cccl_device_binary_transform_compile(
        cccl_device_transform_build_result_t*,
        cccl_iterator_t,
        cccl_iterator_t,
        cccl_iterator_t,
        cccl_op_t,
        int, int, const char*, const char*, const char*, const char*,
        void*
    ) nogil
    cdef CUresult cccl_device_binary_transform_load(
        cccl_device_transform_build_result_t*,
        const char*
    ) nogil
    cdef CUresult cccl_device_binary_transform_serialize(
        const cccl_device_transform_build_result_t*,
        void**,
        size_t*
    ) nogil
    cdef CUresult cccl_device_binary_transform_deserialize(
        cccl_device_transform_build_result_t*,
        const void*,
        size_t
    ) nogil

def _binary_transform_serialize(DeviceBinaryTransform self):
    cdef CUresult status = -1
    cdef void* buf_ptr = NULL
    cdef size_t buf_size = 0
    with nogil:
        status = cccl_device_binary_transform_serialize(&self.build_data, &buf_ptr, &buf_size)
    if status != 0:
        if buf_ptr != NULL:
            cccl_serialization_v2_buffer_free(buf_ptr)
        raise RuntimeError(f"Failed serializing binary_transform, error code: {status}")
    try:
        return PyBytes_FromStringAndSize(<const char*>buf_ptr, buf_size)
    finally:
        cccl_serialization_v2_buffer_free(buf_ptr)

def _binary_transform_deserialize(blob, load=True, check_cc=True):
    blob = bytes(blob)
    cdef DeviceBinaryTransform self = DeviceBinaryTransform.__new__(
        DeviceBinaryTransform)
    cdef CUresult status = -1
    cdef const unsigned char[::1] view = blob
    cdef size_t buf_size = view.shape[0]
    if buf_size == 0:
        raise RuntimeError("Cannot deserialize empty blob")
    cdef const void* buf_ptr = <const void*>&view[0]
    with nogil:
        status = cccl_device_binary_transform_deserialize(
            &self.build_data, buf_ptr, buf_size)
    if status != 0:
        raise RuntimeError(f"Failed deserializing binary_transform, error code: {status}")
    if load:
        with nogil:
            status = cccl_device_binary_transform_load(&self.build_data, NULL)
        if status != 0:
            raise RuntimeError(f"Failed loading binary_transform after deserialize, error code: {status}")
        self._loaded = True
    return self

def _binary_transform_compile(
    Iterator d_in1,
    Iterator d_in2,
    Iterator d_out,
    Op op,
    CommonData common_data,
):
    cdef DeviceBinaryTransform self = DeviceBinaryTransform.__new__(DeviceBinaryTransform)
    cdef CUresult status = -1
    cdef int cc_major = common_data.get_cc_major()
    cdef int cc_minor = common_data.get_cc_minor()
    cdef const char *cub_path = common_data.cub_path_get_c_str()
    cdef const char *thrust_path = common_data.thrust_path_get_c_str()
    cdef const char *libcudacxx_path = common_data.libcudacxx_path_get_c_str()
    cdef const char *ctk_path = common_data.ctk_path_get_c_str()
    with nogil:
        status = cccl_device_binary_transform_compile(
            &self.build_data,
            d_in1.iter_data,
            d_in2.iter_data,
            d_out.iter_data,
            op.op_data,
            cc_major,
            cc_minor,
            cub_path,
            thrust_path,
            libcudacxx_path,
            ctk_path,
            NULL,
        )
    if status != 0:
        raise RuntimeError(f"Failed compiling binary_transform, error code: {status}")
    return self

def _binary_transform_load(DeviceBinaryTransform self):
    cdef CUresult status = -1
    with nogil:
        status = cccl_device_binary_transform_load(&self.build_data, NULL)
    if status != 0:
        raise RuntimeError(f"Failed loading binary_transform, error code: {status}")

cdef extern from "cccl/c/histogram.h":
    cdef CUresult cccl_device_histogram_compile(
        cccl_device_histogram_build_result_t*,
        int,
        int,
        cccl_iterator_t,
        int,
        cccl_iterator_t,
        cccl_type_info,
        int64_t,
        int64_t,
        bint,
        int, int, const char*, const char*, const char*, const char*,
        void*
    ) nogil
    cdef CUresult cccl_device_histogram_load(
        cccl_device_histogram_build_result_t*,
        const char*
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

def _histogram_serialize(DeviceHistogramBuildResult self):
    cdef CUresult status = -1
    cdef void* buf_ptr = NULL
    cdef size_t buf_size = 0
    with nogil:
        status = cccl_device_histogram_serialize(&self.build_data, &buf_ptr, &buf_size)
    if status != 0:
        if buf_ptr != NULL:
            cccl_serialization_v2_buffer_free(buf_ptr)
        raise RuntimeError(f"Failed serializing histogram, error code: {status}")
    try:
        return PyBytes_FromStringAndSize(<const char*>buf_ptr, buf_size)
    finally:
        cccl_serialization_v2_buffer_free(buf_ptr)

def _histogram_deserialize(blob, load=True, check_cc=True):
    blob = bytes(blob)
    cdef DeviceHistogramBuildResult self = DeviceHistogramBuildResult.__new__(
        DeviceHistogramBuildResult)
    cdef CUresult status = -1
    cdef const unsigned char[::1] view = blob
    cdef size_t buf_size = view.shape[0]
    if buf_size == 0:
        raise RuntimeError("Cannot deserialize empty blob")
    cdef const void* buf_ptr = <const void*>&view[0]
    with nogil:
        status = cccl_device_histogram_deserialize(
            &self.build_data, buf_ptr, buf_size)
    if status != 0:
        raise RuntimeError(f"Failed deserializing histogram, error code: {status}")
    if load:
        with nogil:
            status = cccl_device_histogram_load(&self.build_data, NULL)
        if status != 0:
            raise RuntimeError(f"Failed loading histogram after deserialize, error code: {status}")
        self._loaded = True
    return self

def _histogram_compile(
    int num_channels,
    int num_active_channels,
    Iterator d_samples,
    int num_levels,
    Iterator d_histogram,
    TypeInfo level_type,
    int num_rows,
    int row_stride_samples,
    bint is_evenly_segmented,
    CommonData common_data,
):
    cdef DeviceHistogramBuildResult self = DeviceHistogramBuildResult.__new__(DeviceHistogramBuildResult)
    cdef CUresult status = -1
    cdef int cc_major = common_data.get_cc_major()
    cdef int cc_minor = common_data.get_cc_minor()
    cdef const char *cub_path = common_data.cub_path_get_c_str()
    cdef const char *thrust_path = common_data.thrust_path_get_c_str()
    cdef const char *libcudacxx_path = common_data.libcudacxx_path_get_c_str()
    cdef const char *ctk_path = common_data.ctk_path_get_c_str()
    with nogil:
        status = cccl_device_histogram_compile(
            &self.build_data,
            num_channels,
            num_active_channels,
            d_samples.iter_data,
            num_levels,
            d_histogram.iter_data,
            level_type.type_info,
            num_rows,
            row_stride_samples,
            is_evenly_segmented,
            cc_major,
            cc_minor,
            cub_path,
            thrust_path,
            libcudacxx_path,
            ctk_path,
            NULL,
        )
    if status != 0:
        raise RuntimeError(f"Failed compiling histogram, error code: {status}")
    return self

def _histogram_load(DeviceHistogramBuildResult self):
    cdef CUresult status = -1
    with nogil:
        status = cccl_device_histogram_load(&self.build_data, NULL)
    if status != 0:
        raise RuntimeError(f"Failed loading histogram, error code: {status}")

cdef extern from "cccl/c/binary_search.h":
    cdef CUresult cccl_device_binary_search_compile(
        cccl_device_binary_search_build_result_t*,
        cccl_binary_search_mode_t,
        cccl_iterator_t,
        cccl_iterator_t,
        cccl_iterator_t,
        cccl_op_t,
        int, int, const char*, const char*, const char*, const char*,
        void*
    ) nogil
    cdef CUresult cccl_device_binary_search_load(
        cccl_device_binary_search_build_result_t*,
        const char*
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

def _binary_search_serialize(DeviceBinarySearchBuildResult self):
    cdef CUresult status = -1
    cdef void* buf_ptr = NULL
    cdef size_t buf_size = 0
    with nogil:
        status = cccl_device_binary_search_serialize(&self.build_data, &buf_ptr, &buf_size)
    if status != 0:
        if buf_ptr != NULL:
            cccl_serialization_v2_buffer_free(buf_ptr)
        raise RuntimeError(f"Failed serializing binary_search, error code: {status}")
    try:
        return PyBytes_FromStringAndSize(<const char*>buf_ptr, buf_size)
    finally:
        cccl_serialization_v2_buffer_free(buf_ptr)

def _binary_search_deserialize(blob, load=True, check_cc=True):
    blob = bytes(blob)
    cdef DeviceBinarySearchBuildResult self = DeviceBinarySearchBuildResult.__new__(
        DeviceBinarySearchBuildResult)
    cdef CUresult status = -1
    cdef const unsigned char[::1] view = blob
    cdef size_t buf_size = view.shape[0]
    if buf_size == 0:
        raise RuntimeError("Cannot deserialize empty blob")
    cdef const void* buf_ptr = <const void*>&view[0]
    with nogil:
        status = cccl_device_binary_search_deserialize(
            &self.build_data, buf_ptr, buf_size)
    if status != 0:
        raise RuntimeError(f"Failed deserializing binary_search, error code: {status}")
    if load:
        with nogil:
            status = cccl_device_binary_search_load(&self.build_data, NULL)
        if status != 0:
            raise RuntimeError(f"Failed loading binary_search after deserialize, error code: {status}")
        self._loaded = True
    return self

def _binary_search_compile(
    cccl_binary_search_mode_t mode,
    Iterator d_data,
    Iterator d_values,
    Iterator d_out,
    Op op,
    CommonData common_data,
):
    cdef DeviceBinarySearchBuildResult self = DeviceBinarySearchBuildResult.__new__(DeviceBinarySearchBuildResult)
    cdef CUresult status = -1
    cdef int cc_major = common_data.get_cc_major()
    cdef int cc_minor = common_data.get_cc_minor()
    cdef const char *cub_path = common_data.cub_path_get_c_str()
    cdef const char *thrust_path = common_data.thrust_path_get_c_str()
    cdef const char *libcudacxx_path = common_data.libcudacxx_path_get_c_str()
    cdef const char *ctk_path = common_data.ctk_path_get_c_str()
    with nogil:
        status = cccl_device_binary_search_compile(
            &self.build_data,
            mode,
            d_data.iter_data,
            d_values.iter_data,
            d_out.iter_data,
            op.op_data,
            cc_major,
            cc_minor,
            cub_path,
            thrust_path,
            libcudacxx_path,
            ctk_path,
            NULL,
        )
    if status != 0:
        raise RuntimeError(f"Failed compiling binary_search, error code: {status}")
    return self

def _binary_search_load(DeviceBinarySearchBuildResult self):
    cdef CUresult status = -1
    with nogil:
        status = cccl_device_binary_search_load(&self.build_data, NULL)
    if status != 0:
        raise RuntimeError(f"Failed loading binary_search, error code: {status}")

cdef extern from "cccl/c/three_way_partition.h":
    cdef CUresult cccl_device_three_way_partition_compile(
        cccl_device_three_way_partition_build_result_t*,
        cccl_iterator_t,
        cccl_iterator_t,
        cccl_iterator_t,
        cccl_iterator_t,
        cccl_iterator_t,
        cccl_op_t,
        cccl_op_t,
        int, int, const char*, const char*, const char*, const char*,
        void*
    ) nogil
    cdef CUresult cccl_device_three_way_partition_load(
        cccl_device_three_way_partition_build_result_t*,
        const char*
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

def _three_way_partition_serialize(DeviceThreeWayPartitionBuildResult self):
    cdef CUresult status = -1
    cdef void* buf_ptr = NULL
    cdef size_t buf_size = 0
    with nogil:
        status = cccl_device_three_way_partition_serialize(&self.build_data, &buf_ptr, &buf_size)
    if status != 0:
        if buf_ptr != NULL:
            cccl_serialization_v2_buffer_free(buf_ptr)
        raise RuntimeError(f"Failed serializing three_way_partition, error code: {status}")
    try:
        return PyBytes_FromStringAndSize(<const char*>buf_ptr, buf_size)
    finally:
        cccl_serialization_v2_buffer_free(buf_ptr)

def _three_way_partition_deserialize(blob, load=True, check_cc=True):
    blob = bytes(blob)
    cdef DeviceThreeWayPartitionBuildResult self = DeviceThreeWayPartitionBuildResult.__new__(
        DeviceThreeWayPartitionBuildResult)
    cdef CUresult status = -1
    cdef const unsigned char[::1] view = blob
    cdef size_t buf_size = view.shape[0]
    if buf_size == 0:
        raise RuntimeError("Cannot deserialize empty blob")
    cdef const void* buf_ptr = <const void*>&view[0]
    with nogil:
        status = cccl_device_three_way_partition_deserialize(
            &self.build_data, buf_ptr, buf_size)
    if status != 0:
        raise RuntimeError(f"Failed deserializing three_way_partition, error code: {status}")
    if load:
        with nogil:
            status = cccl_device_three_way_partition_load(&self.build_data, NULL)
        if status != 0:
            raise RuntimeError(f"Failed loading three_way_partition after deserialize, error code: {status}")
        self._loaded = True
    return self

def _three_way_partition_compile(
    Iterator d_in,
    Iterator d_first_part_out,
    Iterator d_second_part_out,
    Iterator d_unselected_out,
    Iterator d_num_selected_out,
    Op select_first_part_op,
    Op select_second_part_op,
    CommonData common_data,
):
    cdef DeviceThreeWayPartitionBuildResult self = DeviceThreeWayPartitionBuildResult.__new__(DeviceThreeWayPartitionBuildResult)
    cdef CUresult status = -1
    cdef int cc_major = common_data.get_cc_major()
    cdef int cc_minor = common_data.get_cc_minor()
    cdef const char *cub_path = common_data.cub_path_get_c_str()
    cdef const char *thrust_path = common_data.thrust_path_get_c_str()
    cdef const char *libcudacxx_path = common_data.libcudacxx_path_get_c_str()
    cdef const char *ctk_path = common_data.ctk_path_get_c_str()
    with nogil:
        status = cccl_device_three_way_partition_compile(
            &self.build_data,
            d_in.iter_data,
            d_first_part_out.iter_data,
            d_second_part_out.iter_data,
            d_unselected_out.iter_data,
            d_num_selected_out.iter_data,
            select_first_part_op.op_data,
            select_second_part_op.op_data,
            cc_major,
            cc_minor,
            cub_path,
            thrust_path,
            libcudacxx_path,
            ctk_path,
            NULL,
        )
    if status != 0:
        raise RuntimeError(f"Failed compiling three_way_partition, error code: {status}")
    return self

def _three_way_partition_load(DeviceThreeWayPartitionBuildResult self):
    cdef CUresult status = -1
    with nogil:
        status = cccl_device_three_way_partition_load(&self.build_data, NULL)
    if status != 0:
        raise RuntimeError(f"Failed loading three_way_partition, error code: {status}")

cdef extern from "cccl/c/segmented_sort.h":
    cdef CUresult cccl_device_segmented_sort_compile(
        cccl_device_segmented_sort_build_result_t*,
        cccl_sort_order_t,
        cccl_iterator_t,
        cccl_iterator_t,
        cccl_iterator_t,
        cccl_iterator_t,
        int, int, const char*, const char*, const char*, const char*,
        void*
    ) nogil
    cdef CUresult cccl_device_segmented_sort_load(
        cccl_device_segmented_sort_build_result_t*,
        const char*
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

def _segmented_sort_serialize(DeviceSegmentedSortBuildResult self):
    cdef CUresult status = -1
    cdef void* buf_ptr = NULL
    cdef size_t buf_size = 0
    with nogil:
        status = cccl_device_segmented_sort_serialize(&self.build_data, &buf_ptr, &buf_size)
    if status != 0:
        if buf_ptr != NULL:
            cccl_serialization_v2_buffer_free(buf_ptr)
        raise RuntimeError(f"Failed serializing segmented_sort, error code: {status}")
    try:
        return PyBytes_FromStringAndSize(<const char*>buf_ptr, buf_size)
    finally:
        cccl_serialization_v2_buffer_free(buf_ptr)

def _segmented_sort_deserialize(blob, load=True, check_cc=True):
    blob = bytes(blob)
    cdef DeviceSegmentedSortBuildResult self = DeviceSegmentedSortBuildResult.__new__(
        DeviceSegmentedSortBuildResult)
    cdef CUresult status = -1
    cdef const unsigned char[::1] view = blob
    cdef size_t buf_size = view.shape[0]
    if buf_size == 0:
        raise RuntimeError("Cannot deserialize empty blob")
    cdef const void* buf_ptr = <const void*>&view[0]
    with nogil:
        status = cccl_device_segmented_sort_deserialize(
            &self.build_data, buf_ptr, buf_size)
    if status != 0:
        raise RuntimeError(f"Failed deserializing segmented_sort, error code: {status}")
    if load:
        with nogil:
            status = cccl_device_segmented_sort_load(&self.build_data, NULL)
        if status != 0:
            raise RuntimeError(f"Failed loading segmented_sort after deserialize, error code: {status}")
        self._loaded = True
    return self

def _segmented_sort_compile(
    cccl_sort_order_t order,
    Iterator d_keys_in,
    Iterator d_values_in,
    Iterator begin_offset_in,
    Iterator end_offset_in,
    CommonData common_data,
):
    cdef DeviceSegmentedSortBuildResult self = DeviceSegmentedSortBuildResult.__new__(DeviceSegmentedSortBuildResult)
    cdef CUresult status = -1
    cdef int cc_major = common_data.get_cc_major()
    cdef int cc_minor = common_data.get_cc_minor()
    cdef const char *cub_path = common_data.cub_path_get_c_str()
    cdef const char *thrust_path = common_data.thrust_path_get_c_str()
    cdef const char *libcudacxx_path = common_data.libcudacxx_path_get_c_str()
    cdef const char *ctk_path = common_data.ctk_path_get_c_str()
    with nogil:
        status = cccl_device_segmented_sort_compile(
            &self.build_data,
            order,
            d_keys_in.iter_data,
            d_values_in.iter_data,
            begin_offset_in.iter_data,
            end_offset_in.iter_data,
            cc_major,
            cc_minor,
            cub_path,
            thrust_path,
            libcudacxx_path,
            ctk_path,
            NULL,
        )
    if status != 0:
        raise RuntimeError(f"Failed compiling segmented_sort, error code: {status}")
    return self

def _segmented_sort_load(DeviceSegmentedSortBuildResult self):
    cdef CUresult status = -1
    with nogil:
        status = cccl_device_segmented_sort_load(&self.build_data, NULL)
    if status != 0:
        raise RuntimeError(f"Failed loading segmented_sort, error code: {status}")
