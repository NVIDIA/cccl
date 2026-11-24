# v2 (cccl.c.parallel.v2, HostJIT) — segmented_reduce extern + uniform call
# helper. v2 dropped `size_t max_segment_size`; the helper accepts it for
# signature-compatibility with v1 and silently ignores it.

cdef extern from "cccl/c/segmented_reduce.h":
    cdef CUresult cccl_device_segmented_reduce(
        cccl_device_segmented_reduce_build_result_t,
        void *,
        size_t *,
        cccl_iterator_t,
        cccl_iterator_t,
        uint64_t,
        cccl_iterator_t,
        cccl_iterator_t,
        cccl_op_t,
        cccl_value_t,
        CUstream
    ) nogil


cdef inline CUresult _call_segmented_reduce(
    cccl_device_segmented_reduce_build_result_t bld,
    void* storage_ptr,
    size_t* storage_sz,
    cccl_iterator_t d_in,
    cccl_iterator_t d_out,
    uint64_t num_items,
    cccl_iterator_t start_offsets,
    cccl_iterator_t end_offsets,
    cccl_op_t op_data,
    cccl_value_t init,
    size_t max_segment_size,
    CUstream stream,
) nogil:
    return cccl_device_segmented_reduce(
        bld, storage_ptr, storage_sz, d_in, d_out, num_items,
        start_offsets, end_offsets, op_data, init, stream
    )
