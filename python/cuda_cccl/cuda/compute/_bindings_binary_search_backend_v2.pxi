# v2 (cccl.c.parallel.v2, HostJIT) — binary_search build_result_t struct +
# uniform cubin-bytes helper. v2 flattens cubin/cubin_size to top-level fields.

cdef extern from "cccl/c/binary_search.h":
    cdef struct cccl_device_binary_search_build_result_t 'cccl_device_binary_search_build_result_t':
        void* cubin
        size_t cubin_size


cdef inline bytes _binary_search_cubin_bytes(
    cccl_device_binary_search_build_result_t* b,
):
    return PyBytes_FromStringAndSize(
        <const char*>b.cubin, b.cubin_size
    )
