# v2 (cccl.c.parallel.v2, HostJIT) — binary_search build_result_t struct +
# uniform cubin-bytes helper. v2 uses payload/payload_size matching v1.

cdef extern from "cccl/c/binary_search.h":
    cdef struct cccl_device_binary_search_build_result_t 'cccl_device_binary_search_build_result_t':
        void* payload
        size_t payload_size


cdef inline bytes _binary_search_cubin_bytes(
    cccl_device_binary_search_build_result_t* b,
):
    return PyBytes_FromStringAndSize(
        <const char*>b.payload, b.payload_size
    )
