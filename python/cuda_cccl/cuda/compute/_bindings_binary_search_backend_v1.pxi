# v1 (cccl.c.parallel, NVRTC) — binary_search build_result_t struct +
# uniform cubin-bytes helper. v1 nests a transform build_result and carries
# op-state metadata; v2 (sibling file) flattens to top-level cubin fields.

cdef extern from "cccl/c/binary_search.h":
    cdef struct cccl_device_binary_search_build_result_t 'cccl_device_binary_search_build_result_t':
        cccl_device_transform_build_result_t transform
        size_t op_state_size
        size_t op_state_alignment


cdef inline bytes _binary_search_cubin_bytes(
    cccl_device_binary_search_build_result_t* b,
):
    return PyBytes_FromStringAndSize(
        <const char*>b.transform.cubin, b.transform.cubin_size
    )
