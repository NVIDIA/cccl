# v1 (cccl.c.parallel, NVRTC) — cccl_op_code_type enum + string-to-enum helper.
# Selected at CMake configure time and configure_file'd to the build dir as
# `_bindings_op_code_type.pxi`. v1's types.h does not define CCCL_OP_LLVM_IR.

cdef extern from "cccl/c/types.h":
    cdef enum cccl_op_code_type:
        CCCL_OP_LTOIR
        CCCL_OP_CPP_SOURCE


cdef inline cccl_op_code_type _parse_code_type(str s) noexcept:
    if s == "cpp_source":
        return CCCL_OP_CPP_SOURCE
    return CCCL_OP_LTOIR
