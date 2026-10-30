# v2 (cccl.c.parallel.v2, HostJIT) — cccl_op_code_type enum + string-to-enum
# helper. Selected at CMake configure time and configure_file'd to the build
# dir as `_bindings_op_code_type.pxi`. v2's types.h adds CCCL_OP_LLVM_IR.

cdef extern from "cccl/c/types.h":
    cdef enum cccl_op_code_type:
        CCCL_OP_LTOIR
        CCCL_OP_CPP_SOURCE
        CCCL_OP_LLVM_IR


cdef inline cccl_op_code_type _parse_code_type(str s) noexcept:
    if s == "llvm_ir":
        return CCCL_OP_LLVM_IR
    if s == "cpp_source":
        return CCCL_OP_CPP_SOURCE
    return CCCL_OP_LTOIR
