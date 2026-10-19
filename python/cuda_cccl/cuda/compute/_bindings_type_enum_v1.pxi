# v1 (cccl.c.parallel, NVRTC) — cccl_type_enum. Selected at CMake configure
# time and configure_file'd to the build dir as `_bindings_type_enum.pxi`.
# v1's types.h does not define the emulated floating-point types.

cdef extern from "cccl/c/types.h":
    cpdef enum cccl_type_enum:
        INT8 "CCCL_INT8"
        INT16 "CCCL_INT16"
        INT32 "CCCL_INT32"
        INT64 "CCCL_INT64"
        UINT8 "CCCL_UINT8"
        UINT16 "CCCL_UINT16"
        UINT32 "CCCL_UINT32"
        UINT64 "CCCL_UINT64"
        FLOAT16 "CCCL_FLOAT16"
        FLOAT32 "CCCL_FLOAT32"
        FLOAT64 "CCCL_FLOAT64"
        STORAGE "CCCL_STORAGE"
        BOOLEAN "CCCL_BOOLEAN"
        BFLOAT16 "CCCL_BFLOAT16"
