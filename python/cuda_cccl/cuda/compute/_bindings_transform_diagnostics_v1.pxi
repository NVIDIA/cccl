# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cdef extern from "cccl/c/transform_diagnostics.h":
    cdef const char* cccl_transform_last_error() nogil


cdef str _transform_diagnostic():
    return (<bytes> cccl_transform_last_error()).decode("utf-8", "replace")
