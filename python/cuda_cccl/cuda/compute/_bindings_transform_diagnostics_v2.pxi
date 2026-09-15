# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


cdef str _transform_diagnostic():
    # HostJIT does not expose transform diagnostics through the C API yet.
    return ""
