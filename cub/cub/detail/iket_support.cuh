// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

// Experimental: Optional IKET (In-Kernel Event Tracing) instrumentation based on an internal prototype. Enabled if the
// header is present. Otherwise, instrumentation macros compile to nothing.
#if __has_include(<iket/iket_device_apis.cuh>)
#  include <iket/iket_device_apis.cuh>
#else
#  define CREATE_IKET_START_END_RANGE(EVENT_NAME)
#  define IKET_RANGE_START(EVENT_NAME)
#  define IKET_RANGE_END(EVENT_NAME)
#endif
