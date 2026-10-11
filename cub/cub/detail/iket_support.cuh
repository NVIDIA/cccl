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

// Experimental: optional hooks for external device-side profiling instrumentation. If a compatible IKET (In-Kernel
// Event Tracing) header is present on the include path, the _CCCL_IKET_* macros below bind to it. Otherwise, they
// compile to nothing, so code using them builds unmodified without any such header available.
#if defined(_CCCL_ENABLE_IKET) && __has_include(<iket/iket_device_apis.cuh>)
#  include <iket/iket_device_apis.cuh>
#  define _CCCL_IKET_CREATE_START_END_RANGE(EVENT_NAME) CREATE_IKET_START_END_RANGE(EVENT_NAME)
#  define _CCCL_IKET_RANGE_START(EVENT_NAME)            IKET_RANGE_START(EVENT_NAME)
#  define _CCCL_IKET_RANGE_END(EVENT_NAME)              IKET_RANGE_END(EVENT_NAME)
#  define _CCCL_IKET_CREATE_PUSH_POP_RANGE(EVENT_NAME)  CREATE_IKET_PUSH_POP_RANGE(EVENT_NAME)
#  define _CCCL_IKET_RANGE_PUSH(EVENT_NAME)             IKET_RANGE_PUSH(EVENT_NAME)
#  define _CCCL_IKET_RANGE_POP()                        IKET_RANGE_POP()
#else
#  define _CCCL_IKET_CREATE_START_END_RANGE(EVENT_NAME)
#  define _CCCL_IKET_RANGE_START(EVENT_NAME)
#  define _CCCL_IKET_RANGE_END(EVENT_NAME)
#  define _CCCL_IKET_CREATE_PUSH_POP_RANGE(EVENT_NAME)
#  define _CCCL_IKET_RANGE_PUSH(EVENT_NAME)
#  define _CCCL_IKET_RANGE_POP()
#endif
