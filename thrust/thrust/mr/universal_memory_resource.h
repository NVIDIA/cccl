// SPDX-FileCopyrightText: Copyright (c) 2018-2020, NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <thrust/detail/config.h>

#ifndef CCCL_DISABLE_NVRTC_COMPATIBILITY_CHECK
#  if _CCCL_COMPILER(NVRTC)
#    error \
      "Including <thrust/mr/universal_memory_resource.h> is not supported when compiling with NVRTC, which supports device code only. You can define CCCL_DISABLE_NVRTC_COMPATIBILITY_CHECK to disable this warning."
#  endif // _CCCL_COMPILER(NVRTC)
#endif // CCCL_DISABLE_NVRTC_COMPATIBILITY_CHECK

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <thrust/mr/device_memory_resource.h>
