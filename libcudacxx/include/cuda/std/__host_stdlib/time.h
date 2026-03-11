//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___HOST_STDLIB_TIME_H
#define _CUDA_STD___HOST_STDLIB_TIME_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if !_CCCL_COMPILER(NVRTC)
#  include <time.h>

// Standard C++ library comes with it's own <time.h> C++ compatible header. However, if the include paths are jumbled,
// it might happen that the original C <time.h> is found first. This is a problem because C headers define many of the
// time functions as macros which would change our definitions. So, we check whether any of the functions are defined
// as a macro to distinguish the C++ copatibility header from the C header.
#  if defined(clock) || defined(difftime) || defined(mktime) || defined(time) || defined(asctime) || defined(ctime) \
    || defined(gmtime) || defined(localtime) || defined(strftime)
#    error \
      "libcu++ requires the C++ compatibility <time.h> header, not the C <time.h> header. Please, check your include paths."
#  endif // math functions defined as macros

#endif // !_CCCL_COMPILER(NVRTC)

#endif // _CUDA_STD___HOST_STDLIB_TIME_H
