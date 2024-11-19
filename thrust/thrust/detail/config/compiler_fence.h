/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <thrust/detail/preprocessor.h>

#if _CCCL_COMPILER(MSVC)
#  pragma message( \
    "warning: The functionality in this header is unsafe, deprecated, and will soon be removed. Use C++11 atomics instead.")
#else
#warning The functionality in this header is unsafe, deprecated, and will soon be removed. Use C++11 or C11 atomics instead.
#endif

// msvc case
#if _CCCL_COMPILER(MSVC)

#  ifndef _DEBUG

#    include <intrin.h>
#    pragma intrinsic(_ReadWriteBarrier)
#    define __thrust_compiler_fence() _ReadWriteBarrier()
#  else

#    define __thrust_compiler_fence() \
      do                              \
      {                               \
      } while (0)

#  endif // _DEBUG

// gcc case
#elif _CCCL_COMPILER(GCC)

#  if _CCCL_COMPILER(GCC, >=, 4, 2) // atomic built-ins were introduced ~4.2
#    define __thrust_compiler_fence() __sync_synchronize()
#  else
// allow the code to compile without any guarantees
#    define __thrust_compiler_fence() \
      do                              \
      {                               \
      } while (0)
#  endif // _CCCL_COMPILER(GCC, >=, 4, 2)

// unknown case
#elif _CCCL_COMPILER(CLANG)
#  define __thrust_compiler_fence() __sync_synchronize()
#else

// allow the code to compile without any guarantees
#  define __thrust_compiler_fence() \
    do                              \
    {                               \
    } while (0)

#endif
