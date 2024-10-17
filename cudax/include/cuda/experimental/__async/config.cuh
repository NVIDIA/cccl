//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_ASYNC_DETAIL_CONFIG
#define __CUDAX_ASYNC_DETAIL_CONFIG

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

namespace cuda::experimental::__async
{
}

// Debuggers do not step into functions marked with __attribute__((__artificial__)).
// This is useful for small wrapper functions that just dispatch to other functions and
// that are inlined into the caller.
#if _CCCL_HAS_ATTRIBUTE(__artificial__) && !defined(__CUDACC__)
#  define _CUDAX_ARTIFICIAL __attribute__((__artificial__))
#else
#  define _CUDAX_ARTIFICIAL
#endif

#define _CUDAX_ALWAYS_INLINE _CCCL_FORCEINLINE _CUDAX_ARTIFICIAL _LIBCUDACXX_NODEBUG

// GCC struggles with guaranteed copy elision of immovable types.
#if defined(_CCCL_COMPILER_GCC)
#  define _CUDAX_IMMOVABLE(_XP) _XP(_XP&&)
#else
#  define _CUDAX_IMMOVABLE(_XP) _XP(_XP&&) = delete
#endif

#endif
