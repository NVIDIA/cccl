//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
#pragma once

#include <cuda/std/detail/__config>

namespace cuda::experimental::__async
{
}

// Debuggers do not step into functions marked with __attribute__((__artificial__)).
// This is useful for small wrapper functions that just dispatch to other functions and
// that are inlined into the caller.
#if __has_attribute(__artificial__) && !defined(__CUDACC__)
#  define _CUDAX_ARTIFICIAL __attribute__((__artificial__))
#else
#  define _CUDAX_ARTIFICIAL
#endif

#define _CUDAX_ALWAYS_INLINE _LIBCUDACXX_ALWAYS_INLINE _CUDAX_ARTIFICIAL _LIBCUDACXX_NODEBUG inline

// GCC struggles with guaranteed copy elision of immovable types.
#if defined(_CCCL_COMPILER_GCC)
#  define _CUDAX_IMMOVABLE(XP) XP(XP&&)
#else
#  define _CUDAX_IMMOVABLE(XP) XP(XP&&) = delete
#endif
