//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX__DETAIL_CONFIG_CUH
#define __CUDAX__DETAIL_CONFIG_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

// Debuggers do not step into functions marked with __attribute__((__artificial__)).
// This is useful for small wrapper functions that just dispatch to other functions and
// that are inlined into the caller.
#if _CCCL_HAS_ATTRIBUTE(__artificial__) && !_CCCL_CUDACC
#  define _CUDAX_ARTIFICIAL __attribute__((__artificial__))
#else // ^^^ _CCCL_HAS_ATTRIBUTE(__artificial__) ^^^ / vvv !_CCCL_HAS_ATTRIBUTE(__artificial__) vvv
#  define _CUDAX_ARTIFICIAL
#endif // !_CCCL_HAS_ATTRIBUTE(__artificial__)

// _CCCL_HIDE_FROM_ABI and _CCCL_FORCEINLINE cannot be used together because they both
// try to add `inline` to the function declaration. In cudax we solve this problem with
// two attributes:
// - `_CUDAX_API` declares the function host/device and hides the symbol from the ABI
// - `_CUDAX_TRIVIAL_API` does the same while also forcing inlining and hiding the function from debuggers
#if _CCCL_COMPILER(ICC) // ICC has issues with visibility attributes on symbols with internal linkage
#  define _CUDAX_API        _CCCL_HOST_DEVICE
#  define _CUDAX_HOST_API   _CCCL_HOST
#  define _CUDAX_DEVICE_API _CCCL_DEVICE
#else // ^^^ _CCCL_COMPILER(ICC) ^^^ / vvv !_CCCL_COMPILER(ICC) vvv
#  define _CUDAX_API        _CCCL_HOST_DEVICE _CCCL_VISIBILITY_HIDDEN _CCCL_EXCLUDE_FROM_EXPLICIT_INSTANTIATION
#  define _CUDAX_HOST_API   _CCCL_HOST _CCCL_VISIBILITY_HIDDEN _CCCL_EXCLUDE_FROM_EXPLICIT_INSTANTIATION
#  define _CUDAX_DEVICE_API _CCCL_DEVICE _CCCL_VISIBILITY_HIDDEN _CCCL_EXCLUDE_FROM_EXPLICIT_INSTANTIATION
#endif // !_CCCL_COMPILER(ICC)

// _CUDAX_TRIVIAL_API force-inlines a function, marks its visibility as hidden, and causes debuggers to skip it.
// This is useful for trivial internal functions that do dispatching or other plumbing work. It is particularly
// useful in the definition of customization point objects.
#define _CUDAX_TRIVIAL_API        _CUDAX_API _CCCL_FORCEINLINE _CUDAX_ARTIFICIAL _LIBCUDACXX_NODEBUG
#define _CUDAX_TRIVIAL_HOST_API   _CUDAX_HOST_API _CCCL_FORCEINLINE _CUDAX_ARTIFICIAL _LIBCUDACXX_NODEBUG
#define _CUDAX_TRIVIAL_DEVICE_API _CUDAX_DEVICE_API _CCCL_FORCEINLINE _CUDAX_ARTIFICIAL _LIBCUDACXX_NODEBUG

// GCC struggles with guaranteed copy elision of immovable types.
#if _CCCL_COMPILER(GCC)
#  define _CUDAX_IMMOVABLE(_XP) _XP(_XP&&)
#else // ^^^ _CCCL_COMPILER(GCC) ^^^ / vvv !_CCCL_COMPILER(GCC) vvv
#  define _CUDAX_IMMOVABLE(_XP) _XP(_XP&&) = delete
#endif // !_CCCL_COMPILER(GCC)

namespace cuda::experimental
{
}

#endif // __CUDAX__DETAIL_CONFIG_CUH
