//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___UTILITY_AS_CONST_H
#define _CUDA_STD___UTILITY_AS_CONST_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/add_const.h>

#if _CCCL_COMPILER(CLANG, >=, 15) || _CCCL_COMPILER(GCC, >=, 12)
#  define _CCCL_HAS_BUILTIN_STD_AS_CONST() 1
#else // ^^^ has builtin std::as_const ^^^ / vvv no builtin std::as_const vvv
#  define _CCCL_HAS_BUILTIN_STD_AS_CONST() 0
#endif // ^^^ no builtin std::as_const ^^^

// nvcc warns about host only std::as_const being used in device code
#if _CCCL_CUDA_COMPILER(NVCC) && _CCCL_DEVICE_COMPILATION()
#  undef _CCCL_HAS_BUILTIN_STD_AS_CONST
#  define _CCCL_HAS_BUILTIN_STD_AS_CONST() 0
#endif // _CCCL_CUDA_COMPILER(NVCC) && _CCCL_DEVICE_COMPILATION()

// include minimal std:: headers
#if _CCCL_HAS_BUILTIN_STD_AS_CONST()
#  if _CCCL_HOST_STD_LIB(LIBCXX) && _CCCL_HAS_INCLUDE(<__utility/as_const.h>)
#    include <__utility/as_const.h>
#  elif !_CCCL_COMPILER(NVRTC)
#    include <utility>
#  endif
#endif // _CCCL_HAS_BUILTIN_STD_AS_CONST()

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

#if _CCCL_HAS_BUILTIN_STD_AS_CONST()

// The compiler treats ::std::as_const as a builtin function so it does not need to be
// instantiated and will be compiled away even at -O0.
using ::std::as_const;

#else // ^^^ _CCCL_HAS_BUILTIN_STD_AS_CONST() ^^^ / vvv !_CCCL_HAS_BUILTIN_STD_AS_CONST() vvv

template <class _Tp>
[[nodiscard]] _CCCL_INTRINSIC _CCCL_API constexpr add_const_t<_Tp>& as_const(_Tp& __t) noexcept
{
  return __t;
}

template <class _Tp>
void as_const(const _Tp&&) = delete;

#endif // ^^^ !_CCCL_HAS_BUILTIN_STD_AS_CONST() ^^^

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___UTILITY_AS_CONST_H
