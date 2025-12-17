// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___UTILITY_FORWARD_LIKE_H
#define _CUDA_STD___UTILITY_FORWARD_LIKE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/copy_cvref.h>
#include <cuda/std/__type_traits/remove_reference.h>

#if (_CCCL_COMPILER(CLANG, >=, 17) || _CCCL_COMPILER(GCC, >=, 15)) && __cpp_lib_forward_like >= 202217L
#  define _CCCL_HAS_BUILTIN_STD_FORWARD_LIKE() 1
#else // ^^^ has builtin std::forward_like ^^^ / vvv no builtin std::forward_like vvv
#  define _CCCL_HAS_BUILTIN_STD_FORWARD_LIKE() 0
#endif // ^^^ no builtin std::forward_like ^^^

// nvcc warns about host only std::forward_like being used in device code
#if _CCCL_CUDA_COMPILER(NVCC) && _CCCL_DEVICE_COMPILATION()
#  undef _CCCL_HAS_BUILTIN_STD_FORWARD_LIKE
#  define _CCCL_HAS_BUILTIN_STD_FORWARD_LIKE() 0
#endif // _CCCL_CUDA_COMPILER(NVCC) && _CCCL_DEVICE_COMPILATION()

// include minimal std:: headers
#if _CCCL_HAS_BUILTIN_STD_FORWARD_LIKE()
#  if _CCCL_HOST_STD_LIB(LIBSTDCXX) && _CCCL_HAS_INCLUDE(<bits/move.h>)
#    include <bits/move.h>
#  elif _CCCL_HOST_STD_LIB(LIBCXX) && _CCCL_HAS_INCLUDE(<__utility/forward_like.h>)
#    include <__utility/forward_like.h>
#  elif !_CCCL_COMPILER(NVRTC)
#    include <utility>
#  endif
#endif // _CCCL_HAS_BUILTIN_STD_FORWARD_LIKE()

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

#if _CCCL_HAS_BUILTIN_STD_FORWARD_LIKE()

// The compiler treats ::std::forward_like as a builtin function so it does not need to be
// instantiated and will be compiled away even at -O0.
using ::std::forward_like;

#else // ^^^ _CCCL_HAS_BUILTIN_STD_FORWARD_LIKE() ^^^ / vvv !_CCCL_HAS_BUILTIN_STD_FORWARD_LIKE() vvv

template <class _Ap, class _Bp>
using _ForwardLike = __copy_cvref_t<_Ap&&, remove_reference_t<_Bp>>;

template <class _Tp, class _Up>
[[nodiscard]] _CCCL_INTRINSIC _CCCL_API constexpr auto forward_like(_Up&& __ux) noexcept -> _ForwardLike<_Tp, _Up>
{
  return static_cast<_ForwardLike<_Tp, _Up>>(__ux);
}

#endif // ^^^ !_CCCL_HAS_BUILTIN_STD_FORWARD_LIKE() ^^^

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___UTILITY_FORWARD_LIKE_H
