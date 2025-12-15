// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___UTILITY_FORWARD_H
#define _CUDA_STD___UTILITY_FORWARD_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/is_reference.h>
#include <cuda/std/__type_traits/remove_reference.h>
#include <cuda/std/cstddef>

#if _CCCL_COMPILER(CLANG, >=, 15) || _CCCL_COMPILER(GCC, >=, 12) || _CCCL_COMPILER(NVRTC)
#  define _CCCL_HAS_BUILTIN_STD_FORWARD() 1
#else // ^^^ has builtin std::forward ^^^ / vvv no builtin std::forward vvv
#  define _CCCL_HAS_BUILTIN_STD_FORWARD() 0
#endif // ^^^ no builtin std::forward ^^^

// nvcc always supports std::forward in device code.
#if _CCCL_CUDA_COMPILER(NVCC) && _CCCL_DEVICE_COMPILATION()
#  undef _CCCL_HAS_BUILTIN_STD_FORWARD
#  define _CCCL_HAS_BUILTIN_STD_FORWARD() 1
#endif // _CCCL_CUDA_COMPILER(NVCC) && _CCCL_DEVICE_COMPILATION()

// include minimal std:: headers, nvcc in device mode doesn't need the std:: header
#if _CCCL_HAS_BUILTIN_STD_FORWARD() && !(_CCCL_CUDA_COMPILER(NVCC) && _CCCL_DEVICE_COMPILATION())
#  if _CCCL_HOST_STD_LIB(LIBSTDCXX) && _CCCL_HAS_INCLUDE(<bits/move.h>)
#    include <bits/move.h>
#  elif _CCCL_HOST_STD_LIB(LIBCXX) && _CCCL_HAS_INCLUDE(<__utility/forward.h>)
#    include <__utility/forward.h>
#  elif !_CCCL_COMPILER(NVRTC)
#    include <utility>
#  endif
#endif // _CCCL_HAS_BUILTIN_STD_FORWARD() && !(_CCCL_CUDA_COMPILER(NVCC) && _CCCL_DEVICE_COMPILATION())

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

#if _CCCL_HAS_BUILTIN_STD_FORWARD()

// The compiler treats ::std::forward as a builtin function so it does not need to be instantiated and will be compiled
// away even at -O0.
using ::std::forward;

#else // ^^^ _CCCL_HAS_BUILTIN_STD_FORWARD() ^^^ / vvv !_CCCL_HAS_BUILTIN_STD_FORWARD() vvv

template <class _Tp>
[[nodiscard]] _CCCL_INTRINSIC _CCCL_API constexpr _Tp&& forward(remove_reference_t<_Tp>& __t) noexcept
{
  return static_cast<_Tp&&>(__t);
}

template <class _Tp>
[[nodiscard]] _CCCL_INTRINSIC _CCCL_API constexpr _Tp&& forward(remove_reference_t<_Tp>&& __t) noexcept
{
  static_assert(!is_lvalue_reference_v<_Tp>, "cannot forward an rvalue as an lvalue");
  return static_cast<_Tp&&>(__t);
}

#endif // ^^^ !_CCCL_HAS_BUILTIN_STD_FORWARD() ^^^

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___UTILITY_FORWARD_H
