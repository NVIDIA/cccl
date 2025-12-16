// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___MEMORY_ADDRESSOF_H
#define _CUDA_STD___MEMORY_ADDRESSOF_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_COMPILER(CLANG, >=, 15) || _CCCL_COMPILER(GCC, >=, 12)
#  define _CCCL_HAS_BUILTIN_STD_ADDRESSOF() 1
#else // ^^^ has builtin std::addressof ^^^ / vvv no builtin std::addressof vvv
#  define _CCCL_HAS_BUILTIN_STD_ADDRESSOF() 0
#endif // ^^^ no builtin std::addressof ^^^

// nvcc warns about host only std::addressof being used in device code
#if _CCCL_CUDA_COMPILER(NVCC) && _CCCL_DEVICE_COMPILATION()
#  undef _CCCL_HAS_BUILTIN_STD_ADDRESSOF
#  define _CCCL_HAS_BUILTIN_STD_ADDRESSOF() 0
#endif // _CCCL_CUDA_COMPILER(NVCC) && _CCCL_DEVICE_COMPILATION()

// include minimal std:: headers
#if _CCCL_HAS_BUILTIN_STD_ADDRESSOF()
#  if _CCCL_HOST_STD_LIB(LIBSTDCXX) && _CCCL_HAS_INCLUDE(<bits/move.h>)
#    include <bits/move.h>
#  elif _CCCL_HOST_STD_LIB(LIBCXX) && _CCCL_HAS_INCLUDE(<__memory/addressof.h>)
#    include <__memory/addressof.h>
#  elif !_CCCL_COMPILER(NVRTC)
#    include <cuda/std/__cccl/memory_wrapper.h>
#  endif
#endif // _CCCL_HAS_BUILTIN_STD_ADDRESSOF()

#include <cuda/std/__cccl/prologue.h>

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_MSVC(4312) // warning C4312: 'type cast': conversion from '_Tp' to '_Tp *' of greater size

_CCCL_BEGIN_NAMESPACE_CUDA_STD

#if _CCCL_HAS_BUILTIN_STD_ADDRESSOF()

// The compiler treats ::std::addressof as a builtin function so it does not need to be
// instantiated and will be compiled away even at -O0.
using ::std::addressof;

#elif defined(_CCCL_BUILTIN_ADDRESSOF)

template <class _Tp>
[[nodiscard]] _CCCL_API _CCCL_NO_CFI constexpr _Tp* addressof(_Tp& __x) noexcept
{
  return _CCCL_BUILTIN_ADDRESSOF(__x);
}

template <class _Tp>
_Tp* addressof(const _Tp&&) noexcept = delete;

#else

template <class _Tp>
[[nodiscard]] _CCCL_API _CCCL_NO_CFI _Tp* addressof(_Tp& __x) noexcept
{
  return reinterpret_cast<_Tp*>(const_cast<char*>(&reinterpret_cast<const volatile char&>(__x)));
}

template <class _Tp>
_Tp* addressof(const _Tp&&) noexcept = delete;

#endif // defined(_CCCL_BUILTIN_ADDRESSOF)

_CCCL_END_NAMESPACE_CUDA_STD

_CCCL_DIAG_POP

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___MEMORY_ADDRESSOF_H
