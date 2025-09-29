// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___NEW_LAUNDER_H
#define _CUDA_STD___NEW_LAUNDER_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/is_function.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/remove_cv.h>

#include <cuda/std/__cccl/prologue.h>

#if _CCCL_CHECK_BUILTIN(builtin_launder) || _CCCL_COMPILER(GCC, >=, 7) || _CCCL_COMPILER(MSVC)
#  define _CCCL_BUILTIN_LAUNDER(...) __builtin_launder(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_launder) || _CCCL_COMPILER(GCC, >=, 7) || _CCCL_COMPILER(MSVC)

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr _Tp* launder(_Tp* __p) noexcept
{
  static_assert(!is_function_v<_Tp>, "can't launder functions");
  static_assert(!is_same_v<void, remove_cv_t<_Tp>>, "can't launder cv-void");
#if defined(_CCCL_BUILTIN_LAUNDER)
  return _CCCL_BUILTIN_LAUNDER(__p);
#else
  return __p;
#endif // _CCCL_BUILTIN_LAUNDER
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___NEW_LAUNDER_H
