//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FUNCTIONAL_ALWAYS_TRUE_FALSE_H
#define _CUDA___FUNCTIONAL_ALWAYS_TRUE_FALSE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

//! @brief Function object that always returns \c true regardless of the arguments passed.
struct always_true
{
  template <typename... _Ts>
  [[nodiscard]] _CCCL_API constexpr bool operator()(_Ts&&...) const noexcept
  {
    return true;
  }
};

//! @brief Function object that always returns \c false regardless of the arguments passed.
struct always_false
{
  template <typename... _Ts>
  [[nodiscard]] _CCCL_API constexpr bool operator()(_Ts&&...) const noexcept
  {
    return false;
  }
};

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___FUNCTIONAL_ALWAYS_TRUE_FALSE_H
