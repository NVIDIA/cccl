//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FUNCTIONAL_ALWAYS_PRED_H
#define _CUDA___FUNCTIONAL_ALWAYS_PRED_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__functional/address_stability.h>
#include <cuda/std/__type_traits/integral_constant.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

//! @brief A predicate that always returns false for any input
//! @details This functor can be used with algorithms requiring a predicate
//!          that never matches any elements.
struct always_false
{
  //! @brief Function call operator that always returns false
  //! @param ... Any number of arguments of any type (ignored)
  //! @return Always returns false
  template <typename... _Ts>
  _CCCL_API constexpr bool operator()(_Ts&&...) const noexcept
  {
    return false;
  }
};

//! @brief A predicate that always returns true for any input
//! @details This functor can be used with algorithms requiring a predicate
//!          that matches all elements. Critical for CUB optimizations where
//!          type identity enables specialized fast paths.
struct always_true
{
  //! @brief Function call operator that always returns true
  //! @param ... Any number of arguments of any type (ignored)
  //! @return Always returns true
  template <typename... _Ts>
  _CCCL_API constexpr bool operator()(_Ts&&...) const noexcept
  {
    return true;
  }
};

template <>
struct proclaims_copyable_arguments<::cuda::always_false> : ::cuda::std::true_type
{};

template <>
struct proclaims_copyable_arguments<::cuda::always_true> : ::cuda::std::true_type
{};

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___FUNCTIONAL_ALWAYS_PRED_H
