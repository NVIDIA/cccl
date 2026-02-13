//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___TYPE_TRAITS_ALWAYS_FALSE_H
#define _CUDA_STD___TYPE_TRAITS_ALWAYS_FALSE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

// =============================================================================
// Inline variables for static_assert usage
// =============================================================================

//! @brief Always evaluates to false, useful for static_assert in template code
//! @details This variable template is used to create static_asserts that depend
//!          on template parameters, allowing errors to only trigger when the
//!          template is actually instantiated.
template <class...>
inline constexpr bool __always_false_v = false;

//! @brief Always evaluates to true, useful for static_assert in template code
template <class...>
inline constexpr bool __always_true_v = true;

// =============================================================================
// Predicate functors for algorithms
// =============================================================================

//! @brief A predicate that always returns false for any input
//! @details This functor can be used with algorithms requiring a predicate
//!          that never matches any elements.
struct always_false
{
  //! @brief Function call operator that always returns false
  //! @param ... Any number of arguments of any type (ignored)
  //! @return Always returns false
  template <typename... _Ts>
  _CCCL_HOST_DEVICE constexpr bool operator()(_Ts&&...) const noexcept
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
  _CCCL_HOST_DEVICE constexpr bool operator()(_Ts&&...) const noexcept
  {
    return true;
  }
};

_CCCL_END_NAMESPACE_CUDA_STD

// =============================================================================
// Optimization traits: Mark predicates as copyable for kernel argument passing
// =============================================================================

template <>
struct ::cuda::proclaims_copyable_arguments<_CUDA_VSTD::always_false> : ::cuda::std::true_type
{};

template <>
struct ::cuda::proclaims_copyable_arguments<_CUDA_VSTD::always_true> : ::cuda::std::true_type
{}

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___TYPE_TRAITS_ALWAYS_FALSE_H
