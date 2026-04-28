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
//!
//! @par Overview
//! \c always_true is a function object type whose \c operator() returns \c true for any set of arguments.
//! It is commonly used as a default predicate in algorithms that conditionally take a predicate, such as
//! \c cub::DeviceTransform and parallel STL copy/partition operations. It also serves as a sentinel type:
//! algorithms can check <tt>is_same_v<Predicate, cuda::always_true></tt> at compile time to select optimized
//! code paths when no real predicate is in use.
//!
//! @par Example
//! @code
//! #include <cuda/functional>
//!
//! cuda::always_true pred{};
//! assert(pred(1, 2, 3) == true);
//! assert(pred() == true);
//! @endcode
struct always_true
{
  template <typename... _Ts>
  [[nodiscard]] _CCCL_API constexpr bool operator()(_Ts&&...) const noexcept
  {
    return true;
  }
};

//! @brief Function object that always returns \c false regardless of the arguments passed.
//!
//! @par Overview
//! \c always_false is a function object type whose \c operator() returns \c false for any set of arguments.
//! It is the counterpart to \c always_true and can be used as a default predicate that rejects all elements.
//!
//! @par Example
//! @code
//! #include <cuda/functional>
//!
//! cuda::always_false pred{};
//! assert(pred(1, 2, 3) == false);
//! assert(pred() == false);
//! @endcode
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
