//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL___UTILITY_RESULT_POLICY_CUH
#define _CUDA_EXPERIMENTAL___UTILITY_RESULT_POLICY_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cccl/prologue.h>

// NOLINTBEGIN(bugprone-reserved-identifier)

namespace cuda::experimental
{
template <class _Derived>
struct __result_policy_base
{
  using __derived_type _CCCL_NODEBUG_ALIAS = _Derived;
};

//! @brief Result specifier requesting that only a single rank receives the result.
//!
//! Passed as the leading argument to a cooperative or distributed algorithm to indicate
//! that only the rank `dest` receives the result. Every other participating rank receives
//! either no result or an unspecified value; the precise value seen by the non-destination
//! ranks is defined by the algorithm.
template <class _Tp>
struct returned_to : __result_policy_base<returned_to<_Tp>>
{
  _CCCL_HIDE_FROM_ABI returned_to() = delete;

  _CCCL_API constexpr explicit returned_to(_Tp __rank) noexcept
      : __rank_{__rank}
  {}

  _Tp __rank_;
};

template <class _Tp>
_CCCL_DEDUCTION_GUIDE_ATTRIBUTES returned_to(_Tp) -> returned_to<_Tp>;

struct distributed_t : __result_policy_base<distributed_t>
{};

//! @brief Result specifier requesting that each rank receives a slice of the result.
//!
//! Passed as the leading argument to a cooperative or distributed algorithm to indicate
//! that every participating rank receives a portion of the global result rather than the
//! whole. For example, a distributed sort delivers to each rank a slice of the globally
//! sorted sequence, and a distributed scan delivers to each rank its portion of the global
//! scan.
_CCCL_GLOBAL_CONSTANT distributed_t distributed{};

struct broadcasted_t : __result_policy_base<broadcasted_t>
{};

//! @brief Result specifier requesting that every rank receives an identical result.
//!
//! Passed as the leading argument to a cooperative or distributed algorithm to indicate
//! that all participating ranks receive the same complete result.
//!
//! @snippet this_block.cu broadcasted reduce
_CCCL_GLOBAL_CONSTANT broadcasted_t broadcasted{};
} // namespace cuda::experimental

// NOLINTEND(bugprone-reserved-identifier)

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___UTILITY_RESULT_POLICY_CUH
