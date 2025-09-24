// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/void_t.h>
#include <cuda/std/__utility/declval.h>

CUB_NAMESPACE_BEGIN

namespace detail
{
template <typename T, typename U, typename = void>
struct has_plus_operator : ::cuda::std::false_type
{};

template <typename T, typename U>
struct has_plus_operator<T, U, ::cuda::std::void_t<decltype(::cuda::std::declval<T>() + ::cuda::std::declval<U>())>>
    : ::cuda::std::true_type
{};

template <typename T, typename U>
constexpr bool has_plus_operator_v = has_plus_operator<T, U>::value;

// Helper function that advances a given iterator only if it supports being advanced by the given offset
template <typename IteratorT, typename OffsetT>
CUB_RUNTIME_FUNCTION _CCCL_VISIBILITY_HIDDEN _CCCL_FORCEINLINE IteratorT
advance_iterators_if_supported(IteratorT iter, [[maybe_unused]] OffsetT offset)
{
  if constexpr (has_plus_operator_v<IteratorT, OffsetT>)
  {
    // If operator+ is valid, advance the iterator.
    return iter + offset;
  }
  else
  {
    // Otherwise, return iter unmodified.
    return iter;
  }
}

template <typename T, typename U, typename = void>
struct has_add_assign_operator : ::cuda::std::false_type
{};

template <typename T, typename U>
struct has_add_assign_operator<T,
                               U,
                               ::cuda::std::void_t<decltype(::cuda::std::declval<T&>() += ::cuda::std::declval<U>())>>
    : ::cuda::std::true_type
{};

template <typename T, typename U>
constexpr bool has_add_assign_operator_v = has_add_assign_operator<T, U>::value;

// Helper function that advances a given iterator only if it supports being advanced by the given offset
template <typename IteratorT, typename OffsetT>
CUB_RUNTIME_FUNCTION _CCCL_VISIBILITY_HIDDEN _CCCL_FORCEINLINE void
advance_iterators_inplace_if_supported(IteratorT& iter, [[maybe_unused]] OffsetT offset)
{
  if constexpr (has_add_assign_operator_v<IteratorT, OffsetT>)
  {
    // If operator+ is valid, advance the iterator.
    iter += offset;
  }
}

// Helper function that checks whether all of the given iterators support the + operator with the given offset
template <typename OffsetT, typename... Iterators>
CUB_RUNTIME_FUNCTION _CCCL_VISIBILITY_HIDDEN _CCCL_FORCEINLINE bool
all_iterators_support_plus_operator(OffsetT /*offset*/, Iterators... /*iters*/)
{
  if constexpr ((has_plus_operator_v<Iterators, OffsetT> && ...))
  {
    return true;
  }
  else
  {
    return false;
  }
}

// Helper function that checks whether all of the given iterators support the + operator with the given offset
template <typename OffsetT, typename... Iterators>
CUB_RUNTIME_FUNCTION _CCCL_VISIBILITY_HIDDEN _CCCL_FORCEINLINE bool
all_iterators_support_add_assign_operator(OffsetT /*offset*/, Iterators... /*iters*/)
{
  if constexpr ((has_add_assign_operator_v<Iterators, OffsetT> && ...))
  {
    return true;
  }
  else
  {
    return false;
  }
}

} // namespace detail

CUB_NAMESPACE_END
