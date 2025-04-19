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

#include <cuda/std/type_traits>

CUB_NAMESPACE_BEGIN

// Options for specifying memory aliasing
enum class MayAlias
{
  Yes,
  No
};

// Options for specifying sorting order.
enum class SortOrder
{
  Ascending,
  Descending
};

// Options for specifying the behavior of the stream compaction algorithm.
enum class SelectImpl
{
  // Stream compaction, discarding rejected items. It's required that memory of input and output are disjoint.
  Select,
  // Stream compaction, discarding rejected items. Memory of the input may be identical to the memory of the output.
  SelectPotentiallyInPlace,
  // Partition, keeping rejected items. It's required that memory of input and output are disjoint.
  Partition
};

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
_CCCL_HOST_DEVICE IteratorT advance_iterators_if_supported(IteratorT iter, OffsetT offset)
{
  if constexpr (has_plus_operator_v<IteratorT, OffsetT>)
  {
    // If operator+ is valid, advance the iterator.
    return iter + offset;
  }
  else
  {
    // Otherwise, return iter unmodified.
    static_cast<void>(offset);
    return iter;
  }
}

// Helper function that checks whether all of the given iterators support the + operator with the given offset
template <typename OffsetT, typename... Iterators>
_CCCL_HOST_DEVICE bool all_iterators_support_plus_operator(OffsetT /*offset*/, Iterators... /*iters*/)
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

} // namespace detail

CUB_NAMESPACE_END
