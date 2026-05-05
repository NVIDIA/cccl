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

#include <cuda/std/__algorithm/lower_bound.h>
#include <cuda/std/__algorithm/upper_bound.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/cstddef>
#include <cuda/std/tuple>

CUB_NAMESPACE_BEGIN

namespace detail::find
{
constexpr ::cuda::std::ptrdiff_t linear_lower_bound_threshold = 8;

template <typename RangeIteratorT, typename RangeNumItemsT, typename CompareOpT, typename Mode>
struct comp_wrapper_t
{
  RangeIteratorT first;
  RangeNumItemsT num_items;
  CompareOpT op;

  template <typename Value, typename Output>
  _CCCL_DEVICE _CCCL_FORCEINLINE void operator()(::cuda::std::tuple<Value, Output> args) const
  {
    using DifferenceT = ::cuda::std::iter_difference_t<RangeIteratorT>;
    const auto last   = first + static_cast<DifferenceT>(num_items);

    ::cuda::std::get<1>(args) = Mode::Invoke(first, last, ::cuda::std::get<0>(args), op);
  }
};

template <typename Mode, typename RangeIteratorT, typename RangeNumItemsT, typename CompareOpT>
_CCCL_HOST_DEVICE auto make_comp_wrapper(RangeIteratorT first, RangeNumItemsT num_items, CompareOpT comp)
{
  return comp_wrapper_t<RangeIteratorT, RangeNumItemsT, CompareOpT, Mode>{first, num_items, comp};
}

struct lower_bound
{
  template <typename RangeIteratorT, typename DifferenceT, typename T, typename CompareOpT>
  _CCCL_DEVICE _CCCL_FORCEINLINE static DifferenceT
  Linear(RangeIteratorT first, DifferenceT num_items, const T& value, CompareOpT comp)
  {
    DifferenceT retval = 0;
    for (DifferenceT i = 0; i < num_items; ++i)
    {
      retval += static_cast<DifferenceT>(comp(first[i], value));
    }

    return retval;
  }

  template <typename RangeIteratorT, typename T, typename CompareOpT>
  _CCCL_DEVICE _CCCL_FORCEINLINE static ::cuda::std::ptrdiff_t
  Invoke(RangeIteratorT first, RangeIteratorT last, const T& value, CompareOpT comp)
  {
    return ::cuda::std::lower_bound(first, last, value, comp) - first;
  }
};

struct upper_bound
{
  template <typename RangeIteratorT, typename DifferenceT, typename T, typename CompareOpT>
  _CCCL_DEVICE _CCCL_FORCEINLINE static DifferenceT
  Linear(RangeIteratorT first, DifferenceT num_items, const T& value, CompareOpT comp)
  {
    DifferenceT retval = 0;
    for (DifferenceT i = 0; i < num_items; ++i)
    {
      retval += static_cast<DifferenceT>(!comp(value, first[i]));
    }

    return retval;
  }

  template <typename RangeIteratorT, typename T, typename CompareOpT>
  _CCCL_DEVICE _CCCL_FORCEINLINE static ::cuda::std::ptrdiff_t
  Invoke(RangeIteratorT first, RangeIteratorT last, const T& value, CompareOpT comp)
  {
    return ::cuda::std::upper_bound(first, last, value, comp) - first;
  }
};

template <typename RangeIteratorT, typename RangeNumItemsT, typename CompareOpT, typename Mode>
struct binary_search_transform_op_t
{
  RangeIteratorT first;
  RangeNumItemsT num_items;
  CompareOpT op;

  template <typename Value>
  _CCCL_DEVICE _CCCL_FORCEINLINE ::cuda::std::ptrdiff_t operator()(const Value& value) const
  {
    using DifferenceT = ::cuda::std::iter_difference_t<RangeIteratorT>;
    const auto count  = static_cast<DifferenceT>(num_items);

    if (num_items <= static_cast<RangeNumItemsT>(linear_lower_bound_threshold))
    {
      return Mode::Linear(first, count, value, op);
    }

    return Mode::Invoke(first, first + count, value, op);
  }
};

template <typename Mode, typename RangeIteratorT, typename RangeNumItemsT, typename CompareOpT>
_CCCL_HOST_DEVICE auto make_binary_search_transform_op(RangeIteratorT first, RangeNumItemsT num_items, CompareOpT comp)
{
  return binary_search_transform_op_t<RangeIteratorT, RangeNumItemsT, CompareOpT, Mode>{first, num_items, comp};
}
} // namespace detail::find

CUB_NAMESPACE_END
