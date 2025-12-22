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
#include <cuda/std/tuple>

CUB_NAMESPACE_BEGIN

namespace detail::find
{
template <typename RangeIteratorT, typename CompareOpT, typename Mode>
struct comp_wrapper_t
{
  RangeIteratorT first;
  RangeIteratorT last;
  CompareOpT op;

  template <typename Value, typename Output>
  _CCCL_DEVICE _CCCL_FORCEINLINE void operator()(::cuda::std::tuple<Value, Output> args) const
  {
    ::cuda::std::get<1>(args) = Mode::Invoke(first, last, ::cuda::std::get<0>(args), op);
  }
};

template <typename Mode, typename RangeIteratorT, typename CompareOpT>
_CCCL_HOST_DEVICE auto make_comp_wrapper(RangeIteratorT first, RangeIteratorT last, CompareOpT comp)
{
  return comp_wrapper_t<RangeIteratorT, CompareOpT, Mode>{first, last, comp};
}

struct lower_bound
{
  template <typename RangeIteratorT, typename T, typename CompareOpT>
  _CCCL_DEVICE _CCCL_FORCEINLINE static RangeIteratorT
  Invoke(RangeIteratorT first, RangeIteratorT last, const T& value, CompareOpT comp)
  {
    return ::cuda::std::lower_bound(first, last, value, comp);
  }
};

struct upper_bound
{
  template <typename RangeIteratorT, typename T, typename CompareOpT>
  _CCCL_DEVICE _CCCL_FORCEINLINE static RangeIteratorT
  Invoke(RangeIteratorT first, RangeIteratorT last, const T& value, CompareOpT comp)
  {
    return ::cuda::std::upper_bound(first, last, value, comp);
  }
};
} // namespace detail::find

CUB_NAMESPACE_END
