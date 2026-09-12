// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/*! \file copy.h
 *  \brief HPX implementation of copy/copy_n.
 */

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header
#include <thrust/system/hpx/detail/contiguous_iterator.h>
#include <thrust/system/hpx/detail/execution_policy.h>

#include <hpx/parallel/algorithms/copy.hpp>

THRUST_NAMESPACE_BEGIN
namespace system::hpx::detail
{
template <typename DerivedPolicy, typename InputIterator, typename OutputIterator>
OutputIterator copy(execution_policy<DerivedPolicy>& exec [[maybe_unused]],
                    InputIterator first,
                    InputIterator last,
                    OutputIterator result)
{
  if constexpr (::hpx::traits::is_forward_iterator_v<InputIterator>)
  {
    auto res = ::hpx::copy(
      hpx::detail::to_hpx_execution_policy(exec),
      ::thrust::try_unwrap_contiguous_iterator(first),
      ::thrust::try_unwrap_contiguous_iterator(last),
      ::thrust::try_unwrap_contiguous_iterator(result));
    return detail::rewrap_contiguous_iterator(res, result);
  }
  else
  {
    return ::hpx::copy(first, last, result);
  }
}

template <typename DerivedPolicy, typename InputIterator, typename Size, typename OutputIterator>
OutputIterator
copy_n(execution_policy<DerivedPolicy>& exec [[maybe_unused]], InputIterator first, Size n, OutputIterator result)
{
  if constexpr (::hpx::traits::is_forward_iterator_v<InputIterator>)
  {
    auto res = ::hpx::copy_n(
      hpx::detail::to_hpx_execution_policy(exec),
      ::thrust::try_unwrap_contiguous_iterator(first),
      n,
      ::thrust::try_unwrap_contiguous_iterator(result));
    return detail::rewrap_contiguous_iterator(res, result);
  }
  else
  {
    return ::hpx::copy_n(first, n, result);
  }
}
} // end namespace system::hpx::detail

THRUST_NAMESPACE_END
