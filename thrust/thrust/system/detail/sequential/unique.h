// SPDX-FileCopyrightText: Copyright (c) 2008-2013, NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*! \file unique.h
 *  \brief Sequential implementations of unique algorithms.
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
#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/detail/sequential/execution_policy.h>

THRUST_NAMESPACE_BEGIN
namespace system::detail::sequential
{
_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy, typename InputIterator, typename OutputIterator, typename BinaryPredicate>
_CCCL_HOST_DEVICE OutputIterator unique_copy(
  sequential::execution_policy<DerivedPolicy>&,
  InputIterator first,
  InputIterator last,
  OutputIterator output,
  BinaryPredicate binary_pred)
{
  using T = thrust::detail::it_value_t<InputIterator>;

  if (first != last)
  {
    T prev = *first;

    for (++first; first != last; ++first)
    {
      T temp = *first;

      if (!binary_pred(prev, temp))
      {
        *output = prev;

        ++output;

        prev = temp;
      }
    }

    *output = prev;
    ++output;
  }

  return output;
} // end unique_copy()

template <typename DerivedPolicy, typename ForwardIterator, typename BinaryPredicate>
_CCCL_HOST_DEVICE ForwardIterator unique(
  sequential::execution_policy<DerivedPolicy>& exec,
  ForwardIterator first,
  ForwardIterator last,
  BinaryPredicate binary_pred)
{
  // sequential unique_copy permits in-situ operation
  return sequential::unique_copy(exec, first, last, first, binary_pred);
} // end unique()

template <typename DerivedPolicy, typename ForwardIterator, typename BinaryPredicate>
_CCCL_HOST_DEVICE thrust::detail::it_difference_t<ForwardIterator> unique_count(
  sequential::execution_policy<DerivedPolicy>&, ForwardIterator first, ForwardIterator last, BinaryPredicate binary_pred)
{
  using T = thrust::detail::it_value_t<ForwardIterator>;
  thrust::detail::it_difference_t<ForwardIterator> count{};

  if (first != last)
  {
    count++;
    T prev = *first;

    for (++first; first != last; ++first)
    {
      T temp = *first;

      if (!binary_pred(prev, temp))
      {
        count++;
        prev = temp;
      }
    }
  }

  return count;
} // end unique()
} // namespace system::detail::sequential
THRUST_NAMESPACE_END
