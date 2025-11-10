// SPDX-FileCopyrightText: Copyright (c) 2008-2013, NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <thrust/detail/seq.h>
#include <thrust/detail/static_assert.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/tbb/detail/execution_policy.h>

#include <cuda/std/__iterator/distance.h>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

THRUST_NAMESPACE_BEGIN
namespace system::tbb::detail
{
namespace for_each_detail
{
template <typename RandomAccessIterator, typename Size, typename UnaryFunction>
struct body
{
  RandomAccessIterator m_first;
  UnaryFunction m_f;

  body(RandomAccessIterator first, UnaryFunction f)
      : m_first(first)
      , m_f(f)
  {}

  void operator()(const ::tbb::blocked_range<Size>& r) const
  {
    // we assume that blocked_range specifies a contiguous range of integers
    thrust::for_each_n(seq, m_first + r.begin(), r.size(), m_f);
  } // end operator()()
}; // end body

template <typename Size, typename RandomAccessIterator, typename UnaryFunction>
body<RandomAccessIterator, Size, UnaryFunction> make_body(RandomAccessIterator first, UnaryFunction f)
{
  return body<RandomAccessIterator, Size, UnaryFunction>(first, f);
} // end make_body()
} // namespace for_each_detail

template <typename DerivedPolicy, typename RandomAccessIterator, typename Size, typename UnaryFunction>
RandomAccessIterator for_each_n(execution_policy<DerivedPolicy>&, RandomAccessIterator first, Size n, UnaryFunction f)
{
  ::tbb::parallel_for(::tbb::blocked_range<Size>(0, n), for_each_detail::make_body<Size>(first, f));

  // return the end of the range
  return first + n;
} // end for_each_n

template <typename DerivedPolicy, typename RandomAccessIterator, typename UnaryFunction>
RandomAccessIterator
for_each(execution_policy<DerivedPolicy>& s, RandomAccessIterator first, RandomAccessIterator last, UnaryFunction f)
{
  return tbb::detail::for_each_n(s, first, ::cuda::std::distance(first, last), f);
} // end for_each()
} // end namespace system::tbb::detail
THRUST_NAMESPACE_END
