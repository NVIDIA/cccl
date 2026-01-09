// SPDX-FileCopyrightText: Copyright (c) 2008-2013, NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*! \file reduce.h
 *  \brief TBB implementation of reduce.
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
#include <thrust/detail/function.h>
#include <thrust/detail/static_assert.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/reduce.h>
#include <thrust/system/tbb/detail/execution_policy.h>

#include <cuda/std/__iterator/distance.h>

#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>

THRUST_NAMESPACE_BEGIN
namespace system::tbb::detail
{
namespace reduce_detail
{
template <typename RandomAccessIterator, typename OutputType, typename BinaryFunction>
struct body
{
  RandomAccessIterator first;
  OutputType sum;
  bool first_call; // TBB can invoke operator() multiple times on the same body
  thrust::detail::wrapped_function<BinaryFunction, OutputType> binary_op;

  // note: we only initialize sum with init to avoid calling OutputType's default constructor
  body(RandomAccessIterator first, OutputType init, BinaryFunction binary_op)
      : first(first)
      , sum(init)
      , first_call(true)
      , binary_op{binary_op}
  {}

  // note: we only initialize sum with b.sum to avoid calling OutputType's default constructor
  body(body& b, ::tbb::split)
      : first(b.first)
      , sum(b.sum)
      , first_call(true)
      , binary_op{b.binary_op}
  {}

  template <typename Size>
  void operator()(const ::tbb::blocked_range<Size>& r)
  {
    // we assume that blocked_range specifies a contiguous range of integers

    if (r.empty())
    {
      return; // nothing to do
    }

    RandomAccessIterator iter = first + r.begin();

    OutputType temp = thrust::raw_reference_cast(*iter);

    ++iter;

    for (Size i = r.begin() + 1; i != r.end(); ++i, ++iter)
    {
      temp = binary_op(temp, *iter);
    }

    if (first_call)
    {
      // first time body has been invoked
      first_call = false;
      sum        = temp;
    }
    else
    {
      // body has been previously invoked, accumulate temp into sum
      sum = binary_op(sum, temp);
    }
  } // end operator()()

  void join(body& b)
  {
    sum = binary_op(sum, b.sum);
  }
}; // end body
} // namespace reduce_detail

template <typename DerivedPolicy, typename InputIterator, typename OutputType, typename BinaryFunction>
OutputType reduce(
  execution_policy<DerivedPolicy>&, InputIterator begin, InputIterator end, OutputType init, BinaryFunction binary_op)
{
  using Size = thrust::detail::it_difference_t<InputIterator>;

  Size n = ::cuda::std::distance(begin, end);

  if (n == 0)
  {
    return init;
  }
  else
  {
    using Body = typename reduce_detail::body<InputIterator, OutputType, BinaryFunction>;
    Body reduce_body(begin, init, binary_op);
    ::tbb::parallel_reduce(::tbb::blocked_range<Size>(0, n), reduce_body);
    return binary_op(init, reduce_body.sum);
  }
}
} // end namespace system::tbb::detail
THRUST_NAMESPACE_END
