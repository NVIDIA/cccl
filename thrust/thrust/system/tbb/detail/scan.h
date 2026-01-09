// SPDX-FileCopyrightText: Copyright (c) 2008-2013, NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*! \file scan.h
 *  \brief TBB implementations of scan functions.
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
#include <thrust/detail/type_traits.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/tbb/detail/execution_policy.h>

#include <cuda/std/__functional/invoke.h>
#include <cuda/std/__iterator/advance.h>
#include <cuda/std/__iterator/distance.h>

#include <tbb/blocked_range.h>
#include <tbb/parallel_scan.h>

THRUST_NAMESPACE_BEGIN
namespace system::tbb::detail
{
namespace scan_detail
{
template <typename InputIterator, typename OutputIterator, typename BinaryFunction, typename ValueType, bool HasInit>
struct inclusive_body
{
  InputIterator input;
  OutputIterator output;
  thrust::detail::wrapped_function<BinaryFunction, ValueType> binary_op;
  ValueType sum;
  bool first_call;

  inclusive_body(InputIterator input, OutputIterator output, BinaryFunction binary_op, ValueType init)
      : input(input)
      , output(output)
      , binary_op{binary_op}
      , sum(init)
      , first_call(true)
  {}

  inclusive_body(inclusive_body& b, ::tbb::split)
      : input(b.input)
      , output(b.output)
      , binary_op{b.binary_op}
      , sum(b.sum)
      , first_call(true)
  {}

  template <typename Size>
  void operator()(const ::tbb::blocked_range<Size>& r, ::tbb::pre_scan_tag)
  {
    InputIterator iter = input + r.begin();

    ValueType temp = *iter;

    ++iter;

    for (Size i = r.begin() + 1; i != r.end(); ++i, ++iter)
    {
      temp = binary_op(temp, *iter);
    }

    if (first_call)
    {
      sum = temp;
    }
    else
    {
      sum = binary_op(sum, temp);
    }

    first_call = false;
  }

  template <typename Size>
  void operator()(const ::tbb::blocked_range<Size>& r, ::tbb::final_scan_tag)
  {
    InputIterator iter1  = input + r.begin();
    OutputIterator iter2 = output + r.begin();

    if (first_call)
    {
      if constexpr (HasInit)
      {
        *iter2 = sum = binary_op(sum, *iter1);
      }
      else
      {
        *iter2 = sum = *iter1;
      }
      ++iter1;
      ++iter2;
      for (Size i = r.begin() + 1; i != r.end(); ++i, ++iter1, ++iter2)
      {
        *iter2 = sum = binary_op(sum, *iter1);
      }
    }
    else
    {
      for (Size i = r.begin(); i != r.end(); ++i, ++iter1, ++iter2)
      {
        *iter2 = sum = binary_op(sum, *iter1);
      }
    }

    first_call = false;
  }

  void reverse_join(inclusive_body& b)
  {
    // Only accumulate this functor's partial sum if this functor has been
    // called at least once -- otherwise we'll over-count the initial value.
    if (!first_call)
    {
      sum = binary_op(b.sum, sum);
    }
  }

  void assign(inclusive_body& b)
  {
    sum = b.sum;
  }
};

template <typename InputIterator, typename OutputIterator, typename BinaryFunction, typename ValueType>
struct exclusive_body
{
  InputIterator input;
  OutputIterator output;
  thrust::detail::wrapped_function<BinaryFunction, ValueType> binary_op;
  ValueType sum;
  bool first_call;

  exclusive_body(InputIterator input, OutputIterator output, BinaryFunction binary_op, ValueType init)
      : input(input)
      , output(output)
      , binary_op{binary_op}
      , sum(init)
      , first_call(true)
  {}

  exclusive_body(exclusive_body& b, ::tbb::split)
      : input(b.input)
      , output(b.output)
      , binary_op{b.binary_op}
      , sum(b.sum)
      , first_call(true)
  {}

  template <typename Size>
  void operator()(const ::tbb::blocked_range<Size>& r, ::tbb::pre_scan_tag)
  {
    InputIterator iter = input + r.begin();

    ValueType temp = *iter;

    ++iter;

    for (Size i = r.begin() + 1; i != r.end(); ++i, ++iter)
    {
      temp = binary_op(temp, *iter);
    }

    if (first_call && r.begin() > 0)
    {
      sum = temp;
    }
    else
    {
      sum = binary_op(sum, temp);
    }

    first_call = false;
  }

  template <typename Size>
  void operator()(const ::tbb::blocked_range<Size>& r, ::tbb::final_scan_tag)
  {
    InputIterator iter1  = input + r.begin();
    OutputIterator iter2 = output + r.begin();

    for (Size i = r.begin(); i != r.end(); ++i, ++iter1, ++iter2)
    {
      ValueType temp = binary_op(sum, *iter1);
      *iter2         = sum;
      sum            = temp;
    }

    first_call = false;
  }

  void reverse_join(exclusive_body& b)
  {
    // Only accumulate this functor's partial sum if this functor has been
    // called at least once -- otherwise we'll over-count the initial value.
    if (!first_call)
    {
      sum = binary_op(b.sum, sum);
    }
  }

  void assign(exclusive_body& b)
  {
    sum = b.sum;
  }
};
} // namespace scan_detail

template <typename InputIterator, typename OutputIterator, typename BinaryFunction>
OutputIterator
inclusive_scan(tag, InputIterator first, InputIterator last, OutputIterator result, BinaryFunction binary_op)
{
  using namespace thrust::detail;

  // Use the input iterator's value type per https://wg21.link/P0571
  using ValueType = thrust::detail::it_value_t<InputIterator>;

  using Size = thrust::detail::it_difference_t<InputIterator>;
  Size n     = ::cuda::std::distance(first, last);

  if (n != 0)
  {
    using Body = typename scan_detail::inclusive_body<InputIterator, OutputIterator, BinaryFunction, ValueType, false>;
    Body scan_body(first, result, binary_op, *first);
    ::tbb::parallel_scan(::tbb::blocked_range<Size>(0, n), scan_body);
  }

  ::cuda::std::advance(result, n);

  return result;
}

template <typename InputIterator, typename OutputIterator, typename InitialValueType, typename BinaryFunction>
OutputIterator inclusive_scan(
  tag, InputIterator first, InputIterator last, OutputIterator result, InitialValueType init, BinaryFunction binary_op)
{
  using namespace thrust::detail;

  // Use the input iterator's value type and the initial value type per wg21.link/p2322
  using ValueType =
    typename ::cuda::std::__accumulator_t<BinaryFunction, thrust::detail::it_value_t<InputIterator>, InitialValueType>;

  using Size = thrust::detail::it_difference_t<InputIterator>;
  Size n     = ::cuda::std::distance(first, last);

  if (n != 0)
  {
    using Body = typename scan_detail::inclusive_body<InputIterator, OutputIterator, BinaryFunction, ValueType, true>;
    Body scan_body(first, result, binary_op, init);
    ::tbb::parallel_scan(::tbb::blocked_range<Size>(0, n), scan_body);
  }

  ::cuda::std::advance(result, n);

  return result;
}

template <typename InputIterator, typename OutputIterator, typename InitialValueType, typename BinaryFunction>
OutputIterator exclusive_scan(
  tag, InputIterator first, InputIterator last, OutputIterator result, InitialValueType init, BinaryFunction binary_op)
{
  using namespace thrust::detail;

  // Use the initial value type per https://wg21.link/P0571
  using ValueType = InitialValueType;

  using Size = thrust::detail::it_difference_t<InputIterator>;
  Size n     = ::cuda::std::distance(first, last);

  if (n != 0)
  {
    using Body = typename scan_detail::exclusive_body<InputIterator, OutputIterator, BinaryFunction, ValueType>;
    Body scan_body(first, result, binary_op, init);
    ::tbb::parallel_scan(::tbb::blocked_range<Size>(0, n), scan_body);
  }

  ::cuda::std::advance(result, n);

  return result;
}
} // end namespace system::tbb::detail
THRUST_NAMESPACE_END
