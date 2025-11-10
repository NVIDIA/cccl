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

#include <thrust/detail/function.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/tbb/detail/execution_policy.h>

#include <cuda/std/__iterator/advance.h>
#include <cuda/std/__iterator/distance.h>

#include <tbb/blocked_range.h>
#include <tbb/parallel_scan.h>

THRUST_NAMESPACE_BEGIN
namespace system::tbb::detail
{
namespace copy_if_detail
{
template <typename InputIterator1, typename InputIterator2, typename OutputIterator, typename Predicate, typename Size>
struct body
{
  InputIterator1 first;
  InputIterator2 stencil;
  OutputIterator result;
  thrust::detail::wrapped_function<Predicate, bool> pred;
  Size sum;

  body(InputIterator1 first, InputIterator2 stencil, OutputIterator result, Predicate pred)
      : first(first)
      , stencil(stencil)
      , result(result)
      , pred{pred}
      , sum(0)
  {}

  body(body& b, ::tbb::split)
      : first(b.first)
      , stencil(b.stencil)
      , result(b.result)
      , pred{b.pred}
      , sum(0)
  {}

  void operator()(const ::tbb::blocked_range<Size>& r, ::tbb::pre_scan_tag)
  {
    InputIterator2 iter = stencil + r.begin();

    for (Size i = r.begin(); i != r.end(); ++i, ++iter)
    {
      if (pred(*iter))
      {
        ++sum;
      }
    }
  }

  void operator()(const ::tbb::blocked_range<Size>& r, ::tbb::final_scan_tag)
  {
    InputIterator1 iter1 = first + r.begin();
    InputIterator2 iter2 = stencil + r.begin();
    OutputIterator iter3 = result + sum;

    for (Size i = r.begin(); i != r.end(); ++i, ++iter1, ++iter2)
    {
      if (pred(*iter2))
      {
        *iter3 = *iter1;
        ++sum;
        ++iter3;
      }
    }
  }

  void reverse_join(body& b)
  {
    sum = b.sum + sum;
  }

  void assign(body& b)
  {
    sum = b.sum;
  }
}; // end body
} // namespace copy_if_detail

template <typename InputIterator1, typename InputIterator2, typename OutputIterator, typename Predicate>
OutputIterator
copy_if(tag, InputIterator1 first, InputIterator1 last, InputIterator2 stencil, OutputIterator result, Predicate pred)
{
  using Size = thrust::detail::it_difference_t<InputIterator1>;
  using Body = typename copy_if_detail::body<InputIterator1, InputIterator2, OutputIterator, Predicate, Size>;

  Size n = ::cuda::std::distance(first, last);

  if (n != 0)
  {
    Body body(first, stencil, result, pred);
    ::tbb::parallel_scan(::tbb::blocked_range<Size>(0, n), body);
    ::cuda::std::advance(result, body.sum);
  }

  return result;
} // end copy_if()
} // namespace system::tbb::detail
THRUST_NAMESPACE_END
